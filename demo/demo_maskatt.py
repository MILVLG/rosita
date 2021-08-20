# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import sys
sys.path.append('./')
sys.path.append('rosita/')
import os, torch, datetime, random, copy, logging, argparse, cv2, yaml, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from rosita.data.load_data_mask_att import DataSet
from rosita.config.cfg import Cfg
from rosita.utils.segment import TextSegment
from rosita.modeling.transformer import LayerNorm, FeedForward, TextEmbeddings, VisualEmbeddings, Pooler


class DemoCfg(Cfg):
    def __init__(self, world_size, args):
        super(DemoCfg, self).__init__(world_size, args)
        self.MASK_SIDE_PROB = 0.5  # -1 means masking both sides simultaneously, the other means the probability of text side masking
        self.MASK_PROB = {'text': 0.15, 'image': 0.05}
        self.MASK_PROB_POST = {'mask': 0.8, 'replace': 0.1}

        self.MASK_STRUCT = {'tsg': True, 'tdt': False, 'isg': False, 'bbox': True}
        self.MASK_STRUCT_PROB = {'tsg': 0.3, 'tdt': 0.3, 'isg': 0.3, 'bbox': 0.3}
        self.MASK_STRUCT_PROB_INSIDE = {'tsg': 1.0, 'tdt': 1.0, 'isg': 1.0, 'bbox': 1.0}
        self.MASK_STRUCT_DIST = {
            'tsg': [],
            'tdt': [],
            'isg': [],
            'bbox': [0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.25,
                     0.25],
        }
        self.MASK_STRUCT_PRESERVE_OBJ = False
        self.OBJ_MASK_ATTMAP_IOU_THRESH = 0.1
        self.OBJ_MASK_ATTMAP_IOU_PROB = 0.
        self.OBJ_MASK_IOU_THRESH = 0.2
        self.OBJ_GRAIN_THRESH = 0.5
        self.OBJ_GRAIN_RATIO = 0.9


class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C
        self.dense_v = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.dense_k = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.dense_q = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.dense_merge = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.dense_v(v).view(n_batches, -1, self.__C.HHEAD, self.__C.HBASE).transpose(1, 2)
        k = self.dense_k(k).view(n_batches, -1, self.__C.HHEAD, self.__C.HBASE).transpose(1, 2)
        q = self.dense_q(q).view(n_batches, -1, self.__C.HHEAD, self.__C.HBASE).transpose(1, 2)

        atted, scores_out = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(n_batches, -1, self.__C.HSIZE)
        atted = self.dense_merge(atted)
        return atted, scores_out

    def att(self, value, key, query, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # bert style masking
        scores = scores + mask
        scores_out = scores.cpu().data
        att_map = torch.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value), scores_out


class SelfAtt(nn.Module):
    def __init__(self, __C):
        super(SelfAtt, self).__init__()
        self.mhatt = MHAtt(__C)
        model_dict = {
            'LAYER': __C.LAYER,
            'HSIZE': __C.HSIZE,
            'HHEAD': __C.HHEAD,
            'HBASE': __C.HBASE,
            'HFF': __C.HFF,
        }
        self.ffn = FeedForward(__C, model_dict)

        self.dropout0 = nn.Dropout(__C.DROPOUT_R)
        self.layer_norm0 = LayerNorm(__C.HSIZE, eps=1e-12)
        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.layer_norm1 = LayerNorm(__C.HSIZE, eps=1e-12)

    def forward(self, x, x_mask):
        att, scores_out = self.mhatt(x, x, x, x_mask)
        x = self.layer_norm0(x + self.dropout0(att))
        x = self.layer_norm1(x + self.dropout1(self.ffn(x)))
        return x, scores_out


class Backbone(nn.Module):
    def __init__(self, __C):
        super(Backbone, self).__init__()
        self.layers = nn.ModuleList([SelfAtt(__C) for _ in range(__C.LAYER)])

    def forward(self, x, x_mask):
        scores_out_list = []
        for layer in self.layers:
            x, scores_out= layer(x, x_mask)
            scores_out_list.append(scores_out)
        return x, scores_out_list


class Net(nn.Module):
    def __init__(self, __C, init_map):
        super(Net, self).__init__()
        self.__C = __C
        model_dict = {
            'LAYER': __C.LAYER,
            'HSIZE': __C.HSIZE,
            'HHEAD': __C.HHEAD,
            'HBASE': __C.HBASE,
            'HFF': __C.HFF,
        }

        self.text_embeddings = TextEmbeddings(__C, init_map['vocab_size'])
        self.visual_embeddings = VisualEmbeddings(__C)
        self.backbone = Backbone(__C)
        self.pooler = Pooler(__C)
        self.apply(self.init_weights)

    def forward(self, net_input):
        text_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox = net_input

        text_len = text_ids.size(1)
        text_embedded = self.text_embeddings(text_ids)
        text_mask = self.build_mask(text_mask)

        imgfeat_len = imgfeat_input.size(1)
        imgfeat_embedded = self.visual_embeddings(imgfeat_input, imgfeat_bbox)
        imgfeat_mask = self.build_mask(imgfeat_mask)

        x = torch.cat((text_embedded, imgfeat_embedded), dim=1)
        x_mask = torch.cat((text_mask, imgfeat_mask), dim=-1)

        x, scores_out_list = self.backbone(x, x_mask)

        net_output = scores_out_list
        return net_output

    def build_mask(self, mask):
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        return mask

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.__C.WEIGHT_INIT_FACTOR)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class Execution:
    def __init__(self, __C, RUN_MODE):
        self.__C = __C
        self.RUN_MODE = RUN_MODE
        torch.manual_seed(__C.SEED)
        torch.cuda.manual_seed(__C.SEED)
        torch.cuda.manual_seed_all(__C.SEED)
        np.random.seed(__C.SEED)
        random.seed(__C.SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=1e9)

    def eval(self, dataset, net=None):
        init_map = {
            'vocab_size': dataset.vocab_size,
        }

        if net is None:
            logging.info('Load Checkpoint')
            path = self.__C.CKPT_FILE
            logging.info('Loading the {}'.format(path))

            rank0_devices = [x - self.__C.LRANK * len(self.__C.DEVICE_IDS) for x in self.__C.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.__C.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            ckpt = torch.load(path, map_location=map_location)
            logging.info('Checkpoint Loaded')

            net = Net(self.__C, init_map)

            net.to(self.__C.DEVICE_IDS[0])
            net = DDP(net, device_ids=self.__C.DEVICE_IDS)

            for weight_key in self.__C.CKPT_SAVE_MAP:
                if weight_key not in ['net_optim', 'epoch']:
                    try:
                        getattr(net.module, weight_key).load_state_dict(ckpt[self.__C.CKPT_SAVE_MAP[weight_key]])
                    except Exception as e:
                        print(e)

        net.eval()

        mask_side = 'text'
        iter_id = 480
        mask_id = [2, 7]

        iter_id_ = random.randint(0, len(dataset)-1) if iter_id is None else iter_id

        with torch.no_grad():
            while True:
                quit_demo = False
                print(iter_id_)
                mask_id_ = mask_id if iter_id_ == iter_id else None
                text_input_ids, text_mask, text_mlm_label_ids, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, \
                mask_id_list, text_len, img_filename, boxes = dataset.get_item(iter_id_, mask_side, mask_id_)
                # if not len(mask_id_list):
                #     continue

                text_input_ids_batch = text_input_ids.to(self.__C.DEVICE_IDS[0]).unsqueeze(0)
                text_mask_batch = text_mask.to(self.__C.DEVICE_IDS[0]).unsqueeze(0)
                imgfeat_input_batch = imgfeat_input.to(self.__C.DEVICE_IDS[0]).unsqueeze(0)
                imgfeat_mask_batch = imgfeat_mask.to(self.__C.DEVICE_IDS[0]).unsqueeze(0)
                imgfeat_bbox_batch = imgfeat_bbox.to(self.__C.DEVICE_IDS[0]).unsqueeze(0)

                net_input = (text_input_ids_batch, text_mask_batch, imgfeat_input_batch, imgfeat_mask_batch, imgfeat_bbox_batch)
                net_output = net(net_input)

                scores_out_list = net_output

                text_input = [dataset.tokenizer.ids_to_tokens[t.item()] for t in text_input_ids][1: 1 + text_len]
                text_label = [dataset.tokenizer.ids_to_tokens[t.item()] for t in text_mlm_label_ids][1: 1 + text_len]
                img_root_path = 'path-to/mscoco/image2014/val2014/'
                img_path = img_root_path + img_filename
                print(f'image_path: {img_path}')

                def att_text(text_lbl, msk_id):
                    if msk_id != -1:
                        text_lbl[msk_id] = '|[{}]|'.format(text_lbl[msk_id])
                    text_att = ''
                    for i in range(len(text_lbl)):
                        if i == 0:
                            text_att += text_lbl[i]
                        elif text_lbl[i].startswith('#'):
                            text_att += text_lbl[i].replace('#', '')
                        else:
                            text_att += ' ' + text_lbl[i]
                    return text_att

                if mask_side == 'text':
                    weights_list = []
                    for idx in mask_id_list:
                        scores = copy.deepcopy(scores_out_list[-1][0, :, idx + 1, -36:].sum(-2))
                        weights = F.softmax(scores, dim=-1).numpy()
                        weights_list.append(weights)
                    if len(mask_id_list) == 0:
                        scores = copy.deepcopy(scores_out_list[-1][0, :, :-36, -36:].sum(-3).sum(-2))
                        weights = F.softmax(scores, dim=-1).numpy()
                        weights_list.append(weights)

                    weight_id = 0
                    while weight_id >= 0 and weight_id < len(weights_list):
                        weights = weights_list[weight_id]
                        if len(mask_id_list):
                            text_att = att_text(copy.deepcopy(text_label), mask_id_list[weight_id])
                        else:
                            text_att = att_text(copy.deepcopy(text_label), -1)
                        print(text_att)
                        img_att = cv2.imread(img_path)
                        Vv = 180.
                        max_imgfeat = 36
                        mask = np.zeros_like(img_att[:, :, 2], dtype=np.float32)
                        for ix, box in enumerate(boxes):
                            if ix == max_imgfeat:
                                break
                            x1 = int(box[0])
                            y1 = int(box[1])
                            x2 = int(box[2])
                            y2 = int(box[3])
                            mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2] + weights[ix]
                        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
                        mask = np.power(mask, 0.3)
                        picHSV = cv2.cvtColor(img_att, cv2.COLOR_BGR2HSV).astype(np.float32)
                        picHSV[:, :, 2] = picHSV[:, :, 2] - Vv
                        picHSV[:, :, 2] = picHSV[:, :, 2] + mask * Vv
                        picHSV[:, :, 2][picHSV[:, :, 2] < 0] = 0
                        picHSV[:, :, 2][picHSV[:, :, 2] > 250] = 250
                        img_att = cv2.cvtColor(picHSV.astype(np.uint8), cv2.COLOR_HSV2BGR)

                        cv2.imshow('attention image', img_att)
                        cmd = cv2.waitKey()
                        if cmd == 27:
                            quit_demo = True
                            break
                        elif cmd == ord('w'):
                            weight_id += 1
                        elif cmd == ord('s'):
                            weight_id -= 1
                        elif cmd == ord('c'):
                            save_file = 'demo/saved_att_img/{}.jpg'.format(\
                                str(iter_id_) + '_' + text_att).replace(' ', '_')
                            cv2.imwrite(save_file, img_att)
                            print('save to {}'.format(save_file))
                        # elif cmd == ord('f'):
                        #     new_iter_id = int(input('jumping to:'))
                        #     if new_iter_id < 0 or new_iter_id >= dataset.data_size:
                        #         print('invalid id:', new_iter_id)
                        #     else:
                        #         iter_id_ = new_iter_id
                        #         break

                        if weight_id == len(weights_list):
                            iter_id_ += 1
                            if iter_id_ == dataset.data_size:
                                iter_id_ = 0
                        elif weight_id < 0:
                            iter_id_ -= 1
                            if iter_id_ < 0:
                                iter_id_ = dataset.data_size - 1
                    if quit_demo:
                        break

                else:
                    img_att = cv2.imread(img_path)
                    for ix, idx in enumerate(mask_id_list):
                        cv2.rectangle(img_att, (boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3]), (0, 0, 255), thickness=2)
                        cv2.putText(img_att, f'{ix}-{idx}', (int(boxes[idx, 0]+5), int(boxes[idx, 1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=3)

                    weights_list = []
                    for idx in mask_id_list:
                        scores = copy.deepcopy(scores_out_list[-1][0, :, 20 + idx, 1: 1 + text_len].sum(-2))
                        weights = F.softmax(scores, dim=-1).numpy()
                        weights_list.append(weights)
                    if len(mask_id_list) == 0:
                        scores = copy.deepcopy(scores_out_list[-1][0, :, -36:, 1: 1 + text_len].sum(-3).sum(-2))
                        weights = F.softmax(scores, dim=-1).numpy()
                        weights_list.append(weights)

                    for weights in weights_list:
                        text_att = copy.deepcopy(text_label)
                        for ix in range(len(text_att)):
                            text_att[ix] = text_att[ix] + f"({format(weights[ix], '.2f')})"
                        print(text_att)
                    cv2.imshow('attention image', img_att)
                    cmd = cv2.waitKey()
                    if cmd == 27:
                        break
                    elif cmd == ord('w'):
                        iter_id_ += 1
                    elif cmd == ord('s'):
                        iter_id_ -= 1


    def run(self):
        spacy_tool = None
        eval_text_segment = None
        if self.__C.SEGMENT_TEXT:
            eval_text_segment = TextSegment(self.__C, self.__C.RUN_MODE)

        eval_dataset = DataSet(self.__C, self.__C.RUN_MODE, text_segment=eval_text_segment, spacy_tool=spacy_tool)

        self.eval(eval_dataset)


def mp_entrance(local_rank, world_size, args, __C):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    # initialize the process group
    global_rank = args.NODE_ID * world_size + local_rank
    __C.set_rank(global_rank, local_rank)
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size * args.NODE_SIZE, timeout=datetime.timedelta(minutes=120))

    exec = Execution(__C, __C.RUN_MODE)
    exec.run()


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--NS', dest='NODE_SIZE', default=1, type=int)
    parser.add_argument('--NI', dest='NODE_ID', default=0, type=int)
    parser.add_argument('--MA', dest='MASTER_ADDR', default='127.0.0.1', type=str)
    parser.add_argument('--MP', dest='MASTER_PORT', default='auto', type=str)
    parser.add_argument('--gpu', dest='GPU', default='0', type=str)
    parser.add_argument('--config', dest='config_file', default='configs/demo-maskatt.yaml', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    WORLD_SIZE = len(args.GPU.split(','))
    __C = DemoCfg(WORLD_SIZE, args)
    args_dict = __C.parse_to_dict(args)

    with open(args.config_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(yaml_dict)
    __C.proc(resume=False)
    print(__C)
    if args.MASTER_PORT == 'auto':
        args.MASTER_PORT = str(random.randint(13390, 17799))
    print('MASTER_ADDR:', args.MASTER_ADDR)
    print('MASTER_PORT:', args.MASTER_PORT)
    mp.spawn(
        mp_entrance,
        args=(WORLD_SIZE, args, __C),
        nprocs=WORLD_SIZE,
        join=True
    )