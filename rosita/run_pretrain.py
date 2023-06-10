import yaml, os, torch, datetime, random, logging, argparse
import torch.nn as nn
import torch.optim as Optim
import numpy as np
from modeling.pretrain_tasks.rosita import Net
from config.cfg_pretrain import Cfg

import torch.distributed as dist
import torch.multiprocessing as mp

from data.load_data_pretrain import DataSet

from utils.optimizer import BertAdam, WarmupOptimizer
from utils.sampler import SubsetDistributedSampler
from utils.segment import TextSegment
from utils.weight_filter import qa_cls_weight_filter

import en_vectors_web_lg

try:
    import apex
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel as DDP
    # from apex.fp16_utils import *
    # from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError(
        'Please install apex from https://www.github.com/nvidia/apex to run this example, Or set <APEX> False.')
# assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'


class Execution:
    def __init__(self, __C, Net, RUN_MODE):
        self.__C = __C
        self.Net = Net
        self.RUN_MODE = RUN_MODE
        torch.manual_seed(__C.SEED)
        torch.cuda.manual_seed(__C.SEED)
        torch.cuda.manual_seed_all(__C.SEED)
        np.random.seed(__C.SEED)
        random.seed(__C.SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=1e9)

    def get_optim(self, net, epoch_steps=None):
        # no_decay = ['bias', 'layer_norm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.__C.NET_WEIGHT_DECAY},
        #     {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        # ]
        # filter(lambda p: p.requires_grad, net.parameters()),
        if self.__C.NET_OPTIM in ['warmup_adam']:
            net_optim = WarmupOptimizer(
                self.__C.NET_LR_BASE,
                Optim.Adam(net.parameters(), lr=0, betas=self.__C.OPT_BETAS, eps=self.__C.OPT_EPS, weight_decay=self.__C.NET_WEIGHT_DECAY),
                # Optim.Adam(optimizer_grouped_parameters, lr=0, betas=self.__C.OPT_BETAS, eps=self.__C.OPT_EPS),
                epoch_steps,
                warmup=self.__C.NET_OPTIM_WARMUP,
                warmup_epochs=self.__C.WARMUP_EPOCHS,
            )
            net_optim_inside = net_optim.optimizer

        elif self.__C.NET_OPTIM in ['bert_adam']:
            net_optim = BertAdam(net.parameters(), lr=self.__C.NET_LR_BASE,
                                 warmup=self.__C.WARMUP_EPOCHS / self.__C.OPTIM_EPOCHS,
                                 t_total=epoch_steps * self.__C.OPTIM_EPOCHS, weight_decay=self.__C.NET_WEIGHT_DECAY)
            # net_optim = BertAdam(optimizer_grouped_parameters, lr=self.__C.NET_LR_BASE, warmup=self.__C.WARMUP_EPOCHS / self.__C.OPTIM_EPOCHS, t_total=epoch_steps * self.__C.OPTIM_EPOCHS)

            net_optim_inside = net_optim

        return net_optim, net_optim_inside

    def train(self, train_loader, eval_loader):
        init_map = {
            'vocab_size': train_loader.dataset.vocab_size,
            'ans_size': train_loader.dataset.ans_size,
        }

        net = Net(self.__C, init_map)
        net_optim, net_optim_inside = self.get_optim(net, epoch_steps=len(train_loader))

        if self.__C.BN_SYNC:
            logging.info('Using Apex Synced BN')
            net = apex.parallel.convert_syncbn_model(net)
        net.to(self.__C.DEVICE_IDS[0])
        net, net_optim_inside = amp.initialize(net, optimizers=net_optim_inside, opt_level=self.__C.APEX_LEVEL,
                                                keep_batchnorm_fp32=self.__C.BN_FP32, loss_scale=None, num_losses=1)
        net = DDP(net, delay_allreduce=True)

        # Loading model weight
        if self.__C.CKPT_LOAD:
            logging.info('Load Checkpoint')
            path = self.__C.CKPT_FILE
            logging.info('Loading the {}'.format(path))

            rank0_devices = [x - self.__C.LRANK * len(self.__C.DEVICE_IDS) for x in self.__C.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.__C.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            ckpt = torch.load(path, map_location=map_location)
            logging.info('Checkpoint Loaded')

            for weight_key in self.__C.CKPT_LOAD_MAP:
                if weight_key not in ['net_optim', 'amp', 'epoch']:
                    weight_load = ckpt[self.__C.CKPT_LOAD_MAP[weight_key]]

                    strict = True
                    if weight_key in ['mm_qa_head']:
                        qa_cls_ans_tabel = DataSet.load_ans_vocab()
                        if self.__C.QA_CLS_WEIGHT_MACTH:
                            weight_load['dense1.weight'], weight_load['dense1.bias'] = qa_cls_weight_filter(
                                weight_load['dense1.weight'], weight_load['dense1.bias'], qa_cls_ans_tabel,
                                (train_loader.dataset.ans_to_ix, train_loader.dataset.ix_to_ans))
                        elif train_loader.dataset.ans_to_ix != qa_cls_ans_tabel[0]:
                            logging.info(
                                'answer tabels are not same and do not use qa cls weight match, will remove cls weight')
                            weight_load.pop('dense1.weight')
                            weight_load.pop('dense1.bias')
                            strict = False

                    getattr(net.module, weight_key).load_state_dict(weight_load, strict=strict)
                elif weight_key in ['net_optim']:
                        net_optim_inside.load_state_dict(ckpt[self.__C.CKPT_LOAD_MAP[weight_key]])
                elif weight_key in ['amp'] and self.__C.CKPT_LOAD_MAP[weight_key] in ckpt:
                    amp.load_state_dict(ckpt[self.__C.CKPT_LOAD_MAP[weight_key]])
                else:   # weight_key is epoch
                    self.__C.CKPT_EPOCH = ckpt[self.__C.CKPT_LOAD_MAP[weight_key]]

            if 'net_optim' in self.__C.CKPT_LOAD_MAP:
                start_epoch = self.__C.CKPT_EPOCH
                net_optim.set_start_step(start_epoch * len(train_loader))
            else:
                start_epoch = 0

        else:
            start_epoch = 0

        total_loss_sum = 0
        text_mlm_loss_sum = 0
        imgfeat_feat_loss_sum = 0
        imgfeat_obj_loss_sum = 0
        imgfeat_attr_loss_sum = 0
        mm_itm_loss_sum = 0
        mm_qa_loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        for epoch in range(start_epoch, self.__C.MAX_EPOCH):
            proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['ckpt'] else self.__C.LRANK
            if proc_rank == 0:
                logfile = open(os.path.join(self.__C.LOG_PATH, (self.__C.VERSION + '.txt')), 'a+')
                logfile.write('[epoch {} start time: '.format(epoch + 1) + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
                logfile.close()

            train_loader.sampler.set_epoch(epoch)
            net.train()

            if epoch in self.__C.NET_LR_DECAY_LIST:
                net_optim.decay(self.__C.NET_LR_DECAY_R)

            if 'qa' in self.__C.TASKS['mm'] and epoch == self.__C.LOSS_TRANSFER[0]:
                self.__C.LOSSFUNC_MAPPING['mm']['qa'] = self.__C.LOSS_TRANSFER[1]
                # train_loader.dataset.__C.LOSSFUNC_MAPPING['mm']['qa'] = self.__C.LOSS_TRANSFER[1]

            for step, train_data in enumerate(train_loader):
                proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['screen'] else self.__C.LRANK
                if step % 1000 == 0 and proc_rank == 0:
                    logging.info('[Epoch Trained: {:.2f} %][Lr: {:.7f}]'.format(step / len(train_loader) * 100.,
                                                                                np.array(net_optim.get_lr()).mean()))

                text_input_ids, text_mask, text_outside_weight, \
                text_mlm_label_ids, text_mlm_weight, text_mlm_loss_valid, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, imgfeat_outside_weight, \
                imgfeat_feat_label, imgfeat_feat_weight, imgfeat_feat_loss_valid, \
                imgfeat_obj_label, imgfeat_obj_weight, imgfeat_obj_loss_valid, \
                imgfeat_attr_label, imgfeat_attr_weight, imgfeat_attr_loss_valid, \
                itm_label, itm_loss_valid, qa_label, qa_loss_valid, text_id, img_id = train_data

                text_input_ids = text_input_ids.to(self.__C.DEVICE_IDS[0])
                text_mask = text_mask.to(self.__C.DEVICE_IDS[0])
                imgfeat_input = imgfeat_input.to(self.__C.DEVICE_IDS[0])
                imgfeat_mask = imgfeat_mask.to(self.__C.DEVICE_IDS[0])
                imgfeat_bbox = imgfeat_bbox.to(self.__C.DEVICE_IDS[0])
                net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)

                # network step
                net_optim.zero_grad()
                net_output = net(net_input)

                pooled_output, text_output, imgfeat_output, pred_text_mlm, \
                pred_imgfeat_feat, pred_imgfeat_obj, pred_imgfeat_attr, \
                pred_mm_itm, pred_mm_qa = net_output
                # print(pooled_output.size())

                loss_input = \
                    init_map, text_outside_weight, pred_text_mlm, text_mlm_label_ids, text_mlm_weight, text_mlm_loss_valid, \
                    imgfeat_outside_weight, pred_imgfeat_feat, pred_imgfeat_obj, pred_imgfeat_attr, \
                    imgfeat_feat_label, imgfeat_feat_weight, imgfeat_feat_loss_valid, \
                    imgfeat_obj_label, imgfeat_obj_weight, imgfeat_obj_loss_valid, \
                    imgfeat_attr_label, imgfeat_attr_weight, imgfeat_attr_loss_valid, \
                    pred_mm_itm, pred_mm_qa, itm_label, itm_loss_valid, qa_label, qa_loss_valid
                loss_ot_input = text_output, imgfeat_output, text_mask, imgfeat_mask
                total_loss, losses = net.module.loss(loss_input, loss_ot_input)
                # for avoid backward the unused params
                total_loss += 0 * sum(p.sum() for p in net.parameters())
                # print(total_loss)
                # print(losses)

                with amp.scale_loss(total_loss, net_optim_inside, loss_id=0) as scaled_loss:
                    scaled_loss.backward()

                total_loss_sum += total_loss.item()
                text_mlm_loss_sum += losses[0].item()
                imgfeat_feat_loss_sum += losses[1].item()
                imgfeat_obj_loss_sum += losses[2].item()
                imgfeat_attr_loss_sum += losses[3].item()
                mm_itm_loss_sum += losses[4].item()
                mm_qa_loss_sum += losses[5].item()

                proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['screen'] else self.__C.LRANK
                # if proc_rank == 0:
                if step % 100 == 0 and proc_rank == 0:
                    logging.info(
                        '[epoch: {}][step: {} | {}] - [total: {:.4f}][text_mlm: {:.4f}][img_feat: {:.4f}]'
                        '[img_obj: {:.4f}][img_attr: {:.4f}][mm_itm: {:.4f}][mm_qa: {:.4f}]'.format(
                            epoch + 1, step, len(train_loader), total_loss.item(), losses[0].item(), losses[1].item(),
                            losses[2].item(), losses[3].item(), losses[4].item(), losses[5].item()))

                # gradient clipping
                if self.__C.NET_GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(amp.master_params(net_optim_inside), self.__C.NET_GRAD_CLIP)
                net_optim.step()

                # # gradient check
                # for grad_wt in range(len(named_params)):
                #     norm_v = torch.norm(named_params[grad_wt][1].grad).cpu().data.numpy() \
                #         if named_params[grad_wt][1].grad is not None else 0
                #     grad_norm[grad_wt] += norm_v
                # break

            epoch_finish = epoch + 1
            proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['ckpt'] else self.__C.LRANK
            if proc_rank == 0:
                state = {}
                for weight_key in self.__C.CKPT_SAVE_MAP:
                    if weight_key not in ['net_optim', 'amp', 'epoch']:
                        state[self.__C.CKPT_SAVE_MAP[weight_key]] = getattr(net.module, weight_key).state_dict()
                    elif weight_key in ['net_optim']:
                            state[self.__C.CKPT_SAVE_MAP[weight_key]] = net_optim_inside.state_dict()
                    elif weight_key in ['amp']:
                        state[self.__C.CKPT_SAVE_MAP[weight_key]] = amp.state_dict()
                    else:   # weight_key is epoch
                        state[self.__C.CKPT_SAVE_MAP[weight_key]] = epoch + 1

                # if epoch_finish == self.__C.MAX_EPOCH:
                if epoch_finish == self.__C.MAX_EPOCH or epoch_finish % 10 == 0:
                    save_model_path = os.path.join(self.__C.CKPT_SAVE_PATH, (self.__C.VERSION + '_epoch' + str(
                        epoch_finish) + '.pkl'))
                    torch.save(state, save_model_path)
                last_model_path = os.path.join(self.__C.CKPT_SAVE_PATH, 'last_ckpt.pkl')
                torch.save(state, last_model_path)

                logfile = open(os.path.join(self.__C.LOG_PATH, (self.__C.VERSION + '.txt')), 'a+')
                logfile.write(
                    '[epoch: {}][lr: {:.7f}]\n[total: {:.4f}][text_mlm: {:.4f}][img_feat: {:.4f}]'
                    '[img_obj: {:.4f}][img_attr: {:.4f}][mm_itm: {:.4f}][mm_qa: {:.4f}]\n'.format(
                        epoch_finish, np.array(net_optim.get_lr()).mean(), total_loss_sum / len(train_loader),
                        text_mlm_loss_sum / len(train_loader), imgfeat_feat_loss_sum / len(train_loader),
                        imgfeat_obj_loss_sum / len(train_loader), imgfeat_attr_loss_sum / len(train_loader),
                        mm_itm_loss_sum / len(train_loader), mm_qa_loss_sum / len(train_loader)))
                logfile.close()

            dist.barrier()

            total_loss_sum = 0
            text_mlm_loss_sum = 0
            imgfeat_feat_loss_sum = 0
            imgfeat_obj_loss_sum = 0
            imgfeat_attr_loss_sum = 0
            mm_itm_loss_sum = 0
            mm_qa_loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    def run(self):
        spacy_tool = en_vectors_web_lg.load()

        if self.RUN_MODE in ['train']:
            train_text_segment = None
            if self.__C.SEGMENT_TEXT:
                train_text_segment = TextSegment(self.__C, self.RUN_MODE)

            train_dataset = DataSet(self.__C, self.RUN_MODE, text_segment=train_text_segment, spacy_tool=spacy_tool)
            train_sampler = SubsetDistributedSampler(train_dataset, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.__C.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=self.__C.NUM_WORKERS,
                drop_last=True
            )

            eval_text_segment = None
            eval_loader = None
            if self.__C.EVAL_EVERY_EPOCH:
                if self.__C.SEGMENT_TEXT:
                    eval_text_segment = TextSegment(self.__C, 'val')

                eval_dataset = DataSet(self.__C, 'val', text_segment=eval_text_segment, spacy_tool=spacy_tool)
                eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
                eval_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.__C.EVAL_BATCH_SIZE,
                    sampler=eval_sampler,
                    num_workers=self.__C.NUM_WORKERS
                )

            self.train(train_loader, eval_loader)


def mp_entrance(local_rank, world_size, args, __C):
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    # initialize the process group
    global_rank = args.NODE_ID * world_size + local_rank
    __C.set_rank(global_rank, local_rank)
    dist.init_process_group("nccl", rank=global_rank, world_size=world_size * args.NODE_SIZE, timeout=datetime.timedelta(minutes=120))
    
    exec = Execution(__C, Net, __C.RUN_MODE)
    exec.run()


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--NS', dest='NODE_SIZE', default=1, type=int)
    parser.add_argument('--NI', dest='NODE_ID', default=0, type=int)
    parser.add_argument('--MA', dest='MASTER_ADDR', default='127.0.0.1', type=str)
    parser.add_argument('--MP', dest='MASTER_PORT', default='auto', type=str)
    parser.add_argument('--gpu', dest='GPU', default='0, 1, 2, 3', type=str)
    parser.add_argument('--config', dest='config_file', default='', type=str)
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

    WORLD_SIZE = len(args.GPU.split(','))
    __C = Cfg(WORLD_SIZE, args)
    args_dict = __C.parse_to_dict(args)

    with open(args.config_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(yaml_dict)
    __C.proc(args.resume)
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
