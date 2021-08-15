import torch, datetime, logging, random, copy
import torch.nn as nn
import torch.optim as Optim
import torch.multiprocessing as mp
import numpy as np

import os, argparse, yaml
from config.cfg import Cfg
from modeling.finetune_tasks.rec import Net

import torch.distributed as dist
from data.load_data_rec import DataSet
from utils.optimizer import BertAdam, WarmupOptimizer
from utils.sampler import SubsetDistributedSampler
from utils.segment import TextSegment
from utils.rec.bbox_transform import clip_boxes, bbox_transform_inv
from utils.rec.bbox import bbox_overlaps

from torch.nn.parallel import DistributedDataParallel as DDP


class Execution:
    def __init__(self, cfg, Net, RUN_MODE):
        self.cfg = cfg
        self.Net = Net
        self.RUN_MODE = RUN_MODE
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        torch.cuda.manual_seed_all(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        torch.backends.cudnn.benchmark = True
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=1e9)
        

    def get_optim(self, net, epoch_steps=None):
        if self.cfg.NET_OPTIM in ['warmup_adam']:
            net_optim = WarmupOptimizer(
                self.cfg.NET_LR_BASE,
                Optim.Adam(net.parameters(), lr=0, betas=self.cfg.OPT_BETAS, eps=self.cfg.OPT_EPS, weight_decay=self.cfg.NET_WEIGHT_DECAY),
                epoch_steps,
                warmup=self.cfg.NET_OPTIM_WARMUP,
                warmup_epochs=self.cfg.WARMUP_EPOCHS,
            )
            net_optim_inside = net_optim.optimizer

        elif self.cfg.NET_OPTIM in ['bert_adam']:
            net_optim = BertAdam(net.parameters(), lr=self.cfg.NET_LR_BASE,
                                 warmup=self.cfg.WARMUP_EPOCHS / self.cfg.OPTIM_EPOCHS,
                                 t_total=epoch_steps * self.cfg.OPTIM_EPOCHS, weight_decay=self.cfg.NET_WEIGHT_DECAY)

            net_optim_inside = net_optim

        return net_optim, net_optim_inside

    def train(self, train_loader, eval_loader):
        init_map = {
            'vocab_size': train_loader.dataset.vocab_size,
        }

        net = self.Net(self.cfg, init_map)
        net_optim, net_optim_inside = self.get_optim(net, epoch_steps=len(train_loader))

        net.to(self.cfg.DEVICE_IDS[0])
        net = DDP(net, device_ids=self.cfg.DEVICE_IDS)

        # Loading model weight
        if self.cfg.CKPT_LOAD:
            logging.info('Load Checkpoint')
            path = self.cfg.CKPT_FILE
            logging.info('Loading the {}'.format(path))

            rank0_devices = [x - self.cfg.LRANK * len(self.cfg.DEVICE_IDS) for x in self.cfg.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.cfg.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            ckpt = torch.load(path, map_location=map_location)
            logging.info('Checkpoint Loaded')

            for weight_key in self.cfg.CKPT_LOAD_MAP:
                if weight_key not in ['net_optim', 'epoch']:
                    weight_load = ckpt[self.cfg.CKPT_LOAD_MAP[weight_key]]
                    strict = True
                    getattr(net.module, weight_key).load_state_dict(weight_load, strict=strict)
                elif weight_key in ['net_optim']:
                    net_optim_inside.load_state_dict(ckpt[self.cfg.CKPT_LOAD_MAP[weight_key]])
                else:   # weight_key is epoch
                    self.cfg.CKPT_EPOCH = ckpt[self.cfg.CKPT_LOAD_MAP[weight_key]]

            if 'net_optim' in self.cfg.CKPT_LOAD_MAP:
                start_epoch = self.cfg.CKPT_EPOCH
                net_optim.set_start_step(start_epoch * len(train_loader))
            else:
                start_epoch = 0

        else:
            start_epoch = 0

        total_loss_sum = 0
        refs_rank_loss_sum = 0
        refs_reg_loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        for epoch in range(start_epoch, self.cfg.MAX_EPOCH):
            proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['ckpt'] else self.cfg.LRANK
            if proc_rank == 0:
                logfile = open(os.path.join(self.cfg.LOG_PATH, (self.cfg.VERSION + '.txt')), 'a+')
                logfile.write('nowTime: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
                logfile.close()

            train_loader.sampler.set_epoch(epoch)
            net.train()

            if epoch in self.cfg.NET_LR_DECAY_LIST:
                net_optim.decay(self.cfg.NET_LR_DECAY_R)

            for step, train_data in enumerate(train_loader):
                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                if step % 1000 == 0 and proc_rank == 0:
                    logging.info('[Epoch Trained: {:.2f} %][Lr: {:.7f}]'.format(step / len(train_loader) * 100.,
                                                                                np.array(net_optim.get_lr()).mean()))

                text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox, \
                refs_gt_boxes, refs_img_boxes, refs_img_shape, \
                refs_rank_label, refs_rank_weight, refs_rank_loss_valid, \
                refs_reg_label, refs_reg_weight, refs_reg_loss_valid = train_data

                text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])
                net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)

                # network step
                net_optim.zero_grad()
                net_output = net(net_input)
                pooled_output, text_output, imgfeat_output, pred_refs_rank, pred_refs_reg = net_output

                loss_input = \
                    pred_refs_rank, pred_refs_reg, \
                    refs_rank_label, refs_rank_weight, refs_rank_loss_valid, \
                    refs_reg_label, refs_reg_weight, refs_reg_loss_valid
                total_loss, losses = net.module.loss(loss_input)
                # for avoid backward the unused params
                total_loss += 0 * sum(p.sum() for p in net.parameters())

                total_loss.backward()

                total_loss_sum += total_loss.item()
                refs_rank_loss_sum += losses[0].item()
                refs_reg_loss_sum += losses[1].item()

                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                if step % 100 == 0 and proc_rank == 0:
                    logging.info(
                        '[epoch: {}][step: {} | {}] - [total: {:.4f}][refs_rank: {:.4f}][refs_reg: {:.4f}]'.format(
                            epoch + 1, step, len(train_loader), total_loss.item(), losses[0].item(), losses[1].item()))

                # gradient clipping
                if self.cfg.NET_GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), self.cfg.NET_GRAD_CLIP)
                net_optim.step()

            epoch_finish = epoch + 1
            proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['ckpt'] else self.cfg.LRANK
            if proc_rank == 0:
                state = {}
                for weight_key in self.cfg.CKPT_SAVE_MAP:
                    if weight_key not in ['net_optim', 'epoch']:
                        state[self.cfg.CKPT_SAVE_MAP[weight_key]] = getattr(net.module, weight_key).state_dict()
                    elif weight_key in ['net_optim']:
                        state[self.cfg.CKPT_SAVE_MAP[weight_key]] = net_optim_inside.state_dict()
                    else:   # weight_key is epoch
                        state[self.cfg.CKPT_SAVE_MAP[weight_key]] = epoch + 1

                save_model_path = os.path.join(self.cfg.CKPT_SAVE_PATH, (self.cfg.VERSION + '_epoch' + str(
                    epoch_finish) + '.pkl'))
                torch.save(state, save_model_path)
                last_model_path = os.path.join(self.cfg.CKPT_SAVE_PATH, 'last_ckpt.pkl')
                torch.save(state, last_model_path)

                logfile = open(os.path.join(self.cfg.LOG_PATH, (self.cfg.VERSION + '.txt')), 'a+')
                logfile.write(
                    '[epoch: {}][lr: {:.7f}]\n[total: {:.4f}][refs_rank: {:.4f}][refs_reg: {:.4f}]\n'.format(
                        epoch_finish, np.array(net_optim.get_lr()).mean(), total_loss_sum / len(train_loader),
                        refs_rank_loss_sum / len(train_loader), refs_reg_loss_sum / len(train_loader)))
                logfile.close()

            dist.barrier()

            total_loss_sum = 0
            refs_rank_loss_sum = 0
            refs_reg_loss_sum = 0
            grad_norm = np.zeros(len(named_params))

            if eval_loader is not None:
                self.eval(eval_loader, net=net)


    def eval(self, loader, net=None):
        init_map = {
            'vocab_size': loader.dataset.vocab_size,
        }

        if net is None:
            logging.info('Load Checkpoint')
            path = self.cfg.CKPT_FILE
            logging.info('Loading the {}'.format(path))

            rank0_devices = [x - self.cfg.LRANK * len(self.cfg.DEVICE_IDS) for x in self.cfg.DEVICE_IDS]
            device_pairs = zip(rank0_devices, self.cfg.DEVICE_IDS)
            map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
            ckpt = torch.load(path, map_location=map_location)
            logging.info('Checkpoint Loaded')

            net = self.Net(self.cfg, init_map)
            
            net.to(self.cfg.DEVICE_IDS[0])
            net = DDP(net, device_ids=self.cfg.DEVICE_IDS)

            for weight_key in self.cfg.CKPT_SAVE_MAP:
                if weight_key not in ['net_optim', 'epoch']:
                    getattr(net.module, weight_key).load_state_dict(ckpt[self.cfg.CKPT_SAVE_MAP[weight_key]])

        net.eval()
        loader.sampler.set_shuffle(False)

        with torch.no_grad():
            orin_reg_weight, orin_reg_bias = None, None
            if self.cfg.BBOX_NORM:
                for name, params in net.named_parameters():
                    if 'mm_refs_head.dense_reg.weight' in name:
                        orin_reg_weight = copy.deepcopy(params.data)
                        params.data = params.data * torch.unsqueeze(
                            torch.from_numpy(np.array(self.cfg.BBOX_NORM_STDS)).to(self.cfg.DEVICE_IDS[0]).float(), 1)
                    if 'mm_refs_head.dense_reg.bias' in name:
                        orin_reg_bias = copy.deepcopy(params.data)
                        params.data = params.data * torch.from_numpy(np.array(self.cfg.BBOX_NORM_STDS)).to(
                            self.cfg.DEVICE_IDS[0]).float() + torch.from_numpy(np.array(self.cfg.BBOX_NORM_MEANS)).to(
                            self.cfg.DEVICE_IDS[0]).float()

            conut_acc = 0
            conut = 0
            for step, data in enumerate(loader):
                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['eval'] else self.cfg.LRANK
                if step % 10 == 0 and proc_rank == 0:
                    logging.info('Evaluated [{:.2f} %]'.format(step / len(loader) * 100.))

                text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox, \
                refs_gt_boxes, refs_img_boxes, refs_img_shape, \
                refs_rank_label, refs_rank_weight, refs_rank_loss_valid, \
                refs_reg_label, refs_reg_weight, refs_reg_loss_valid = data

                text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])

                net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                net_output = net(net_input)
                pooled_output, text_output, imgfeat_output, pred_refs_rank, pred_refs_reg = net_output
                refs_gt_boxes, refs_img_boxes, refs_img_shape = refs_gt_boxes.numpy(), refs_img_boxes.numpy(), refs_img_shape.numpy()
                pred_refs_rank, pred_refs_reg = pred_refs_rank.cpu().data.numpy(), pred_refs_reg.cpu().data.numpy()

                bbox_reg = bbox_transform_inv(
                    refs_img_boxes.reshape(-1, 4), pred_refs_reg.reshape(-1, 4)).reshape(-1, self.cfg.PAD_MAX['image'], 4)
                arg_pred_refs_rank = np.argmax(pred_refs_rank, axis=1)

                for step_ix in range(pred_refs_rank.shape[0]):
                    cliped_bbox_reg_ix = clip_boxes(bbox_reg[step_ix], refs_img_shape[step_ix])
                    overlaps = bbox_overlaps(
                        np.ascontiguousarray(cliped_bbox_reg_ix[arg_pred_refs_rank[step_ix]][np.newaxis, :], dtype=np.float),
                        np.ascontiguousarray(refs_gt_boxes[step_ix], dtype=np.float))[:, 0]
                    conut += 1
                    conut_acc += int(overlaps[0] >= self.cfg.OVERLAP_THRESHOLD)

            if self.cfg.BBOX_NORM:
                for name, params in net.named_parameters():
                 if 'mm_refs_head.dense_reg.weight' in name:
                     params.data = orin_reg_weight
                 if 'mm_refs_head.dense_reg.bias' in name:
                     params.data = orin_reg_bias

            conut_acc = torch.tensor([conut_acc]).to(self.cfg.DEVICE_IDS[0])
            torch.distributed.all_reduce(conut_acc)
            conut_acc = conut_acc.item()

            conut = torch.tensor([conut]).to(self.cfg.DEVICE_IDS[0])
            torch.distributed.all_reduce(conut)
            conut = conut.item()

            accuracy = conut_acc / float(conut) * 100.
            proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['eval'] else self.cfg.LRANK
            if proc_rank == 0:
                logging.info("Overall Accuracy is: %.02f\n\n" % (accuracy))
                print('accuracy = ' + str(accuracy) + ' %')
                logfile = open(os.path.join(self.cfg.LOG_PATH, (self.cfg.VERSION + '.txt')), 'a+')
                logfile.write("Overall Accuracy is: %.02f\n\n" % (accuracy))
                logfile.close()

    def run(self):
        spacy_tool = None

        if self.RUN_MODE in ['train']:
            train_text_segment = None
            if self.cfg.SEGMENT_TEXT:
                train_text_segment = TextSegment(self.cfg, self.RUN_MODE)

            train_dataset = DataSet(self.cfg, self.RUN_MODE, text_segment=train_text_segment, spacy_tool=spacy_tool)
            train_sampler = SubsetDistributedSampler(train_dataset, shuffle=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.cfg.BATCH_SIZE,
                sampler=train_sampler,
                num_workers=self.cfg.NUM_WORKERS,
                drop_last=True
            )

            eval_text_segment = None
            eval_loader = None
            if self.cfg.EVAL_EVERY_EPOCH:
                if self.cfg.SEGMENT_TEXT:
                    eval_text_segment = TextSegment(self.cfg, 'val')

                eval_dataset = DataSet(self.cfg, 'val', text_segment=eval_text_segment, spacy_tool=spacy_tool)
                eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
                eval_loader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.cfg.EVAL_BATCH_SIZE,
                    sampler=eval_sampler,
                    num_workers=self.cfg.NUM_WORKERS
                )

            self.train(train_loader, eval_loader)


        elif self.RUN_MODE in ['val', 'test']:
            eval_text_segment = None
            if self.cfg.SEGMENT_TEXT:
                eval_text_segment = TextSegment(self.cfg, self.RUN_MODE)

            eval_dataset = DataSet(self.cfg, self.RUN_MODE, text_segment=eval_text_segment, spacy_tool=spacy_tool)
            eval_sampler = SubsetDistributedSampler(eval_dataset, shuffle=False)
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.cfg.EVAL_BATCH_SIZE,
                sampler=eval_sampler,
                num_workers=self.cfg.NUM_WORKERS
            )

            self.eval(eval_loader)


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

    WORLD_SIZE = len(args.GPU.split(','))
    __C = Cfg(WORLD_SIZE, args)
    args_dict = __C.parse_to_dict(args)

    with open(args.config_file, 'r') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(yaml_dict)
    __C.proc(args.resume)
    print(__C)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
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