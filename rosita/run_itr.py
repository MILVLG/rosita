# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 
 
import os, datetime, argparse
from modeling.finetune_tasks.itr import Net
from config.cfg import Cfg
import yaml
import torch, datetime, logging, random, math
import torch.nn as nn
import torch.optim as Optim
import torch.multiprocessing as mp
import numpy as np

import torch.distributed as dist
from data.load_data_itr import DataSet, DataSet_Neg
from utils.optimizer import BertAdam, WarmupOptimizer
from utils.sampler import SubsetDistributedSampler
from utils.segment import TextSegment

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

    def train(self, train_loader, eval_dataset, neg_text_loader, neg_img_loader):
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
        mm_itm_tri_loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        text_input_ids_all, text_mask_all, \
        imgfeat_input_all, imgfeat_mask_all, imgfeat_bbox_all = train_loader.dataset.load_all_data()
        all_eval_data = eval_dataset.load_all_data()
        
        for epoch in range(start_epoch, self.cfg.MAX_EPOCH):
            proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['ckpt'] else self.cfg.LRANK
            if proc_rank == 0:
                logfile = open(os.path.join(self.cfg.LOG_PATH, (self.cfg.VERSION + '.txt')), 'a+')
                logfile.write('[epoch {} start time: '.format(epoch + 1) + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']\n')
                logfile.close()

            if epoch % self.cfg.NEG_NEPOCH == 0 and epoch >= self.cfg.NEG_START_EPOCH:
                net.eval()
                with torch.no_grad():
                    # Hardest Negative Texts Mining
                    proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                    if proc_rank == 0:
                        logging.info('Hardest Negative Texts Mining')
                    neg_text_hard_ids_list = []
                    for step, data in enumerate(neg_text_loader):
                        proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                        if step % 10 == 0 and proc_rank == 0:
                            logging.info(f'Hardest Negative Texts Mining [{step / len(neg_text_loader) * 100}%]')

                        if self.cfg.IMGFEAT_FORMAT == 'npz':
                            text_idx, img_idx, neg_idx = data
                            text_idx = text_idx.view(-1)
                            img_idx = img_idx.view(-1)
                            text_input_ids = text_input_ids_all[text_idx, :]
                            text_mask = text_mask_all[text_idx, :]
                            imgfeat_input = imgfeat_input_all[img_idx, :]
                            imgfeat_mask = imgfeat_mask_all[img_idx, :]
                            imgfeat_bbox = imgfeat_bbox_all[img_idx, :]
                        else:
                            text_idx, imgfeat_input, imgfeat_mask, imgfeat_bbox, neg_idx = data
                            text_idx = text_idx.view(-1)
                            text_input_ids = text_input_ids_all[text_idx, :]
                            text_mask = text_mask_all[text_idx, :]
                            imgfeat_input = imgfeat_input.view(-1, imgfeat_input.size(2), imgfeat_input.size(3))
                            imgfeat_mask = imgfeat_mask.view(-1, imgfeat_mask.size(2))
                            imgfeat_bbox = imgfeat_bbox.view(-1, imgfeat_bbox.size(2), imgfeat_bbox.size(3))

                        text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                        text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])

                        net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                        pred_itm = net(net_input)[-1].view(-1, self.cfg.NEG_RANDSIZE, 2)
                        scores = torch.softmax(pred_itm, dim=-1)[:, :, 1]

                        arg_scores = torch.argsort(scores, dim=-1, descending=True)[:, :self.cfg.NEG_HARDSIZE]
                        arg_scores_bi = torch.arange(arg_scores.size(0)).unsqueeze(1).expand_as(arg_scores)
                        scores_ind = neg_idx[arg_scores_bi, arg_scores].to(scores.device)
                        neg_text_hard_ids_list.append(scores_ind)

                    neg_text_hard_ids_list = torch.cat(neg_text_hard_ids_list, dim=0)
                    neg_text_hard_ids_list_gather = [torch.zeros_like(neg_text_hard_ids_list.unsqueeze(1))
                                                     for _ in range(self.cfg.WORLD_SIZE * self.cfg.NODE_SIZE)]
                    dist.all_gather(neg_text_hard_ids_list_gather, neg_text_hard_ids_list.unsqueeze(1))
                    neg_text_hard_ids_list_gather = torch.cat(neg_text_hard_ids_list_gather, dim=1).view(-1, self.cfg.NEG_HARDSIZE).cpu()
                    neg_text_hard_ids_list_gather = neg_text_hard_ids_list_gather[:len(neg_img_loader.dataset), :]
                    train_loader.dataset.neg_text_hard_ids = neg_text_hard_ids_list_gather


                    # Hardest Negative Images Mining
                    proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                    if proc_rank == 0:
                        logging.info('Hardest Negative Images Mining')
                    neg_img_hard_ids_list = []
                    for step, data in enumerate(neg_img_loader):
                        proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                        if step % 10 == 0 and proc_rank == 0:
                            logging.info(f'Hardest Negative Images Mining [{step / len(neg_img_loader) * 100}%]')

                        if self.cfg.IMGFEAT_FORMAT == 'npz':
                            text_idx, img_idx, neg_idx = data

                            text_idx = text_idx.view(-1)
                            img_idx = img_idx.view(-1)
                            text_input_ids = text_input_ids_all[text_idx, :]
                            text_mask = text_mask_all[text_idx, :]
                            imgfeat_input = imgfeat_input_all[img_idx, :]
                            imgfeat_mask = imgfeat_mask_all[img_idx, :]
                            imgfeat_bbox = imgfeat_bbox_all[img_idx, :]
                        else:
                            text_idx, imgfeat_input, imgfeat_mask, imgfeat_bbox, neg_idx = data
                            text_idx = text_idx.view(-1)
                            text_input_ids = text_input_ids_all[text_idx, :]
                            text_mask = text_mask_all[text_idx, :]
                            imgfeat_input = imgfeat_input.view(-1, imgfeat_input.size(2), imgfeat_input.size(3))
                            imgfeat_mask = imgfeat_mask.view(-1, imgfeat_mask.size(2))
                            imgfeat_bbox = imgfeat_bbox.view(-1, imgfeat_bbox.size(2), imgfeat_bbox.size(3))

                        text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                        text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                        imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])

                        net_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                        pred_itm = net(net_input)[-1].view(-1, self.cfg.NEG_RANDSIZE, 2)
                        scores = torch.softmax(pred_itm, dim=-1)[:, :, 1]

                        arg_scores = torch.argsort(scores, dim=-1, descending=True)[:, :self.cfg.NEG_HARDSIZE]
                        arg_scores_bi = torch.arange(arg_scores.size(0)).unsqueeze(1).expand_as(arg_scores)
                        scores_ind = neg_idx[arg_scores_bi, arg_scores].to(scores.device)
                        neg_img_hard_ids_list.append(scores_ind)

                    neg_img_hard_ids_list = torch.cat(neg_img_hard_ids_list, dim=0)
                    neg_img_hard_ids_list_gather = [torch.zeros_like(neg_img_hard_ids_list.unsqueeze(1))
                                                     for _ in range(self.cfg.WORLD_SIZE * self.cfg.NODE_SIZE)]
                    dist.all_gather(neg_img_hard_ids_list_gather, neg_img_hard_ids_list.unsqueeze(1))
                    neg_img_hard_ids_list_gather = torch.cat(neg_img_hard_ids_list_gather, dim=1).view(-1, self.cfg.NEG_HARDSIZE).cpu()
                    neg_img_hard_ids_list_gather = neg_img_hard_ids_list_gather[:len(neg_img_loader.dataset), :]
                    train_loader.dataset.neg_img_hard_ids = neg_img_hard_ids_list_gather

            train_loader.sampler.set_epoch(epoch)
            net.train()

            if epoch in self.cfg.NET_LR_DECAY_LIST:
                net_optim.decay(self.cfg.NET_LR_DECAY_R)

            for step, train_data in enumerate(train_loader):
                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                if step % 1000 == 0 and proc_rank == 0:
                    logging.info('[Epoch Trained: {:.2f} %][Lr: {:.7f}]'.format(step / len(train_loader) * 100.,
                                                                                np.array(net_optim.get_lr()).mean()))

                if self.cfg.IMGFEAT_FORMAT == 'npz':
                    pos_text_idx, pos_img_idx, neg_text_idx, neg_img_idx = train_data
                    
                    text_input_ids = text_input_ids_all[pos_text_idx, :]
                    text_mask = text_mask_all[pos_text_idx, :]
                    imgfeat_input = imgfeat_input_all[pos_img_idx, :]
                    imgfeat_mask = imgfeat_mask_all[pos_img_idx, :]
                    imgfeat_bbox = imgfeat_bbox_all[pos_img_idx, :]
                    neg_text_input_ids = text_input_ids_all[neg_text_idx, :]
                    neg_text_mask = text_mask_all[neg_text_idx, :]
                    neg_imgfeat_input = imgfeat_input_all[neg_img_idx, :]
                    neg_imgfeat_mask = imgfeat_mask_all[neg_img_idx, :]
                    neg_imgfeat_bbox = imgfeat_bbox_all[neg_img_idx, :]
                else:
                    pos_text_idx, imgfeat_input, imgfeat_mask, imgfeat_bbox,\
                    neg_text_idx, neg_imgfeat_input, neg_imgfeat_mask, neg_imgfeat_bbox = train_data

                    text_input_ids = text_input_ids_all[pos_text_idx, :]
                    text_mask = text_mask_all[pos_text_idx, :]
                    neg_text_input_ids = text_input_ids_all[neg_text_idx, :]
                    neg_text_mask = text_mask_all[neg_text_idx, :]

                text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])
                neg_text_input_ids = neg_text_input_ids.to(self.cfg.DEVICE_IDS[0])
                neg_text_mask = neg_text_mask.to(self.cfg.DEVICE_IDS[0])
                neg_imgfeat_input = neg_imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                neg_imgfeat_mask = neg_imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                neg_imgfeat_bbox = neg_imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])

                net_input_pos = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                net_input_neg_text = (neg_text_input_ids, neg_text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                net_input_neg_img = (text_input_ids, text_mask, neg_imgfeat_input, neg_imgfeat_mask, neg_imgfeat_bbox)

                # network step
                net_optim.zero_grad()
                pred_itm_pos = net(net_input_pos)[-1]
                pred_itm_neg_text = net(net_input_neg_text)[-1]
                pred_itm_neg_img = net(net_input_neg_img)[-1]


                loss_input = pred_itm_pos, pred_itm_neg_text, pred_itm_neg_img
                total_loss, losses = net.module.loss(loss_input)
                # for avoid backward the unused params
                total_loss += 0 * sum(p.sum() for p in net.parameters())

                total_loss.backward()

                total_loss_sum += total_loss.item()
                mm_itm_tri_loss_sum += losses[0].item()

                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                if step % 100 == 0 and proc_rank == 0:
                    logging.info(
                        '[epoch: {}][step: {} | {}] - [total loss: {:.4f}][itm_tri loss: {:.4f}]'.format(
                            epoch + 1, step, len(train_loader), total_loss.item(), losses[0].item()))

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
                    '[epoch: {}][lr: {:.7f}]\n[total loss: {:.4f}][itm_tri loss: {:.4f}]\n'.format(
                        epoch_finish, np.array(net_optim.get_lr()).mean(), total_loss_sum / len(train_loader),
                        mm_itm_tri_loss_sum / len(train_loader)))
                logfile.close()

            dist.barrier()

            total_loss_sum = 0
            mm_itm_tri_loss_sum = 0
            grad_norm = np.zeros(len(named_params))

            if eval_dataset is not None:
                self.eval(eval_dataset, all_eval_data, net=net)


    def eval(self, dataset, all_data, net=None):
        init_map = {
            'vocab_size': dataset.vocab_size,
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
        with torch.no_grad():
            text_input_ids_all, text_mask_all, \
            imgfeat_input_all, imgfeat_mask_all, imgfeat_bbox_all = all_data

            bs_x = self.cfg.EVAL_BATCH_SIZE
            total_size_x = text_input_ids_all.size(0)
            col_x = math.ceil(total_size_x / bs_x)
            total_end_x = total_size_x

            if self.cfg.IMGFEAT_FORMAT == 'npz':
                total_size_y = imgfeat_input_all.size(0)
            else:
                total_size_y = len(dataset.data_aggr) // 5
            row_y = math.ceil(total_size_y / (self.cfg.WORLD_SIZE * self.cfg.NODE_SIZE))
            logging.info(f'Steps [{row_y}]')
            base_y = row_y * self.cfg.GRANK
            total_end_y = min(row_y * (self.cfg.GRANK + 1), total_size_y)

            scores_mat = torch.zeros(total_size_y, total_size_x).cuda(self.cfg.DEVICE_IDS[0])
            for step_y in range(row_y):
                proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['screen'] else self.cfg.LRANK
                if step_y % 5 == 0 and proc_rank == 0:
                    logging.info(f'Evaluated [{step_y / row_y * 100.}%]')

                start_y = base_y + step_y
                end_y = start_y + 1
                if end_y > total_end_y:
                    break
                if self.cfg.IMGFEAT_FORMAT == 'npz':
                    imgfeat_input_ = imgfeat_input_all[start_y: end_y]
                    imgfeat_mask_ = imgfeat_mask_all[start_y: end_y]
                    imgfeat_bbox_ = imgfeat_bbox_all[start_y: end_y]
                else:
                    start_idx = dataset.feat_idx_to_idx[str(start_y)]
                    formatted_data = dataset.load_formatted_data(start_idx)
                    imgfeat_input_, imgfeat_mask_, imgfeat_bbox_ = dataset.getitem__img(formatted_data)
                    imgfeat_input_ = imgfeat_input_.unsqueeze(0)
                    imgfeat_mask_ = imgfeat_mask_.unsqueeze(0)
                    imgfeat_bbox_ = imgfeat_bbox_.unsqueeze(0)

                for step_x in range(col_x):
                    start_x = step_x * bs_x
                    end_x = min((step_x + 1) * bs_x, total_end_x)
                    text_input_ids = text_input_ids_all[start_x: end_x]
                    text_mask = text_mask_all[start_x: end_x]
                    n_batches = text_input_ids.size(0)

                    imgfeat_input = imgfeat_input_.repeat(n_batches, 1, 1)
                    imgfeat_mask = imgfeat_mask_.repeat(n_batches, 1)
                    imgfeat_bbox = imgfeat_bbox_.repeat(n_batches, 1, 1)

                    text_input_ids = text_input_ids.to(self.cfg.DEVICE_IDS[0])
                    text_mask = text_mask.to(self.cfg.DEVICE_IDS[0])
                    imgfeat_input = imgfeat_input.to(self.cfg.DEVICE_IDS[0])
                    imgfeat_mask = imgfeat_mask.to(self.cfg.DEVICE_IDS[0])
                    imgfeat_bbox = imgfeat_bbox.to(self.cfg.DEVICE_IDS[0])

                    eval_input = (text_input_ids, text_mask, imgfeat_input, imgfeat_mask, imgfeat_bbox)
                    pred_itm = net(eval_input)[-1]
                    scores = torch.softmax(pred_itm, dim=-1)[:, 1]
                    scores_mat[start_y, start_x: end_x] = scores
            dist.barrier()
            dist.all_reduce(scores_mat)
            dist.barrier()

            proc_rank = self.cfg.GRANK if self.cfg.MP_STORAGE_SHR['eval'] else self.cfg.LRANK
            if proc_rank == 0:
                score_matrix = scores_mat.cpu().data.numpy()
                logging.info(f'Scores Matrix Shape is {score_matrix.shape}')

                npts = score_matrix.shape[0]
                # i2t
                stat_num = 0
                minnum_rank_image = np.array([1e7] * npts)
                for i in range(npts):
                    cur_rank = np.argsort(score_matrix[i])[::-1]
                    for index, j in enumerate(cur_rank):
                        if j in range(5 * i, 5 * i + 5):
                            stat_num += 1
                            minnum_rank_image[i] = index
                            break

                i2t_r1 = 100.0 * len(np.where(minnum_rank_image < 1)[0]) / len(minnum_rank_image)
                i2t_r5 = 100.0 * len(np.where(minnum_rank_image < 5)[0]) / len(minnum_rank_image)
                i2t_r10 = 100.0 * len(np.where(minnum_rank_image < 10)[0]) / len(minnum_rank_image)
                logging.info("i2t(TR) results: %.02f %.02f %.02f\n" % (i2t_r1, i2t_r5, i2t_r10))

                # t2i
                stat_num = 0
                score_matrix = score_matrix.transpose()
                minnum_rank_caption = np.array([1e7] * npts * 5)
                for i in range(5 * npts):
                    img_id = i // 5
                    cur_rank = np.argsort(score_matrix[i])[::-1]
                    for index, j in enumerate(cur_rank):
                        if j == img_id:
                            stat_num += 1
                            minnum_rank_caption[i] = index
                            break

                t2i_r1 = 100.0 * len(np.where(minnum_rank_caption < 1)[0]) / len(minnum_rank_caption)
                t2i_r5 = 100.0 * len(np.where(minnum_rank_caption < 5)[0]) / len(minnum_rank_caption)
                t2i_r10 = 100.0 * len(np.where(minnum_rank_caption < 10)[0]) / len(minnum_rank_caption)
                logging.info("t2i(IR) results: %.02f %.02f %.02f\n" % (t2i_r1, t2i_r5, t2i_r10))

                logfile = open(os.path.join(self.cfg.LOG_PATH, (self.cfg.VERSION + '.txt')), 'a+')
                logfile.write(
                    "i2t(TR) results: %.02f %.02f %.02f\n" % (i2t_r1, i2t_r5, i2t_r10))
                logfile.write(
                    "t2i(IR) results: %.02f %.02f %.02f\n" % (t2i_r1, t2i_r5, t2i_r10))
                logfile.write("\n")
                logfile.close()
            logging.info('Thread Done')
            dist.barrier()


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

                
            neg_text_dataset = DataSet_Neg(self.cfg, 'img', self.RUN_MODE, text_segment=train_text_segment, spacy_tool=spacy_tool)
            neg_text_sampler = SubsetDistributedSampler(neg_text_dataset, shuffle=False)
            neg_text_loader = torch.utils.data.DataLoader(
                neg_text_dataset,
                batch_size=self.cfg.NEG_BATCHSIZE,
                sampler=neg_text_sampler,
                num_workers=self.cfg.NUM_WORKERS_NEG
            )
            neg_img_dataset = DataSet_Neg(self.cfg, 'text', self.RUN_MODE, text_segment=train_text_segment, spacy_tool=spacy_tool)
            neg_img_sampler = SubsetDistributedSampler(neg_img_dataset, shuffle=False)
            neg_img_loader = torch.utils.data.DataLoader(
                neg_img_dataset,
                batch_size=self.cfg.NEG_BATCHSIZE,
                sampler=neg_img_sampler,
                num_workers=self.cfg.NUM_WORKERS_NEG
            )

            self.train(train_loader, eval_dataset, neg_text_loader, neg_img_loader)


        elif self.RUN_MODE in ['val', 'test']:
            eval_text_segment = None
            if self.cfg.SEGMENT_TEXT:
                eval_text_segment = TextSegment(self.cfg, self.RUN_MODE)
            eval_dataset = DataSet(self.cfg, self.RUN_MODE, text_segment=eval_text_segment, spacy_tool=spacy_tool)
            self.eval(eval_dataset, eval_dataset.load_all_data())


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
