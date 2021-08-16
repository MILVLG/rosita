# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import torch
import torch.nn as nn
from modeling.transformer import LayerNorm, TextEmbeddings, VisualEmbeddings, Backbone, Pooler, gelu


class MmRefsHead(nn.Module):
    def __init__(self, cfg):
        super(MmRefsHead, self).__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.HSIZE, cfg.HSIZE)
        self.layer_norm = LayerNorm(cfg.HSIZE, eps=1e-12)
        self.dense_rank = nn.Linear(cfg.HSIZE, 1)
        self.dense_reg = nn.Linear(cfg.HSIZE, 4)

    def forward(self, x):
        x = self.layer_norm(gelu(self.dense(x)))
        pred_rank = self.dense_rank(x).squeeze(-1)
        pred_reg = self.dense_reg(x)
        return pred_rank, pred_reg


class Net(nn.Module):
    def __init__(self, cfg, init_map):
        super(Net, self).__init__()
        self.cfg = cfg

        self.text_embeddings = TextEmbeddings(cfg, init_map['vocab_size'])
        self.visual_embeddings = VisualEmbeddings(cfg)

        model_dict = {
            'LAYER': cfg.LAYER,
            'HSIZE': cfg.HSIZE,
            'HHEAD': cfg.HHEAD,
            'HBASE': cfg.HBASE,
            'HFF': cfg.HFF,
        }
        self.backbone = Backbone(cfg, model_dict)
        self.pooler = Pooler(cfg)
        self.mm_refs_head = MmRefsHead(cfg)
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
        x = self.backbone(x, x_mask)

        pooled_output = self.pooler(x)
        text_output, imgfeat_output = torch.split(x, (text_len, imgfeat_len), dim=1)
        pred_refs_rank, pred_refs_reg = self.mm_refs_head(imgfeat_output)

        net_output = (pooled_output, text_output, imgfeat_output, pred_refs_rank, pred_refs_reg)
        return net_output

    # Loss Part
    def mm_refs_rank_loss(self, pred_refs_rank, refs_rank_label, refs_rank_weight, refs_rank_loss_valid):
        refs_rank_label = refs_rank_label.to(self.cfg.DEVICE_IDS[0])
        refs_rank_weight = refs_rank_weight.to(self.cfg.DEVICE_IDS[0])
        refs_rank_loss_valid = refs_rank_loss_valid.to(self.cfg.DEVICE_IDS[0])

        loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['refs-rank'](reduction='none')
        pred_refs_rank = torch.log_softmax(pred_refs_rank, dim=-1)
        loss = loss_fn(pred_refs_rank, refs_rank_label)
        assert len(loss.size()) == 2

        valid_weight = refs_rank_loss_valid.unsqueeze(1).expand_as(refs_rank_weight).contiguous()
        if 'refs-rank' in self.cfg.INSIDE_WEIGHTING['mm']:
            valid_weight *= refs_rank_weight
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.cfg.LOSS_REDUCTION['mm']['refs-rank'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.cfg.LOSS_REDUCTION['mm']['refs-rank'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss
    
    def mm_refs_reg_loss(self, pred_refs_reg, refs_reg_label, refs_reg_weight, refs_reg_loss_valid):
        refs_reg_label = refs_reg_label.to(self.cfg.DEVICE_IDS[0])
        refs_reg_weight = refs_reg_weight.to(self.cfg.DEVICE_IDS[0])
        refs_reg_loss_valid = refs_reg_loss_valid.to(self.cfg.DEVICE_IDS[0])

        loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['refs-reg'](reduction='none')
        loss = loss_fn(pred_refs_reg, refs_reg_label)
        if self.cfg.LOSS_REDUCTION['mm']['refs-reg'] == 'mean':
            loss = loss.mean(-1)
        elif self.cfg.LOSS_REDUCTION['mm']['refs-reg'] == 'sum':
            loss = loss.sum(-1)
        assert len(loss.size()) == 2

        valid_weight = refs_reg_loss_valid.unsqueeze(1).expand_as(refs_reg_weight).contiguous()
        if 'refs-reg' in self.cfg.INSIDE_WEIGHTING['mm']:
            valid_weight *= refs_reg_weight
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.cfg.LOSS_REDUCTION['mm']['refs-reg'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.cfg.LOSS_REDUCTION['mm']['refs-reg'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss


    def loss(self, loss_input):
        pred_refs_rank, pred_refs_reg, \
        refs_rank_label, refs_rank_weight, refs_rank_loss_valid, \
        refs_reg_label, refs_reg_weight, refs_reg_loss_valid = loss_input

        total_loss = 0
        losses = 2 * [torch.tensor(0)]
        if 'refs-rank' in self.cfg.TASKS['mm']:
            mm_refs_rank_loss_output = self.cfg.LOSSFUNC_WEIGHT['mm']['refs-rank'] * self.mm_refs_rank_loss(
                pred_refs_rank, refs_rank_label, refs_rank_weight, refs_rank_loss_valid)
            total_loss += mm_refs_rank_loss_output
            losses[0] = mm_refs_rank_loss_output

        if 'refs-reg' in self.cfg.TASKS['mm']:
            mm_refs_reg_loss_output = self.cfg.LOSSFUNC_WEIGHT['mm']['refs-reg'] * self.mm_refs_reg_loss(
                pred_refs_reg, refs_reg_label, refs_reg_weight, refs_reg_loss_valid)
            total_loss += mm_refs_reg_loss_output
            losses[1] = mm_refs_reg_loss_output

        return total_loss, losses

    def build_mask(self, mask):
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        mask = (1.0 - mask) * -10000.0
        return mask

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.WEIGHT_INIT_FACTOR)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()