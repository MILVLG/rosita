# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import torch
import torch.nn as nn
from modeling.transformer import TextEmbeddings, VisualEmbeddings, Backbone, Pooler, LayerNorm


class MmITMHead(nn.Module):
    def __init__(self, cfg):
        super(MmITMHead, self).__init__()
        self.cfg = cfg
        self.dense = nn.Linear(cfg.HSIZE, 2)
    
    def forward(self, x_pooled):
        pred = self.dense(x_pooled)
        return pred

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
        self.mm_itm_head = MmITMHead(cfg)
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
        pred_mm_itm = self.mm_itm_head(pooled_output)

        net_output = (pooled_output, text_output, imgfeat_output, pred_mm_itm)
        return net_output

    # Loss Part
    def mm_itm_triplet_loss(self, pred_pos, pred_neg_text, pred_neg_img):
        loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['itm-tri'](reduction=self.cfg.LOSS_REDUCTION['mm']['itm-tri'])

        label_pos = torch.ones(pred_pos.size()[:-1], dtype=torch.long).to(pred_pos.device)
        loss_pos = loss_fn(pred_pos, label_pos)
        label_neg_text = torch.zeros(pred_neg_text.size()[:-1], dtype=torch.long).to(pred_neg_text.device)
        loss_neg_text = loss_fn(pred_neg_text, label_neg_text)
        label_neg_img = torch.zeros(pred_neg_img.size()[:-1], dtype=torch.long).to(pred_neg_img.device)
        loss_neg_img = loss_fn(pred_neg_img, label_neg_img)

        loss = loss_pos + loss_neg_text + loss_pos + loss_neg_img
        return loss
    
    def mm_itm_margin_loss(self, pred_pos, pred_neg_text, pred_neg_img):
        margin = 0.2
        cost_text = (margin + pred_neg_text - pred_pos).clamp(min=0)
        cost_img = (margin + pred_neg_img - pred_pos).clamp(min=0)
        return (cost_text + cost_img).mean()

    def loss(self, loss_input):
        total_loss = 0
        losses = [torch.tensor(0)]
        if 'itm-tri' in self.cfg.TASKS['mm']:
            mm_itm_tri_loss_output = self.cfg.LOSSFUNC_WEIGHT['mm']['itm-tri'] * self.mm_itm_triplet_loss(*loss_input)
            total_loss += mm_itm_tri_loss_output
            losses[0] = mm_itm_tri_loss_output

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