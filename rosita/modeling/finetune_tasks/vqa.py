# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import torch
import torch.nn as nn
from modeling.transformer import TextEmbeddings, VisualEmbeddings, Backbone, Pooler, gelu, LayerNorm


class MmQAHead(nn.Module):
    def __init__(self, cfg, ans_size):
        super(MmQAHead, self).__init__()
        self.cfg = cfg
        self.dense0 = nn.Linear(cfg.HSIZE, cfg.HSIZE * 2)
        self.dense1 = nn.Linear(cfg.HSIZE * 2, ans_size)
        self.layer_norm = LayerNorm(cfg.HSIZE * 2, eps=1e-12)

    def forward(self, x_pooled):
        pred = self.dense1(self.layer_norm(gelu(self.dense0(x_pooled))))
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
        self.mm_qa_head = MmQAHead(cfg, init_map['ans_size'])
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
        pred_mm_qa = self.mm_qa_head(pooled_output)

        net_output = (pooled_output, text_output, imgfeat_output, pred_mm_qa)
        return net_output

    # Loss Part
    def mm_qa_loss(self, pred_mm_qa, qa_label, qa_loss_valid, ans_size):
        qa_label = qa_label.to(self.cfg.DEVICE_IDS[0])
        qa_loss_valid = qa_loss_valid.to(self.cfg.DEVICE_IDS[0])

        if self.cfg.LOSSFUNC_MAPPING['mm']['qa'] == nn.CrossEntropyLoss:
            loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['qa'](reduction='none', ignore_index=-1)
        elif self.cfg.LOSSFUNC_MAPPING['mm']['qa'] == nn.BCEWithLogitsLoss:
            loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['qa'](reduction='none')
        elif self.cfg.LOSSFUNC_MAPPING['mm']['qa'] == nn.KLDivLoss:
            loss_fn = self.cfg.LOSSFUNC_MAPPING['mm']['qa'](reduction='none')
            pred_mm_qa = torch.log_softmax(pred_mm_qa, dim=-1)

        loss = loss_fn(pred_mm_qa, qa_label)
        if len(qa_label.size()) > 1:
            if self.cfg.LOSS_REDUCTION['mm']['qa'] == 'mean':
                loss = loss.mean(-1)
            elif self.cfg.LOSS_REDUCTION['mm']['qa'] == 'sum':
                loss = loss.sum(-1)

        valid_weight = qa_loss_valid
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.cfg.LOSS_REDUCTION['mm']['qa'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.cfg.LOSS_REDUCTION['mm']['qa'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def loss(self, loss_input):
        init_map, pred_mm_qa, qa_label, qa_loss_valid, = loss_input

        total_loss = 0
        loss = torch.tensor(0)

        assert 'qa' in self.cfg.TASKS['mm']
        mm_qa_loss_output = self.cfg.LOSSFUNC_WEIGHT['mm']['qa'] * self.mm_qa_loss(
            pred_mm_qa, qa_label, qa_loss_valid, init_map['ans_size'])
        total_loss += mm_qa_loss_output
        loss = mm_qa_loss_output

        return total_loss, loss

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