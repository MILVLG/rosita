# -------------------------------------------------------- 
# ROSITA
# Licensed under The Apache License 2.0 [see LICENSE for details] 
# Written by Yuhao Cui and Tong-An Luo
# -------------------------------------------------------- 

import torch
import torch.nn as nn
from modeling.transformer import TextEmbeddings, VisualEmbeddings, Backbone, Pooler, gelu, LayerNorm
from utils.optimal_transport import optimal_transport_dist


class TextMLMHead(nn.Module):
    def __init__(self, __C, embedding_weights):
        super(TextMLMHead, self).__init__()
        self.__C = __C
        # if 'mlm' in __C.TASKS['text']:
        self.dense = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.layer_norm = LayerNorm(__C.HSIZE, eps=1e-12)
        self.decode = nn.Linear(embedding_weights.size(1), embedding_weights.size(0), bias=False)
        self.decode.weight = embedding_weights
        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))

    def forward(self, x):
        x = self.layer_norm(gelu(self.dense(x)))
        pred = self.decode(x) + self.bias
        return pred


class ImgfeatHead(nn.Module):
    def __init__(self, __C):
        super(ImgfeatHead, self).__init__()
        self.__C = __C
        self.dense = nn.Linear(__C.HSIZE, __C.HSIZE)
        self.layer_norm = LayerNorm(__C.HSIZE, eps=1e-12)
        self.dense_feat = nn.Linear(__C.HSIZE, __C.IMGFEAT_SIZE)
        self.dense_obj = nn.Linear(__C.HSIZE, __C.IMGFEAT_OBJ_CLASSNUM)
        self.dense_attr = nn.Linear(__C.HSIZE, __C.IMGFEAT_ATTR_CLASSNUM)

    def forward(self, x):
        x = self.layer_norm(gelu(self.dense(x)))
        pred_feat = self.dense_feat(x)
        pred_obj = self.dense_obj(x)
        pred_attr = self.dense_attr(x)
        return pred_feat, pred_obj, pred_attr


class MmITMHead(nn.Module):
    def __init__(self, __C):
        super(MmITMHead, self).__init__()
        self.__C = __C
        self.dense = nn.Linear(__C.HSIZE, 2)

    def forward(self, x_pooled):
        pred = self.dense(x_pooled)
        return pred


class MmQAHead(nn.Module):
    def __init__(self, __C, ans_size):
        super(MmQAHead, self).__init__()
        self.__C = __C
        self.dense0 = nn.Linear(__C.HSIZE, __C.HSIZE * 2)
        self.dense1 = nn.Linear(__C.HSIZE * 2, ans_size)
        self.layer_norm = LayerNorm(__C.HSIZE * 2, eps=1e-12)

    def forward(self, x_pooled):
        pred = self.dense1(self.layer_norm(gelu(self.dense0(x_pooled))))
        return pred


class Net(nn.Module):
    def __init__(self, __C, init_map):
        super(Net, self).__init__()
        self.__C = __C

        self.text_embeddings = TextEmbeddings(__C, init_map['vocab_size'])
        self.visual_embeddings = VisualEmbeddings(__C)

        model_dict = {
            'LAYER': __C.LAYER,
            'HSIZE': __C.HSIZE,
            'HHEAD': __C.HHEAD,
            'HBASE': __C.HBASE,
            'HFF': __C.HFF,
        }
        self.backbone = Backbone(__C, model_dict)

        self.pooler = Pooler(__C)
        self.text_mlm_head = TextMLMHead(__C, self.text_embeddings.word_embeddings.weight)
        self.imgfeat_head = ImgfeatHead(__C)
        self.mm_itm_head = MmITMHead(__C)
        self.mm_qa_head = MmQAHead(__C, init_map['ans_size'])
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

        pred_text_mlm = self.text_mlm_head(text_output)
        pred_imgfeat_feat, pred_imgfeat_obj, pred_imgfeat_attr = self.imgfeat_head(imgfeat_output)
        pred_mm_itm = self.mm_itm_head(pooled_output)
        pred_mm_qa = self.mm_qa_head(pooled_output)

        net_output = (pooled_output, text_output, imgfeat_output, pred_text_mlm,
                      pred_imgfeat_feat, pred_imgfeat_obj, pred_imgfeat_attr, pred_mm_itm, pred_mm_qa)
        return net_output

    # Loss Part
    def text_mlm_loss(self, text_outside_weight, pred_text_mlm, text_mlm_label_ids, text_mlm_weight,
                      text_mlm_loss_valid, vocab_size):
        text_outside_weight = text_outside_weight.to(self.__C.DEVICE_IDS[0])
        text_mlm_label_ids = text_mlm_label_ids.to(self.__C.DEVICE_IDS[0])
        text_mlm_weight = text_mlm_weight.to(self.__C.DEVICE_IDS[0])
        text_mlm_loss_valid = text_mlm_loss_valid.to(self.__C.DEVICE_IDS[0])

        loss_fn = self.__C.LOSSFUNC_MAPPING['text']['mlm'](reduction='none')
        loss = loss_fn(pred_text_mlm.view(-1, vocab_size), text_mlm_label_ids.view(-1))

        valid_weight = text_mlm_loss_valid.unsqueeze(1).expand_as(text_outside_weight).contiguous().view(-1)  # (160)
        if 'mlm' in self.__C.INSIDE_WEIGHTING['text']:
            valid_weight *= text_mlm_weight.view(-1)
        if 'mlm' in self.__C.OUTSIDE_WEIGHTING['text']:
            valid_weight *= text_outside_weight.view(-1)
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['text']['mlm'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['text']['mlm'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def imgfeat_feat_loss(self, imgfeat_outside_weight, pred_imgfeat_feat, imgfeat_feat_label, imgfeat_feat_weight,
                          imgfeat_feat_loss_valid):
        imgfeat_outside_weight = imgfeat_outside_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_feat_label = imgfeat_feat_label.to(self.__C.DEVICE_IDS[0])
        imgfeat_feat_weight = imgfeat_feat_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_feat_loss_valid = imgfeat_feat_loss_valid.to(self.__C.DEVICE_IDS[0])

        loss_fn = self.__C.LOSSFUNC_MAPPING['image']['feat'](reduction='none')
        loss = loss_fn(pred_imgfeat_feat.view(-1, self.__C.IMGFEAT_SIZE),
                       imgfeat_feat_label.view(-1, self.__C.IMGFEAT_SIZE))
        if self.__C.LOSS_REDUCTION['image']['feat'] == 'mean':
            loss = loss.mean(-1)
        elif self.__C.LOSS_REDUCTION['image']['feat'] == 'sum':
            loss = loss.sum(-1)

        valid_weight = imgfeat_feat_loss_valid.unsqueeze(1).expand_as(imgfeat_outside_weight).contiguous().view(-1)  # (288)
        if 'feat' in self.__C.INSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_feat_weight.view(-1)
        if 'feat' in self.__C.OUTSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_outside_weight.view(-1)
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['image']['feat'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['image']['feat'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def imgfeat_obj_loss(self, imgfeat_outside_weight, pred_imgfeat_obj, imgfeat_obj_label, imgfeat_obj_weight,
                         imgfeat_obj_loss_valid):
        imgfeat_outside_weight = imgfeat_outside_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_obj_label = imgfeat_obj_label.to(self.__C.DEVICE_IDS[0])
        imgfeat_obj_weight = imgfeat_obj_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_obj_loss_valid = imgfeat_obj_loss_valid.to(self.__C.DEVICE_IDS[0])

        loss_fn = self.__C.LOSSFUNC_MAPPING['image']['obj'](reduction='none')
        loss = loss_fn(pred_imgfeat_obj.view(-1, self.__C.IMGFEAT_OBJ_CLASSNUM), imgfeat_obj_label.view(-1))

        valid_weight = imgfeat_obj_loss_valid.unsqueeze(1).expand_as(imgfeat_outside_weight).contiguous().view(-1)  # (288)
        if 'obj' in self.__C.INSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_obj_weight.view(-1)
        if 'obj' in self.__C.OUTSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_outside_weight.view(-1)
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['image']['obj'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['image']['obj'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def imgfeat_attr_loss(self, imgfeat_outside_weight, pred_imgfeat_attr, imgfeat_attr_label, imgfeat_attr_weight,
                          imgfeat_attr_loss_valid):
        imgfeat_outside_weight = imgfeat_outside_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_attr_label = imgfeat_attr_label.to(self.__C.DEVICE_IDS[0])
        imgfeat_attr_weight = imgfeat_attr_weight.to(self.__C.DEVICE_IDS[0])
        imgfeat_attr_loss_valid = imgfeat_attr_loss_valid.to(self.__C.DEVICE_IDS[0])

        loss_fn = self.__C.LOSSFUNC_MAPPING['image']['attr'](reduction='none')
        loss = loss_fn(pred_imgfeat_attr.view(-1, self.__C.IMGFEAT_ATTR_CLASSNUM), imgfeat_attr_label.view(-1))

        valid_weight = imgfeat_attr_loss_valid.unsqueeze(1).expand_as(imgfeat_outside_weight).contiguous().view(-1)  # (288)
        if 'attr' in self.__C.INSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_attr_weight.view(-1)
        if 'attr' in self.__C.OUTSIDE_WEIGHTING['image']:
            valid_weight *= imgfeat_outside_weight.view(-1)
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['image']['attr'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['image']['attr'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def mm_itm_loss(self, pred_mm_itm, itm_label, itm_loss_valid, loss_ot_input):
        itm_label = itm_label.to(self.__C.DEVICE_IDS[0])
        itm_loss_valid = itm_loss_valid.to(self.__C.DEVICE_IDS[0])

        loss_fn = self.__C.LOSSFUNC_MAPPING['mm']['itm'](reduction='none', ignore_index=-1)
        loss = loss_fn(pred_mm_itm, itm_label)

        valid_weight = itm_loss_valid
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['mm']['itm'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['mm']['itm'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        if self.__C.OT_LAMBDA > 0:
            text_output, imgfeat_output, text_mask, imgfeat_mask = loss_ot_input
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(text_output.float(), imgfeat_output.float(),
                (1.0 - text_mask).type(torch.uint8), (1.0 - imgfeat_mask).type(torch.uint8)).to(text_output)
            ot_pos_dist = ot_dist.masked_select((itm_label == 1) * (itm_loss_valid == 1))
            ot_neg_dist = ot_dist.masked_select((itm_label == 0) * (itm_loss_valid == 1))
            if ot_pos_dist.size(0) + ot_neg_dist.size(0) > 0:
                ot_loss = (ot_pos_dist.sum() - ot_neg_dist.sum()) / (ot_pos_dist.size(0) + ot_neg_dist.size(0))
            else:
                ot_loss = 0

            loss += self.__C.OT_LAMBDA * ot_loss

        return loss

    def mm_qa_loss(self, pred_mm_qa, qa_label, qa_loss_valid, ans_size):
        qa_label = qa_label.to(self.__C.DEVICE_IDS[0])
        qa_loss_valid = qa_loss_valid.to(self.__C.DEVICE_IDS[0])

        if self.__C.LOSSFUNC_MAPPING['mm']['qa'] == nn.CrossEntropyLoss:
            loss_fn = self.__C.LOSSFUNC_MAPPING['mm']['qa'](reduction='none', ignore_index=-1)
        elif self.__C.LOSSFUNC_MAPPING['mm']['qa'] == nn.BCEWithLogitsLoss:
            loss_fn = self.__C.LOSSFUNC_MAPPING['mm']['qa'](reduction='none')
        elif self.__C.LOSSFUNC_MAPPING['mm']['qa'] == nn.KLDivLoss:
            loss_fn = self.__C.LOSSFUNC_MAPPING['mm']['qa'](reduction='none')
            pred_mm_qa = torch.log_softmax(pred_mm_qa, dim=-1)

        loss = loss_fn(pred_mm_qa, qa_label)
        if len(qa_label.size()) > 1:
            if self.__C.LOSS_REDUCTION['mm']['qa'] == 'mean':
                loss = loss.mean(-1)
            elif self.__C.LOSS_REDUCTION['mm']['qa'] == 'sum':
                loss = loss.sum(-1)

        valid_weight = qa_loss_valid
        loss *= valid_weight
        if (valid_weight > 0).float().sum() > 0:
            if self.__C.LOSS_REDUCTION['mm']['qa'] == 'mean':
                loss = loss.sum() / (valid_weight > 0).float().sum()
            elif self.__C.LOSS_REDUCTION['mm']['qa'] == 'sum':
                loss = loss.sum()
        else:
            loss = 0. * loss.sum()

        return loss

    def loss(self, loss_input, loss_ot_input=None):
        init_map, text_outside_weight, pred_text_mlm, \
        text_mlm_label_ids, text_mlm_weight, text_mlm_loss_valid, \
        imgfeat_outside_weight, pred_imgfeat_feat, pred_imgfeat_obj, pred_imgfeat_attr, \
        imgfeat_feat_label, imgfeat_feat_weight, imgfeat_feat_loss_valid, \
        imgfeat_obj_label, imgfeat_obj_weight, imgfeat_obj_loss_valid, \
        imgfeat_attr_label, imgfeat_attr_weight, imgfeat_attr_loss_valid, \
        pred_mm_itm, pred_mm_qa, itm_label, itm_loss_valid, qa_label, qa_loss_valid, = loss_input

        total_loss = 0
        losses = 6 * [torch.tensor(0)]

        if 'mlm' in self.__C.TASKS['text']:
            text_mlm_loss_output = self.__C.LOSSFUNC_WEIGHT['text']['mlm'] * self.text_mlm_loss(
                text_outside_weight, pred_text_mlm, text_mlm_label_ids, text_mlm_weight, text_mlm_loss_valid, init_map['vocab_size'])
            total_loss += text_mlm_loss_output
            losses[0] = text_mlm_loss_output

        if 'feat' in self.__C.TASKS['image']:
            imgfeat_feat_loss_output = self.__C.LOSSFUNC_WEIGHT['image']['feat'] * self.imgfeat_feat_loss(
                imgfeat_outside_weight, pred_imgfeat_feat, imgfeat_feat_label, imgfeat_feat_weight, imgfeat_feat_loss_valid)
            total_loss += imgfeat_feat_loss_output
            losses[1] = imgfeat_feat_loss_output

        if 'obj' in self.__C.TASKS['image']:
            imgfeat_obj_loss_output = self.__C.LOSSFUNC_WEIGHT['image']['obj'] * self.imgfeat_obj_loss(
                imgfeat_outside_weight, pred_imgfeat_obj, imgfeat_obj_label, imgfeat_obj_weight, imgfeat_obj_loss_valid)
            total_loss += imgfeat_obj_loss_output
            losses[2] = imgfeat_obj_loss_output

        if 'attr' in self.__C.TASKS['image']:
            imgfeat_attr_loss_output = self.__C.LOSSFUNC_WEIGHT['image']['attr'] * self.imgfeat_attr_loss(
                imgfeat_outside_weight, pred_imgfeat_attr, imgfeat_attr_label, imgfeat_attr_weight,
                imgfeat_attr_loss_valid)
            total_loss += imgfeat_attr_loss_output
            losses[3] = imgfeat_attr_loss_output

        if 'itm' in self.__C.TASKS['mm']:
            mm_itm_loss_output = self.__C.LOSSFUNC_WEIGHT['mm']['itm'] * self.mm_itm_loss(
                pred_mm_itm, itm_label, itm_loss_valid, loss_ot_input)
            total_loss += mm_itm_loss_output
            losses[4] = mm_itm_loss_output

        if 'qa' in self.__C.TASKS['mm']:
            mm_qa_loss_output = self.__C.LOSSFUNC_WEIGHT['mm']['qa'] * self.mm_qa_loss(
                pred_mm_qa, qa_label, qa_loss_valid, init_map['ans_size'])
            total_loss += mm_qa_loss_output
            losses[5] = mm_qa_loss_output

        return total_loss, losses

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
