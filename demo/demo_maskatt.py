# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import torch, math
import torch.nn as nn
from rosita.modeling.transformer import LayerNorm, FeedForward, TextEmbeddings, VisualEmbeddings, Pooler
import numpy as np
import re, torch, collections, copy, os
from utils.tokenizer import BertTokenizer


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


class DataSet():
    def __init__(self, __C):
        self.__C = __C

        self.tokenizer = BertTokenizer(self.load_vocab(__C.BERT_VOCAB_PATH))
        self.vocab_size = len(self.tokenizer.vocab)


    def get_item(self, img_filename, mask_side, mask_id, text_id=None, text=None):
        assert not (text_id is None and text is None)

        # Load text and image features
        np_file = os.path.join('demo', 'features', (img_filename + '.npz'))
        npz_loaded = np.load(np_file)
        if text is None:
            text = list(npz_loaded['text'])[text_id]
        imgfeat_x = npz_loaded['x']
        image_h = npz_loaded['image_h']
        image_w = npz_loaded['image_w']
        boxes = npz_loaded['boxes']
        text = self.clean_text(text)

        # Proc masking tasks
        is_text_masking = mask_side == 'text'
        is_imgfeat_masking = mask_side == 'img'

        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        self.check_tsg(text, tokenized_text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        # Masking text
        text_input = copy.deepcopy(tokenized_text)
        text_mlm_label = copy.deepcopy(tokenized_text)
        if is_text_masking:
            text_input = self.masking_text(text_input, mask_id)

        # Padding and convert text ids
        text_input_ids, text_mask, text_mlm_label_ids= self.proc_text(
            text_input, text_mlm_label)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)
        text_mlm_label_ids = torch.tensor(text_mlm_label_ids, dtype=torch.int64)

        # Masking image features
        imgfeat_input = imgfeat_x

        if is_imgfeat_masking:
            imgfeat_input = self.masking_imgfeat(imgfeat_input, mask_id)

        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))

        imgfeat_input, imgfeat_mask, \
        imgfeat_bbox = self.proc_imgfeat(
            imgfeat_input,
            imgfeat_bbox, text_input_ids.size(0))

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)

        return  text_input_ids, text_mask, text_mlm_label_ids, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, \
                len(tokenized_text), boxes


    def check_tsg(self, text_input, tokenized_text):
        bpe_ids = []
        for step, text in enumerate(tokenized_text):      
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)

        assert len(bpe_ids) == len(self.clean_text(text_input).split())
        

    def clean_text(self, text):
        text = re.sub(r'([^\s\w]|_)+', '', text)
        return text


    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


    def masking_text(self, tokenized_text, mask_id):
        masked_tokenized_text = copy.deepcopy(tokenized_text)
        mlm_pos = [0.] * len(masked_tokenized_text)
        
        # MLM
        assert mask_id is not None
        for i in range(len(masked_tokenized_text)):
            if i in mask_id:
                mlm_pos[i] = 1.

        # Do Mask
        for i in range(len(masked_tokenized_text)):
            if mlm_pos[i] == 1.:
                masked_tokenized_text[i] = '[MASK]'

        return masked_tokenized_text


    def proc_text(self, text_input, text_mlm_label):
        # concatenate lm labels and account for CLS, SEP
        text_input = ['[CLS]'] + text_input + ['[SEP]']
        text_mlm_label = ['[CLS]'] + text_mlm_label + ['[SEP]']

        text_input_ids = self.tokenizer.convert_tokens_to_ids(text_input)
        text_mlm_label_ids = self.tokenizer.convert_tokens_to_ids(text_mlm_label)

        # Mask & Segment Word
        text_mask = [1] * len(text_input_ids)

        pad_length = self.__C.PAD_MAX['text'] - len(text_input_ids)
        if self.__C.PAD_INSIDE and len(text_input_ids) < self.__C.PAD_MAX['text']:
            text_input_ids += [0] * pad_length
            text_mlm_label_ids += [0] * pad_length
            text_mask += [0] * pad_length

        return text_input_ids, text_mask, text_mlm_label_ids


    def masking_imgfeat(self, imgfeat_input, mask_id):
        masked_imgfeat_input = imgfeat_input.copy()
        mlm_pos = np.zeros(imgfeat_input.shape[0], dtype=np.float32)

        # MRM
        assert mask_id is not None
        for i in range(imgfeat_input.shape[0]):
            if i in mask_id:
                mlm_pos[i] = 1.

        # Do Mask
        for i in range(imgfeat_input.shape[0]):
            if mlm_pos[i] == 1.:
                masked_imgfeat_input[i, :] = 0.

        return masked_imgfeat_input


    def np_pad_1d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, (0, length - tensor.shape[0]), mode='constant', constant_values=value)


    def np_pad_2d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, ((0, length - tensor.shape[0]), (0, 0)), mode='constant', constant_values=value)
        

    def proc_imgfeat(self, imgfeat_input, imgfeat_bbox, length_pre):
        length_pad = self.__C.PAD_MAX['image'] + self.__C.PAD_MAX['text'] - length_pre

        imgfeat_mask = torch.ones(imgfeat_input.shape[0], dtype=torch.float32)
        imgfeat_mask = self.np_pad_1d(imgfeat_mask, length_pad)

        imgfeat_input = self.np_pad_2d(imgfeat_input, length_pad)
        imgfeat_bbox = self.np_pad_2d(imgfeat_bbox, length_pad)

        return imgfeat_input, imgfeat_mask, \
               imgfeat_bbox


    def proc_bbox(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat

