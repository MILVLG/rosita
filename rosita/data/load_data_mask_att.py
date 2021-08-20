# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import numpy as np
import json, re, torch, logging, collections, random, copy, math, base64, os
import torch.utils.data as Data
import torch.nn as nn
from utils.answer_punct import preprocess_answer
from utils.tokenizer import BertTokenizer
from utils.tsv_file import TSVFile


class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE, text_segment=None, spacy_tool=None):
        self.__C = __C
        self.RUN_MODE = RUN_MODE
        self.text_segment = text_segment
        assert self.__C.IMGFEAT_FORMAT in ['npz', 'tsv']
        self.use_tsv = False
        if self.__C.IMGFEAT_FORMAT == 'tsv':
            self.use_tsv = True
        logging.info('[dataset: {}] Loader Initializing'.format(RUN_MODE))
        
        if self.use_tsv:
            self.tsv_files = {}
            self.img_feat_offset_maps = {}
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                fset = self.__C.DATASET_FEATMAP[tset]
                tsv_file, img_feat_offset_map = self.load_tsv(fset)
                self.tsv_files[fset] = tsv_file
                self.img_feat_offset_maps[fset] = img_feat_offset_map
        if text_segment is not None:
            logging.info('Use Text Segment for Memory Efficiency')
        else:
            self.data_aggr = []
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                ttype = dataset_name.split(':')[1]
                formatted_data = json.load(open(__C.DATASET_ANNO_MAP[tset], 'r'))[ttype]
                self.data_aggr += formatted_data
                logging.info('[dataset: {}] Loading [{}] data: {}'.format(RUN_MODE, dataset_name, len(formatted_data)))
            logging.info('[dataset: {}] Total Data: {}'.format(RUN_MODE, len(self.data_aggr)))

        self.tokenizer = BertTokenizer(self.load_vocab(__C.BERT_VOCAB_PATH))
        self.vocab_size = len(self.tokenizer.vocab)
        logging.info('[dataset: {}] Total Vocab: {}'.format(RUN_MODE, self.vocab_size))
        logging.info('[dataset: {}] Loader Initialized'.format(RUN_MODE))
        self.spacy_tool = spacy_tool

        if self.text_segment is not None:
            self.data_size = self.text_segment.total_len
        else:
            self.data_size = len(self.data_aggr)


    def get_item(self, idx, mask_side, mask_id=None):
        if self.text_segment is not None:
            formatted_data = self.text_segment.load(idx)
        else:
            formatted_data = self.data_aggr[idx]

        formatted_data_text = formatted_data
        formatted_data_img = formatted_data

        # Load text
        text = self.clean_text(formatted_data_text['text'])
        lemmas = self.clean_text(formatted_data_text['lemmas'])

        # Load negative image features
        img_src = formatted_data_img['img_src']
        img_filename = formatted_data_img['img_file']
        if self.use_tsv:
            tsv_file = self.tsv_files[img_src]
            img_offset_map = self.img_feat_offset_maps[img_src]
            img_idx = img_offset_map[img_filename]
            row = tsv_file.seek(img_idx)
            imgfeat = {}
            imgfeat['filename'] = row[0]
            num_bboxes = int(row[4])
            imgfeat['x'] = np.frombuffer(base64.b64decode(row[1]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['image_h'], imgfeat['image_w'] = int(row[2]), int(row[3])
            imgfeat['num_boxes'] = num_bboxes
            imgfeat['boxes'] = np.frombuffer(base64.b64decode(row[5]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['objects_id'] = np.frombuffer(base64.b64decode(row[6]), dtype=np.float32)
            imgfeat['objects_conf'] = np.frombuffer(base64.b64decode(row[7]), dtype=np.float32)
            imgfeat['attrs_id'] = np.frombuffer(base64.b64decode(row[8]), dtype=np.float32)
            imgfeat['attrs_conf'] = np.frombuffer(base64.b64decode(row[9]), dtype=np.float32)
        else:
            imgfeat = self.load_npz(img_src, img_filename)

        # Proc masking tasks
        is_text_masking = mask_side == 'text'
        is_imgfeat_masking = mask_side == 'img'

        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        self.check_tsg(text, tokenized_text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        mask_id_list = None

        # Masking text
        text_input = copy.deepcopy(tokenized_text)
        text_mlm_label = copy.deepcopy(tokenized_text)
        if is_text_masking:
            text_input, mask_id_list = self.masking_text(text_input, formatted_data_text, lemmas, mask_id)

        # Padding and convert text ids
        text_input_ids, text_mask, text_mlm_label_ids= self.proc_text(
            text_input, text_mlm_label)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)
        text_mlm_label_ids = torch.tensor(text_mlm_label_ids, dtype=torch.int64)

        # Masking image features
        imgfeat_x = imgfeat['x']
        image_h = imgfeat['image_h']
        image_w = imgfeat['image_w']
        boxes = imgfeat['boxes']

        imgfeat_input = imgfeat_x

        if is_imgfeat_masking:
            imgfeat_input, mask_id_list = self.masking_imgfeat(imgfeat_input, boxes, mask_id)

        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))

        imgfeat_input, imgfeat_mask, \
        imgfeat_bbox = self.proc_imgfeat(
            imgfeat_input,
            imgfeat_bbox, text_input_ids.size(0))

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)

        # Get text and image id
        # text_id = formatted_data_text['text_id']
        # img_id = formatted_data_img['img_id']

        return  text_input_ids, text_mask, text_mlm_label_ids, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, \
                mask_id_list, len(tokenized_text), img_filename, boxes


    def __len__(self):
        if self.text_segment is not None:
            return self.text_segment.total_len
        else:
            return len(self.data_aggr)


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

    
    @staticmethod
    def filter_ans(__C, stat_ans_list, is_punct, ans_freq_thresh):
        ans_set = set()
        ans_freq_map = {}

        for stat_ans_name in stat_ans_list:
            tset = stat_ans_name.split(':')[0]
            ttype = stat_ans_name.split(':')[1]
            t_formatted_data = json.load(open(__C.DATASET_PATHMAP[tset], 'r'))[ttype]

            for part_data in t_formatted_data:
                ans = part_data['label']
                if ans == '':
                    continue
                if is_punct:
                    ans = preprocess_answer(ans)

                if ans not in ans_freq_map:
                    ans_freq_map[ans] = 1
                else:
                    ans_freq_map[ans] += 1

        ans_freq_filter = ans_freq_map.copy()
        for ans in ans_freq_map:
            if ans_freq_map[ans] <= ans_freq_thresh:
                ans_freq_filter.pop(ans)

        for ans in ans_freq_filter:
            ans_set.add(ans)

        return ans_set


    @staticmethod
    def sets_to_idmap(ans_sets):
        ans_to_ix = {}
        ix_to_ans = {}
        for ans in ans_sets:
            ix_to_ans[len(ans_to_ix)] = ans
            ans_to_ix[ans] = len(ans_to_ix)

        return ans_to_ix, ix_to_ans

    
    def load_npz(self, img_src, img_filename):
        np_file = os.path.join(self.__C.IMGFEAT_PATHMAP[img_src], 'npz_files', (img_filename + '.npz'))
        npz_loaded = np.load(np_file)

        return npz_loaded
    
    def load_tsv(self, img_src):
        tsv_file = self.__C.IMGFEAT_PATHMAP[img_src] + '/imgfeat.tsv'
        img_feat_offset_map_file = self.__C.IMGFEAT_PATHMAP[img_src] + '/img_feat_offset_map.json'
        with open(img_feat_offset_map_file) as f:
            img_feat_offset_map = json.load(f)
        return TSVFile(tsv_file), img_feat_offset_map



    def filter_tsg_obj(self, tokenized_text, formatted_data, weight_filter):
        bpe_ids = []
        for step, text in enumerate(tokenized_text):      
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)

        text_sg = formatted_data['tsg']
        objs_ids = text_sg['objs']
        attrs_ids = text_sg['attrs']
        rels_ids = text_sg['rels']
        # mixed_ids = text_sg['mixed']

        all_sg_ids = objs_ids + attrs_ids + rels_ids

        grain_ids_obj = []
        for sg_ids in all_sg_ids:
            if len(sg_ids) == 1:
                grain_ids_obj.append(sg_ids[0])

        grain_ids_obj = sum(grain_ids_obj, [])
        for mask_id in grain_ids_obj:
            if mask_id < len(bpe_ids):
                for sent_id in bpe_ids[mask_id]:
                    weight_filter[sent_id] = 1.

        return weight_filter


    def masking_text_repalce(self, ids, masked_tokenized_text):
        prob = random.random()
        if prob < self.__C.MASK_PROB_POST['mask']:
            masked_tokenized_text[ids] = '[MASK]'
        elif prob < self.__C.MASK_PROB_POST['mask'] + self.__C.MASK_PROB_POST['replace']:
            masked_tokenized_text[ids] = random.choice(list(self.tokenizer.vocab.items()))[0]
        
        return masked_tokenized_text


    def tsg_map_to_obj(self, formatted_data, tokens):
        text_sg = formatted_data['tsg']
        # objs_ids = text_sg['objs']
        attrs_ids = text_sg['attrs']
        rels_ids = text_sg['rels']
        # mixed_ids = text_sg['mixed']
        # all_sg_ids = objs_ids + attrs_ids + rels_ids

        attr_map = {}
        attr_map_ids = {}
        for ids in attrs_ids:
            cent = tokens[ids[0][0]]
            sur = [tokens[t] for t in ids[1]]
            if cent not in attr_map:
                attr_map[cent] = [sur]
                attr_map_ids[cent] = [ids[1]]
            else:
                attr_map[cent].append(sur)
                attr_map_ids[cent].append(ids[1])
        for cent in attr_map:
            for cent_i, cent_v in enumerate(attr_map[cent]):
                attr_map[cent][cent_i] = ' '.join(cent_v)

        rel_map = {}
        rel_map_ids = {}
        for ids in rels_ids:
            cent = tokens[ids[0][0]]
            sur = [tokens[t] for t in ids[1]] + [tokens[t] for t in ids[2]]
            if cent not in rel_map:
                rel_map[cent] = [sur]
                rel_map_ids[cent] = [ids[1]]
            else:
                rel_map[cent].append(sur)
                rel_map_ids[cent].append(ids[1])
        for ids in rels_ids:
            cent = tokens[ids[2][0]]
            sur = [tokens[t] for t in ids[0]] + [tokens[t] for t in ids[1]]
            if cent not in rel_map:
                rel_map[cent] = [sur]
                rel_map_ids[cent] = [ids[1]]
            else:
                rel_map[cent].append(sur)
                rel_map_ids[cent].append(ids[1])
        for cent in rel_map:
            for cent_i, cent_v in enumerate(rel_map[cent]):
                rel_map[cent][cent_i] = ' '.join(cent_v)

        return attr_map, attr_map_ids, rel_map, rel_map_ids


    def masking_struct_tsg_1d(self, mlm_pos, tokenized_text, formatted_data, lemmas):
        bpe_ids = []
        for step, text in enumerate(tokenized_text):
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)
        tokens = lemmas.strip().split()
        text_sg = formatted_data['tsg']
        objs_ids = text_sg['objs']
        # attr_map, attr_map_ids, rel_map, rel_map_ids = self.tsg_map_to_obj(formatted_data, tokens)

        mask_ids = []
        mask_prob = torch.zeros(len(tokens), dtype=torch.float32)
        for objs_id in objs_ids:
            if random.random() < (self.__C.MASK_STRUCT_PROB['tsg']):
                mask_ids.append(objs_id[0][0])
        for i in range(mask_prob.size(0)):
            if random.random() < mask_prob[i]:
                mask_ids.append(i)

        mask_id_list = []
        # mask
        for mask_id in mask_ids:
            if mask_id < len(bpe_ids):
                for sent_id in bpe_ids[mask_id]:
                    mlm_pos[sent_id] = 1.
                    mask_id_list.append(sent_id)

        return mlm_pos, mask_id_list


    def masking_text(self, tokenized_text, formatted_data, lemmas, mask_id):
        masked_tokenized_text = copy.deepcopy(tokenized_text)
        mlm_pos = [0.] * len(masked_tokenized_text)
        
        # MLM
        if mask_id is not None:
            for i in range(len(masked_tokenized_text)):
                if i in mask_id:
                    mlm_pos[i] = 1.
            mask_id_list = mask_id
        else:
            mlm_pos, mask_id_list = self.masking_struct_tsg_1d(mlm_pos, masked_tokenized_text, formatted_data, lemmas)

        # Do Mask
        for i in range(len(masked_tokenized_text)):
            if mlm_pos[i] == 1.:
                masked_tokenized_text[i] = '[MASK]'

        return masked_tokenized_text, mask_id_list


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


    def rand_imgfeat_from_aggr(self, only_obj=True):
        if self.text_segment is not None:
            t_formatted_data = self.text_segment.load(random.randint(0, self.text_segment.total_len - 1))
        else:
            t_formatted_data = self.data_aggr[random.randint(0, len(self.data_aggr)-1)]
        img_src = t_formatted_data['img_src']
        img_filename = t_formatted_data['img_file']
        imgfeat = self.load_npz(img_src, img_filename)

        if only_obj:
            return imgfeat['x'][random.randint(0, imgfeat['num_boxes']-1), :]
        else:
            return imgfeat


    def masking_imgfeat_repalce(self, ids, masked_imgfeat_input):
        prob = random.random()
        if prob < self.__C.MASK_PROB_POST['mask']:
            if self.__C.MASK_IMGFEAT_WITH in ['zero']:
                masked_imgfeat_input[ids, :] = 0.
            elif self.__C.MASK_IMGFEAT_WITH in ['gaussian']:
                masked_imgfeat_input[ids, :] = np.random.randn(masked_imgfeat_input.shape[1])
            elif self.__C.MASK_IMGFEAT_WITH in ['uniform']:
                masked_imgfeat_input[ids, :] = np.random.rand(masked_imgfeat_input.shape[1])

        elif prob < self.__C.MASK_PROB_POST['mask'] + self.__C.MASK_PROB_POST['replace']:
            masked_imgfeat_input[ids, :] = self.rand_imgfeat_from_aggr(only_obj=True)
        
        return masked_imgfeat_input


    def masking_struct_bbox(self, ids, mlm_outside_weight, mlm_pos, ious, iou_thres, mask_id_list):
        for i in range(mlm_outside_weight.shape[0]):
            if ids == i:
                continue

            if ious[ids][i] > iou_thres:
                if random.random() <= self.__C.MASK_STRUCT_PROB_INSIDE['bbox']:
                    mlm_pos[i] = 1.
                    mask_id_list.append(i)

        return mlm_outside_weight, mlm_pos


    def masking_imgfeat(self, imgfeat_input, boxes, mask_id):
        masked_imgfeat_input = imgfeat_input.copy()
        mlm_outside_weight = np.zeros(imgfeat_input.shape[0], dtype=np.float32)
        mlm_pos = np.zeros(imgfeat_input.shape[0], dtype=np.float32)
        # ious = torch.from_numpy(DataSet.cal_iou(boxes))

        mask_id_list = []
        # MRM
        if mask_id is not None:
            mask_id_list = mask_id
            for i in range(imgfeat_input.shape[0]):
                if i in mask_id:
                    mlm_pos[i] = 1.

        else:
            for i in range(imgfeat_input.shape[0]):
                if random.random() < self.__C.MASK_PROB['image']:
                    mlm_pos[i] = 1.
                    mask_id_list.append(i)

        # Do Mask
        for i in range(imgfeat_input.shape[0]):
            if mlm_pos[i] == 1.:
                masked_imgfeat_input[i, :] = 0.

        return masked_imgfeat_input, mlm_outside_weight, mask_id_list


    def np_pad_1d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, (0, length - tensor.shape[0]), mode='constant', constant_values=value)


    def np_pad_2d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, ((0, length - tensor.shape[0]), (0, 0)), mode='constant', constant_values=value)
        

    def proc_imgfeat(
            self, imgfeat_input,
            imgfeat_bbox, length_pre):
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


    def get_score(self, occur, maxocc=10):
        assert maxocc in [1, 10]
        if maxocc == 10:
            if occur == 0:
                return .0
            elif occur == 1:
                return .3
            elif occur == 2:
                return .6
            elif occur == 3:
                return .9
            else:
                return 1.
        elif maxocc == 1:
            if occur == 0:
                return .0
            elif occur == 1:
                return 1.


    @staticmethod
    def cal_iou(box_list):
        box_listA = copy.deepcopy(box_list)
        box_listB = copy.deepcopy(box_list)

        box_listA = box_listA[:, np.newaxis, :]
        box_listB = box_listB[np.newaxis, :, :]

        ixmin = np.maximum(box_listA[:, :, 0], box_listB[:, :, 0])
        iymin = np.maximum(box_listA[:, :, 1], box_listB[:, :, 1])
        ixmax = np.minimum(box_listA[:, :, 2], box_listB[:, :, 2])
        iymax = np.minimum(box_listA[:, :, 3], box_listB[:, :, 3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        box_listA_area = (box_listA[:, :, 2] - box_listA[:, :, 0] + 1.) * (box_listA[:, :, 3] - box_listA[:, :, 1] + 1.)
        box_listB_area = (box_listB[:, :, 2] - box_listB[:, :, 0] + 1.) * (box_listB[:, :, 3] - box_listB[:, :, 1] + 1.)
        uni = (box_listA_area + box_listB_area - inters)
        iou = inters / uni

        return iou


    @staticmethod
    def cal_iou_unsymm(box_list):
        box_listA = copy.deepcopy(box_list)
        box_listB = copy.deepcopy(box_list)

        box_listA = box_listA[:, np.newaxis, :]
        box_listB = box_listB[np.newaxis, :, :]

        ixmin = np.maximum(box_listA[:, :, 0], box_listB[:, :, 0])
        iymin = np.maximum(box_listA[:, :, 1], box_listB[:, :, 1])
        ixmax = np.minimum(box_listA[:, :, 2], box_listB[:, :, 2])
        iymax = np.minimum(box_listA[:, :, 3], box_listB[:, :, 3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        box_listA_area = (box_listA[:, :, 2] - box_listA[:, :, 0] + 1.) * (box_listA[:, :, 3] - box_listA[:, :, 1] + 1.)
        iou = inters / box_listA_area

        return iou


    @staticmethod
    def cal_cosine_scalar(vec0, vec1):
        return 1 - torch.sum(vec0 * vec1, dim=-1) / (torch.norm(vec0, p=2, dim=-1) * torch.norm(vec1, p=2, dim=-1))

    @staticmethod
    def cal_cosine_vector(vec0, vec1):
        return vec0 * vec1 / (torch.norm(vec0, p=2, dim=-1, keepdim=True) * torch.norm(vec1, p=2, dim=-1, keepdim=True))

    @staticmethod
    def cal_dist_batch(vec0, vec1, type='scalar'):
        base0 = vec0.size(0)
        base1 = vec1.size(0)
        vec0 = vec0.unsqueeze(1).repeat(1, base1, 1)
        vec1 = vec1.unsqueeze(0).repeat(base0, 1, 1)
        
        cos_dist = None
        if type == 'scalar':
            cos_dist = DataSet.cal_cosine_scalar(vec0, vec1).unsqueeze(2)
            cos_dist = (cos_dist * 1.3).clamp(max=1.)
            cos_dist = torch.exp(cos_dist * 2 - 1.1) - 1

            max_value = math.exp(0.9) - 1
            min_value = math.exp(-1.1) - 1
            cos_dist = (cos_dist - min_value) / (max_value - min_value)

        elif type == 'vector':
            cos_dist =  DataSet.cal_cosine_vector(vec0, vec1)

        return cos_dist
