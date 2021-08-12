import numpy as np
import json, re, torch, logging, collections, copy, math, os
import torch.utils.data as Data
import torch.nn as nn

from rosita.utils.answer_punct import preprocess_answer
from rosita.utils.tokenizer import BertTokenizer


class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE, text_segment=None):
        self.__C = __C
        self.RUN_MODE = RUN_MODE
        self.text_segment = text_segment
        logging.info('[dataset: {}] Loader Initializing'.format(RUN_MODE))

        if text_segment is not None:
            logging.info('Use Text Segment for Memory Efficiency')
        else:
            self.data_aggr = []
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                ttype = dataset_name.split(':')[1]
                formatted_data = json.load(open(__C.DATASET_ROOTPATH_MAP[tset], 'r'))[ttype]
                self.data_aggr += formatted_data
                logging.info('[dataset: {}] Loading [{}] data: {}'.format(RUN_MODE, dataset_name, len(formatted_data)))
            logging.info('[dataset: {}] Total Data: {}'.format(RUN_MODE, len(self.data_aggr)))

        self.tokenizer = BertTokenizer(self.load_vocab(__C.BERT_VOCAB_PATH))
        self.vocab_size = len(self.tokenizer.vocab)
        logging.info('[dataset: {}] Total Vocab: {}'.format(RUN_MODE, self.vocab_size))

        self.ans_to_ix, self.ix_to_ans = DataSet.load_ans_tabel(__C.ANS_TABEL)
        self.ans_size = len(self.ans_to_ix)
        logging.info('[dataset: {}] Total Ans: {}'.format(RUN_MODE, len(self.ans_to_ix)))

        logging.info('[dataset: {}] Loader Initialized'.format(RUN_MODE))
        
        obj_vocab = open('rosita/utils/genome_vocabs/objects_vocab.txt', 'r').readlines()
        self.obj_id_map = {i: v.strip().split(',')[0] for i, v in enumerate(obj_vocab)}

        attr_vocab = open('rosita/utils/genome_vocabs/attributes_vocab.txt', 'r').readlines()
        self.attr_id_map = {i: v.strip().split(',')[0] for i, v in enumerate(attr_vocab)}


    def __getitem__(self, idx):
        if self.text_segment is not None:
            formatted_data = self.text_segment.load(idx)
        else:
            formatted_data = self.data_aggr[idx]

        # Load text and image features
        formatted_data_text = formatted_data
        formatted_data_img = formatted_data

        # Load text
        text = self.clean_text(formatted_data_text['text'])

        # Load image features
        img_src = formatted_data_img['img_src']
        img_filename = formatted_data_img['img_file']
        imgfeat = self.load_npz(img_src, img_filename)

        # Proc qa tasks
        is_qa = self.RUN_MODE in ['train'] and 'qa' in self.__C.TASKS['mm']
        is_qa = is_qa and formatted_data_text['label'] not in ['', None]

        qa_loss_func = self.__C.LOSSFUNC_MAPPING['mm']['qa']
        if qa_loss_func in [nn.CrossEntropyLoss]:
            qa_label = torch.tensor(-1, dtype=torch.int64)
        elif qa_loss_func in [nn.BCEWithLogitsLoss, nn.KLDivLoss]:
            qa_label = torch.zeros(len(self.ans_to_ix), dtype=torch.float32)

        qa_loss_valid = torch.tensor(0, dtype=torch.float32)
        if is_qa:
            qa_label, qa_loss_valid = self.proc_qa(formatted_data_text, qa_loss_func)
        
        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        self.check_tsg(text, tokenized_text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        # Masking text
        text_input = copy.deepcopy(tokenized_text)

        # Padding and convert text ids
        text_input_ids, text_mask = self.proc_text(text_input)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)

        # Masking image features
        imgfeat_x = imgfeat['x']
        image_h = imgfeat['image_h']
        image_w = imgfeat['image_w']
        boxes = imgfeat['boxes']

        imgfeat_input = imgfeat_x

        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))

        imgfeat_input, imgfeat_mask, imgfeat_bbox, \
         = self.proc_imgfeat(
            imgfeat_input, imgfeat_bbox, text_input_ids.size(0))

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)

        # Get text and image id
        text_id = formatted_data_text['text_id']

        return  text_input_ids, text_mask, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, qa_label, qa_loss_valid, text_id


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
        assert not vocab_file.startswith('oss://')

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
    def load_ans_tabel(ans_tabel):
        return json.load(open('rosita/utils/answer_tabels/answer_tabel_[{}].json'.format(ans_tabel), 'r'))

    
    @staticmethod
    def filter_ans(__C, stat_ans_list, is_punct, ans_freq_thresh):
        ans_set = set()
        ans_freq_map = {}

        for stat_ans_name in stat_ans_list:
            tset = stat_ans_name.split(':')[0]
            ttype = stat_ans_name.split(':')[1]
            t_formatted_data = json.load(open(__C.DATASET_ROOTPATH_MAP[tset], 'r'))[ttype]

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
        np_file = os.path.join(self.__C.IMGFEAT_ROOTPATH_MAP[img_src], (img_filename + '.npz'))
        npz_loaded = np.load(np_file)

        return npz_loaded


    def proc_text(self, text_input):
        # concatenate lm labels and account for CLS, SEP
        text_input = ['[CLS]'] + text_input + ['[SEP]']
        text_input_ids = self.tokenizer.convert_tokens_to_ids(text_input)

        # Mask & Segment Word
        text_mask = [1] * len(text_input_ids)

        pad_length = self.__C.PAD_MAX['text'] - len(text_input_ids)
        if self.__C.PAD_INSIDE and len(text_input_ids) < self.__C.PAD_MAX['text']:
            text_input_ids += [0] * pad_length
            text_mask += [0] * pad_length

        return text_input_ids, text_mask


    def np_pad_1d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, (0, length - tensor.shape[0]), mode='constant', constant_values=value)


    def np_pad_2d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, ((0, length - tensor.shape[0]), (0, 0)), mode='constant', constant_values=value)
        

    def proc_imgfeat(
            self, imgfeat_input, imgfeat_bbox, length_pre):
        length_pad = self.__C.PAD_MAX['image']  + self.__C.PAD_MAX['text'] - length_pre

        imgfeat_mask = torch.ones(imgfeat_input.shape[0], dtype=torch.float32)
        imgfeat_mask = self.np_pad_1d(imgfeat_mask, length_pad)

        imgfeat_input = self.np_pad_2d(imgfeat_input, length_pad)
        imgfeat_bbox = self.np_pad_2d(imgfeat_bbox, length_pad)

        return imgfeat_input, imgfeat_mask, imgfeat_bbox


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

    
    def proc_qa(self, formatted_data, qa_loss_func):
        label_prob_map = {}
        for label in formatted_data['multi_label']:
            if self.__C.PUNCT_ANS_MAP[formatted_data['text_set']]:
                label = preprocess_answer(label)
            if label not in label_prob_map:
                label_prob_map[label] = 1
            else:
                label_prob_map[label] += 1

        if qa_loss_func == nn.CrossEntropyLoss:
            if self.__C.MULTINOMIAL_QA_LABEL:
                label_prob_list = []
                label_list = []
                for label in label_prob_map:
                    label_prob_list.append(label_prob_map[label] / len(formatted_data['multi_label']))
                    label_list.append(label)
                label = label_list[np.random.multinomial(1, label_prob_list).argmax()]
            else:
                label = formatted_data['label']
                if self.__C.PUNCT_ANS_MAP[formatted_data['text_set']]:
                    label = preprocess_answer(label)

            if label in self.ans_to_ix:
                qa_label = torch.tensor(self.ans_to_ix[label], dtype=torch.int64)
                qa_loss_valid = torch.tensor(1, dtype=torch.float32)
            else:
                qa_label = torch.tensor(-1, dtype=torch.int64)
                qa_loss_valid = torch.tensor(0, dtype=torch.float32)

        elif qa_loss_func == nn.BCEWithLogitsLoss:
            qa_loss_valid = torch.tensor(1, dtype=torch.float32)
            qa_label = torch.zeros(len(self.ans_to_ix), dtype=torch.float32)
            for label in label_prob_map:
                if label in self.ans_to_ix:
                    qa_label[self.ans_to_ix[label]] = self.get_score(label_prob_map[label], len(formatted_data['multi_label']))

        elif qa_loss_func == nn.KLDivLoss:
            qa_loss_valid = torch.tensor(1, dtype=torch.float32)
            qa_label = torch.zeros(len(self.ans_to_ix), dtype=torch.float32)
            for label in label_prob_map:
                if label in self.ans_to_ix:
                    qa_label[self.ans_to_ix[label]] = label_prob_map[label] / len(formatted_data['multi_label'])

        return qa_label, qa_loss_valid


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
