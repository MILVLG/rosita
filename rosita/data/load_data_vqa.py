# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import numpy as np
import json, re, torch, logging, collections, copy, math, os, base64
import torch.utils.data as Data
import torch.nn as nn
from utils.answer_punct import preprocess_answer
from utils.tokenizer import BertTokenizer
from utils.tsv_file import TSVFile


class DataSet(Data.Dataset):
    def __init__(self, __C, RUN_MODE, text_segment=None):
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

        self.ans_to_ix, self.ix_to_ans = DataSet.load_ans_vocab()
        self.ans_size = len(self.ans_to_ix)
        logging.info('[dataset: {}] Total Ans: {}'.format(RUN_MODE, len(self.ans_to_ix)))

        logging.info('[dataset: {}] Loader Initialized'.format(RUN_MODE))


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
    def load_ans_vocab():
        return json.load(open('rosita/utils/vqa/answer_vocab.json', 'r'))


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

