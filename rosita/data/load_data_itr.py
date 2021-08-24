# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import numpy as np
import json, re, torch, logging, collections, random, copy, os, base64
import torch.utils.data as Data
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
        self.data_aggr = []
        if text_segment is not None:
            logging.info('Use Text Segment for Memory Efficiency')
        else:
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

        tset, ttype = self.__C.DATASET_LIST[RUN_MODE][0].split(':')
        self.feat_to_ids, self.ids_to_feat, self.idx_to_feat_idx, self.feat_idx_to_idx = \
            json.load(open(os.path.join(__C.DATASET_PATHMAP[tset], 'img_text_map.json')))[ttype]
        assert self.data_size == len(self.feat_to_ids) * 5

        self.neg_text_hard_ids = torch.randint(high=self.data_size, size=(len(self.feat_to_ids), self.__C.NEG_HARDSIZE)).long()
        self.neg_img_hard_ids = torch.randint(high=self.data_size, size=(self.data_size, self.__C.NEG_HARDSIZE)).long()


    def __getitem__text(self, formatted_data):
        # Load text
        text = self.clean_text(formatted_data['text'])
        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        # Proc text
        text_input = tokenized_text
        text_input_ids, text_mask = self.proc_text(text_input)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)
        return text_input_ids, text_mask


    def getitem__img(self, formatted_data):
        # Load image features
        img_src = formatted_data['img_src']
        img_filename = formatted_data['img_file']
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

        # Proc image features
        imgfeat_x = imgfeat['x']
        image_h = int(imgfeat['image_h'])
        image_w = int(imgfeat['image_w'])
        boxes = imgfeat['boxes']
        imgfeat_input = imgfeat_x

        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))
        imgfeat_input, imgfeat_mask, imgfeat_bbox = self.proc_imgfeat(imgfeat_input, imgfeat_bbox)

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)

        return imgfeat_input, imgfeat_mask, imgfeat_bbox

    def load_formatted_data(self, idx):
        if self.text_segment is not None:
            formatted_data = self.text_segment.load(idx)
        else:
            formatted_data = self.data_aggr[idx]
        return formatted_data


    def __getitem__(self, idx):
        pos_text_idx = torch.tensor(idx).long()
        pos_img_idx = torch.tensor(self.idx_to_feat_idx[str(idx)]).long()
        neg_text_idx = self.neg_text_hard_ids[self.idx_to_feat_idx[str(idx)], random.randint(0, self.__C.NEG_HARDSIZE - 1)].item()
        neg_img_idx_idx = self.neg_img_hard_ids[idx, random.randint(0, self.__C.NEG_HARDSIZE - 1)].item()
        neg_img_idx = self.idx_to_feat_idx[str(neg_img_idx_idx)]
        assert self.idx_to_feat_idx[str(neg_text_idx)] != self.idx_to_feat_idx[str(pos_text_idx.item())]
        assert neg_img_idx != pos_img_idx.item()
        neg_text_idx = torch.tensor(neg_text_idx).long()
        neg_img_idx = torch.tensor(neg_img_idx).long()

        if self.__C.IMGFEAT_FORMAT == 'tsv':
            formatted_data = self.load_formatted_data(idx)
            imgfeat_input, imgfeat_mask, imgfeat_bbox = self.getitem__img(formatted_data)
            neg_formatted_data = self.load_formatted_data(neg_img_idx_idx)
            neg_imgfeat_input, neg_imgfeat_mask, neg_imgfeat_bbox = self.getitem__img(neg_formatted_data)

            return pos_text_idx, imgfeat_input, imgfeat_mask, imgfeat_bbox,\
                    neg_text_idx, neg_imgfeat_input, neg_imgfeat_mask, neg_imgfeat_bbox
        else:
            return pos_text_idx, pos_img_idx, neg_text_idx, neg_img_idx


    def __len__(self):
        return self.data_size


    def load_all_data(self):
        text_input_ids_all = []
        text_mask_all = []
        imgfeat_input_all = []
        imgfeat_mask_all = []
        imgfeat_bbox_all = []
        imgfeat_load_set = set()
        for idx in range(self.data_size):
            proc_rank = self.__C.GRANK if self.__C.MP_STORAGE_SHR['screen'] else self.__C.LRANK
            if idx % 5000 == 0 and proc_rank == 0:
                logging.info(f'All data loading [{idx / self.data_size * 100.}%]')
            formatted_data = self.load_formatted_data(idx)
            text_input_ids, text_mask = self.__getitem__text(formatted_data)
            text_input_ids_all.append(text_input_ids.unsqueeze(0))
            text_mask_all.append(text_mask.unsqueeze(0))
            if self.__C.IMGFEAT_FORMAT == 'tsv':
                continue
            if formatted_data['img_file'] not in imgfeat_load_set:
                assert formatted_data['img_file'] == self.ids_to_feat[str(len(imgfeat_load_set))]
                imgfeat_load_set.add(formatted_data['img_file'])
                imgfeat_input, imgfeat_mask, imgfeat_bbox = self.getitem__img(formatted_data)
                imgfeat_input_all.append(imgfeat_input.unsqueeze(0))
                imgfeat_mask_all.append(imgfeat_mask.unsqueeze(0))
                imgfeat_bbox_all.append(imgfeat_bbox.unsqueeze(0))
            assert self.idx_to_feat_idx[str(idx)] == len(imgfeat_load_set) - 1
        text_input_ids_all = torch.cat(text_input_ids_all, dim=0)
        text_mask_all = torch.cat(text_mask_all, dim=0)
        if self.__C.IMGFEAT_FORMAT == 'npz':
            imgfeat_input_all = torch.cat(imgfeat_input_all, dim=0)
            imgfeat_mask_all = torch.cat(imgfeat_mask_all, dim=0)
            imgfeat_bbox_all = torch.cat(imgfeat_bbox_all, dim=0)
            assert imgfeat_input_all.size(0) * 5 == text_input_ids_all.size(0)

        return text_input_ids_all, text_mask_all, imgfeat_input_all, imgfeat_mask_all, imgfeat_bbox_all


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


    def proc_imgfeat(self, imgfeat_input, imgfeat_bbox):
        length_pad = self.__C.PAD_MAX['image']

        imgfeat_mask = torch.ones(imgfeat_input.shape[0], dtype=torch.float32)
        imgfeat_mask = self.np_pad_1d(imgfeat_mask, length_pad)
        imgfeat_input = self.np_pad_2d(imgfeat_input, length_pad)
        imgfeat_bbox = self.np_pad_2d(imgfeat_bbox, length_pad)

        return imgfeat_input, imgfeat_mask, imgfeat_bbox


    def proc_bbox(self, bbox, img_shape):
        bbox = copy.deepcopy(bbox)
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat


class DataSet_Neg(Data.Dataset):
    def __init__(self, __C, keep, RUN_MODE, text_segment=None, spacy_tool=None):
        self.__C = __C
        assert keep in ['text', 'img']
        self.keep = keep
        self.text_segment = text_segment
        assert self.__C.IMGFEAT_FORMAT in ['npz', 'tsv']
        self.use_tsv = False
        if self.__C.IMGFEAT_FORMAT == 'tsv':
            self.use_tsv = True
        logging.info(f'Negative [Keep {keep}] Loader Initializing')

        if self.use_tsv:
            self.tsv_files = {}
            self.img_feat_offset_maps = {}
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                fset = self.__C.DATASET_FEATMAP[tset]
                tsv_file, img_feat_offset_map = self.load_tsv(fset)
                self.tsv_files[fset] = tsv_file
                self.img_feat_offset_maps[fset] = img_feat_offset_map

        self.data_aggr = []
        if text_segment is not None:
            logging.info('Use Text Segment for Memory Efficiency')
        else:
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

        tset, ttype = self.__C.DATASET_LIST[RUN_MODE][0].split(':')
        self.feat_to_ids, self.ids_to_feat, self.idx_to_feat_idx, self.feat_idx_to_idx = \
            json.load(open(os.path.join(__C.DATASET_PATHMAP[tset], 'img_text_map.json')))[ttype]
        assert self.data_size == len(self.feat_to_ids) * 5


    def __getitem__text(self, formatted_data):
        # Load text
        text = self.clean_text(formatted_data['text'])
        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        # Proc text
        text_input = tokenized_text
        text_input_ids, text_mask = self.proc_text(text_input)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)
        # print(text_input_ids)
        return text_input_ids, text_mask

    def __getitem__img(self, formatted_data):
        # Load image features
        img_src = formatted_data['img_src']
        img_filename = formatted_data['img_file']
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

        # Proc image features
        imgfeat_x = imgfeat['x']
        image_h = int(imgfeat['image_h'])
        image_w = int(imgfeat['image_w'])
        boxes = imgfeat['boxes']
        imgfeat_input = imgfeat_x

        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))
        imgfeat_input, imgfeat_mask, imgfeat_bbox = self.proc_imgfeat(imgfeat_input, imgfeat_bbox)

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)

        return imgfeat_input, imgfeat_mask, imgfeat_bbox

    def load_formatted_data(self, idx):
        if self.text_segment is not None:
            formatted_data = self.text_segment.load(idx)
        else:
            formatted_data = self.data_aggr[idx]
        return formatted_data


    def __getitem__(self, idx):
        if self.keep == 'text':
            text_idx = torch.tensor([idx for _ in range(self.__C.NEG_RANDSIZE)]).long()
            img_idx = torch.zeros(self.__C.NEG_RANDSIZE).long()
            neg_idx = torch.zeros(self.__C.NEG_RANDSIZE).long()
            for step in range(self.__C.NEG_RANDSIZE):
                rid = random.randint(0, self.data_size - 1)
                while self.idx_to_feat_idx[str(idx)] == self.idx_to_feat_idx[str(rid)]:
                    rid = random.randint(0, self.data_size - 1)
                assert self.idx_to_feat_idx[str(idx)] != self.idx_to_feat_idx[str(rid)]
                img_idx[step] = self.idx_to_feat_idx[str(rid)]
                neg_idx[step] = rid

        else:
            img_idx = torch.tensor([idx for _ in range(self.__C.NEG_RANDSIZE)]).long()
            text_idx = torch.zeros(self.__C.NEG_RANDSIZE).long()
            neg_idx = torch.zeros(self.__C.NEG_RANDSIZE).long()
            for step in range(self.__C.NEG_RANDSIZE):
                rid = random.randint(0, self.data_size - 1)
                while idx == self.idx_to_feat_idx[str(rid)]:
                    rid = random.randint(0, self.data_size - 1)
                assert idx != self.idx_to_feat_idx[str(rid)]
                text_idx[step] = rid
                neg_idx[step] = rid

        if self.use_tsv:
            imgfeat_input_all, imgfeat_mask_all, imgfeat_bbox_all = [], [], []
            for i in range(img_idx.size(0)):
                sample_idx = self.feat_idx_to_idx[str(int(img_idx[i]))]
                formatted_data = self.load_formatted_data(sample_idx)
                imgfeat_input, imgfeat_mask, imgfeat_bbox = self.__getitem__img(formatted_data)
                imgfeat_input_all.append(imgfeat_input.unsqueeze(0))
                imgfeat_mask_all.append(imgfeat_mask.unsqueeze(0))
                imgfeat_bbox_all.append(imgfeat_bbox.unsqueeze(0))
            imgfeat_input_all = torch.cat(imgfeat_input_all, dim=0)
            imgfeat_mask_all = torch.cat(imgfeat_mask_all, dim=0)
            imgfeat_bbox_all = torch.cat(imgfeat_bbox_all, dim=0)
            return text_idx, imgfeat_input_all, imgfeat_mask_all, imgfeat_bbox_all, neg_idx
        else:
            return text_idx, img_idx, neg_idx


    def __len__(self):
        if self.keep == 'text':
            return self.data_size
        else:
            return len(self.feat_to_ids)


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

    def proc_imgfeat(self, imgfeat_input, imgfeat_bbox):
        length_pad = self.__C.PAD_MAX['image']

        imgfeat_mask = torch.ones(imgfeat_input.shape[0], dtype=torch.float32)
        imgfeat_mask = self.np_pad_1d(imgfeat_mask, length_pad)
        imgfeat_input = self.np_pad_2d(imgfeat_input, length_pad)
        imgfeat_bbox = self.np_pad_2d(imgfeat_bbox, length_pad)

        return imgfeat_input, imgfeat_mask, imgfeat_bbox

    def proc_bbox(self, bbox, img_shape):
        bbox = copy.deepcopy(bbox)
        bbox_feat = np.zeros((bbox.shape[0], 5), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])
        bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox_feat

