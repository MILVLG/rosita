# -------------------------------------------------------- 
# ROSITA
# Licensed under The Apache License 2.0 [see LICENSE for details] 
# Written by Yuhao Cui and Tong-An Luo
# -------------------------------------------------------- 

import numpy as np
import json, re, torch, logging, collections, random, copy, math, os, base64, glob
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
        self.use_tsv_feat = False
        assert self.__C.ANNO_FORMAT in ['json', 'tsv']
        self.use_tsv_anno = False
        if self.__C.IMGFEAT_FORMAT == 'tsv':
            self.use_tsv_feat = True
        if self.__C.ANNO_FORMAT == 'tsv':
            self.use_tsv_anno = True
            self.on_memory = self.__C.TSV_ON_MEMORY
        else:
            self.on_memory = True
        logging.info('[dataset: {}] Loader Initializing'.format(RUN_MODE))

        if self.use_tsv_feat:
            self.datasets_with_splits = ['sbu', 'conceptual']
            self.split_size = 300000
            self.tsv_files = {}
            self.img_feat_offset_maps = {}
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                fset = self.__C.DATASET_FEATMAP[tset]
                if fset in self.datasets_with_splits:
                    tsv_file, img_feat_offset_map = self.load_tsv_with_split(fset)  # tsv_file is a list of TSVFile
                else:
                    tsv_file, img_feat_offset_map = self.load_tsv(fset) # tsv_file is a TSVFile
                self.tsv_files[fset] = tsv_file
                self.img_feat_offset_maps[fset] = img_feat_offset_map
        if text_segment is not None:
            logging.info('Use Text Segment for Memory Efficiency')
        else:
            self.data_aggr = []
            self.data_tsv = []
            self.data_start_idx = []
            self.all_data_num = 0
            for dataset_name in self.__C.DATASET_LIST[RUN_MODE]:
                tset = dataset_name.split(':')[0]
                ttype = dataset_name.split(':')[1]
                if self.use_tsv_anno:
                    logging.info('Use tsv format data for Memory Efficiency')
                    formatted_data_tsv = TSVFile(__C.DATASET_ANNO_MAP[tset] + '_{}.tsv'.format(ttype))
                    if self.on_memory:
                        for row_idx in range(len(formatted_data_tsv)):
                            row = formatted_data_tsv.seek(row_idx)
                            if RUN_MODE == 'train':
                                self.data_aggr.append(self.formatted_tsv_row(row))
                            else:
                                self.data_aggr.append(self.formatted_tsv_row2(row))
                    else:
                        self.data_tsv.append(formatted_data_tsv)
                        self.data_start_idx.append(self.all_data_num)
                    data_num = len(formatted_data_tsv)
                    self.all_data_num += data_num
                else:
                    formatted_data = json.load(open(__C.DATASET_ANNO_MAP[tset] + '.json', 'r'))[ttype]
                    self.data_aggr += formatted_data
                    data_num = len(formatted_data)
                logging.info('[dataset: {}] Loading [{}] data: {}'.format(RUN_MODE, dataset_name, data_num))
            logging.info('[dataset: {}] Total Data: {}'.format(RUN_MODE, len(self.data_aggr)))

        self.tokenizer = BertTokenizer(self.load_vocab(__C.BERT_VOCAB_PATH))
        self.vocab_size = len(self.tokenizer.vocab)
        logging.info('[dataset: {}] Total Vocab: {}'.format(RUN_MODE, self.vocab_size))

        self.ans_to_ix, self.ix_to_ans = DataSet.load_ans_vocab()
        self.ans_size = len(self.ans_to_ix)
        logging.info('[dataset: {}] Total Ans: {}'.format(RUN_MODE, len(self.ans_to_ix)))

        logging.info('[dataset: {}] Loader Initialized'.format(RUN_MODE))
        
        obj_vocab = open('rosita/utils/genome_vocabs/objects_vocab.txt', 'r').readlines()
        self.obj_id_map = {i: v.strip().split(',')[0] for i, v in enumerate(obj_vocab)}

        attr_vocab = open('rosita/utils/genome_vocabs/attributes_vocab.txt', 'r').readlines()
        self.attr_id_map = {i: v.strip().split(',')[0] for i, v in enumerate(attr_vocab)}

        self.spacy_tool = spacy_tool
        text_piror = json.load(open(os.path.join(__C.DATASET_PATHMAP['pt-coco'], 'text_piror_coco_vg_cc_sbu.json'), 'r'))
        self.attr_count = text_piror['attr_count']
        self.rel_count = text_piror['rel_count']


    def __getitem__(self, idx):
        if self.text_segment is not None:
            formatted_data = self.text_segment.load(idx)
        elif self.on_memory:
            formatted_data = self.data_aggr[idx]
        else:
            for tsv_idx in range(len(self.data_start_idx)-1, -1, -1):
                if idx >= self.data_start_idx[tsv_idx]:
                    break
            tsv_data_idx = idx - self.data_start_idx[tsv_idx]
            row = self.data_tsv[tsv_idx].seek(tsv_data_idx)
            if self.RUN_MODE == 'train':
                formatted_data = self.formatted_tsv_row(row)
            else:
                formatted_data = self.formatted_tsv_row2(row)

        # Proc itm tasks
        is_itm = self.RUN_MODE in ['train'] and 'itm' in self.__C.TASKS['mm']
        is_matched = True
        itm_label = torch.tensor(-1, dtype=torch.int64)
        itm_loss_valid = torch.tensor(0, dtype=torch.float32)
        if is_itm:
            is_matched = random.random() > 0.5
            itm_label = torch.tensor(int(is_matched), dtype=torch.int64)
            itm_loss_valid = torch.tensor(1, dtype=torch.float32)

        # Load text and image features
        if is_matched:
            formatted_data_text = formatted_data
            formatted_data_img = formatted_data
        else:
            if self.__C.MATCH_NEG_SHUFFLE in ['text']:
                formatted_data_img = formatted_data
                formatted_data_text = self.sample_negative(formatted_data)
            elif self.__C.MATCH_NEG_SHUFFLE in ['image']:
                formatted_data_text = formatted_data
                formatted_data_img = self.sample_negative(formatted_data)

        # Load text
        text = self.clean_text(formatted_data_text['text'])
        lemmas = self.clean_text(formatted_data_text['lemmas'])

        # Load image features
        img_src = formatted_data_img['img_src']
        img_filename = formatted_data_img['img_file']
        if self.use_tsv_feat:
            tsv_file = self.tsv_files[img_src]
            img_offset_map = self.img_feat_offset_maps[img_src]
            img_idx = img_offset_map[img_filename]
            if img_src in self.datasets_with_splits:
                chunk_idx = img_idx // self.split_size
                img_idx = img_idx - chunk_idx * self.split_size
                row = tsv_file[chunk_idx].seek(img_idx)
            else:
                row = tsv_file.seek(img_idx)
            imgfeat = {}
            imgfeat['filename'] = row[0]
            num_bboxes = int(row[4])
            imgfeat['x'] = np.frombuffer(base64.b64decode(row[1]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['image_h'], imgfeat['image_w'] = int(row[2]), int(row[3])
            imgfeat['num_boxes'] = num_bboxes
            imgfeat['boxes'] = np.frombuffer(base64.b64decode(row[5]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['objects_id'] = np.frombuffer(base64.b64decode(row[6]), dtype=np.int64)
            imgfeat['objects_conf'] = np.frombuffer(base64.b64decode(row[7]), dtype=np.float32)
            imgfeat['attrs_id'] = np.frombuffer(base64.b64decode(row[8]), dtype=np.int64)
            imgfeat['attrs_conf'] = np.frombuffer(base64.b64decode(row[9]), dtype=np.float32)

        else:
            imgfeat = self.load_npz(img_src, img_filename)

        # Proc qa tasks
        is_qa = self.RUN_MODE in ['train'] and 'qa' in self.__C.TASKS['mm']
        is_qa = is_qa and (not (not is_matched and 'qa' in self.__C.MATCH_CONSTRAIN))
        is_qa = is_qa and formatted_data_text['label'] not in ['', None]

        qa_loss_func = self.__C.LOSSFUNC_MAPPING['mm']['qa']
        if qa_loss_func in [nn.CrossEntropyLoss]:
            qa_label = torch.tensor(-1, dtype=torch.int64)
        elif qa_loss_func in [nn.BCEWithLogitsLoss, nn.KLDivLoss]:
            qa_label = torch.zeros(len(self.ans_to_ix), dtype=torch.float32)

        qa_loss_valid = torch.tensor(0, dtype=torch.float32)
        if is_qa:
            qa_label, qa_loss_valid = self.proc_qa(formatted_data_text, qa_loss_func)
            
        # Proc masking tasks
        rand_mask_side_prob = random.random()
        is_text_masking = self.RUN_MODE in ['train'] and len(self.__C.TASKS['text']) > 0
        is_text_masking = is_text_masking and (self.__C.MASK_SIDE_PROB < 0 or rand_mask_side_prob < self.__C.MASK_SIDE_PROB)
        is_imgfeat_masking = self.RUN_MODE in ['train'] and len(self.__C.TASKS['image']) > 0
        is_imgfeat_masking = is_imgfeat_masking and (self.__C.MASK_SIDE_PROB < 0 or rand_mask_side_prob >= self.__C.MASK_SIDE_PROB)

        # Cliping text
        tokenized_text = self.tokenizer.tokenize(text)
        self.check_tsg(text, tokenized_text)
        if len(tokenized_text) > self.__C.PAD_MAX['text'] - 2:
            tokenized_text = tokenized_text[:(self.__C.PAD_MAX['text'] - 2)]

        grain_label = self.proc_grain(lemmas, tokenized_text, formatted_data_text, imgfeat['objects_id'], imgfeat['attrs_id'])

        # Masking text
        text_input = copy.deepcopy(tokenized_text)
        text_mlm_label = copy.deepcopy(tokenized_text)
        text_outside_weight = [0.] * len(tokenized_text)
        text_mlm_loss_valid = torch.tensor(0, dtype=torch.float32)
        ros_obj_id = None
        if is_text_masking:
            # Masking text
            # print(is_matched)
            do_mask_loss = not (not is_matched and 'text' in self.__C.MATCH_CONSTRAIN)
            text_input, text_outside_weight, ros_obj_id = self.masking_text(text_input, formatted_data_text, grain_label, imgfeat['boxes'], lemmas, do_mask_loss)
            if 'mlm' in self.__C.TASKS['text']:
                text_mlm_loss_valid = torch.tensor(1, dtype=torch.float32)

        # Padding and convert text ids
        text_input_ids, text_mask, text_outside_weight, text_mlm_label_ids, text_mlm_weight, grain_label = self.proc_text(
            text_input, text_outside_weight, text_mlm_label, grain_label)
        text_input_ids = torch.tensor(text_input_ids, dtype=torch.int64)
        text_mask = torch.tensor(text_mask, dtype=torch.float32)
        text_outside_weight = torch.tensor(text_outside_weight, dtype=torch.float32)
        text_mlm_label_ids = torch.tensor(text_mlm_label_ids, dtype=torch.int64)
        text_mlm_weight = torch.tensor(text_mlm_weight, dtype=torch.float32)

        # Masking image features
        imgfeat_x = imgfeat['x']
        image_h = imgfeat['image_h']
        image_w = imgfeat['image_w']
        boxes = imgfeat['boxes']
        objects_id = imgfeat['objects_id']
        objects_conf = imgfeat['objects_conf']
        attrs_id = imgfeat['attrs_id']
        attrs_conf = imgfeat['attrs_conf']

        imgfeat_input = imgfeat_x
        imgfeat_feat_label = imgfeat_x
        imgfeat_feat_weight = np.ones(imgfeat_feat_label.shape[0], dtype=np.float32)
        imgfeat_outside_weight = np.zeros(imgfeat_feat_label.shape[0], dtype=np.float32)
        imgfeat_obj_label = objects_id
        imgfeat_obj_weight = objects_conf
        imgfeat_attr_label = attrs_id
        imgfeat_attr_weight = attrs_conf

        imgfeat_feat_loss_valid = torch.tensor(0, dtype=torch.float32)
        imgfeat_obj_loss_valid = torch.tensor(0, dtype=torch.float32)
        imgfeat_attr_loss_valid = torch.tensor(0, dtype=torch.float32)
        ros_word_id = None
        if is_imgfeat_masking:
            # Masking img features
            # print(is_matched)
            do_mask_loss = not (not is_matched and 'image' in self.__C.MATCH_CONSTRAIN)
            imgfeat_input, imgfeat_outside_weight, ros_word_id = self.masking_imgfeat(imgfeat_input, boxes, grain_label, formatted_data_text, tokenized_text, lemmas, do_mask_loss)
            if 'feat' in self.__C.TASKS['image']:
                imgfeat_feat_loss_valid = torch.tensor(1, dtype=torch.float32)
            if 'obj' in self.__C.TASKS['image']:
                imgfeat_obj_loss_valid = torch.tensor(1, dtype=torch.float32)
            if 'attr' in self.__C.TASKS['image']:
                imgfeat_attr_loss_valid = torch.tensor(1, dtype=torch.float32)


        # Padding and process bbox relation
        imgfeat_bbox = self.proc_bbox(boxes, (image_h, image_w))

        imgfeat_input, imgfeat_mask, imgfeat_outside_weight, imgfeat_feat_label, imgfeat_feat_weight, \
        imgfeat_obj_label, imgfeat_obj_weight, imgfeat_attr_label, imgfeat_attr_weight, imgfeat_bbox, \
         = self.proc_imgfeat(
            imgfeat_input, imgfeat_outside_weight, imgfeat_feat_label, imgfeat_feat_weight, imgfeat_obj_label,
            imgfeat_obj_weight, imgfeat_attr_label, imgfeat_attr_weight, imgfeat_bbox, text_input_ids.size(0))

        imgfeat_input = torch.from_numpy(imgfeat_input)
        imgfeat_mask = torch.from_numpy(imgfeat_mask)
        imgfeat_outside_weight = torch.from_numpy(imgfeat_outside_weight)
        imgfeat_feat_label = torch.from_numpy(imgfeat_feat_label)
        imgfeat_feat_weight = torch.from_numpy(imgfeat_feat_weight)
        imgfeat_obj_label = torch.from_numpy(imgfeat_obj_label)
        imgfeat_obj_weight = torch.from_numpy(imgfeat_obj_weight)
        imgfeat_attr_label = torch.from_numpy(imgfeat_attr_label)
        imgfeat_attr_weight = torch.from_numpy(imgfeat_attr_weight)
        imgfeat_bbox = torch.from_numpy(imgfeat_bbox)


        # Get text and image id
        text_id = formatted_data_text['text_id']
        img_id = formatted_data_img['img_id']

        return  text_input_ids, text_mask, text_outside_weight, \
                text_mlm_label_ids, text_mlm_weight, text_mlm_loss_valid, \
                imgfeat_input, imgfeat_mask, imgfeat_bbox, imgfeat_outside_weight, \
                imgfeat_feat_label, imgfeat_feat_weight, imgfeat_feat_loss_valid, \
                imgfeat_obj_label, imgfeat_obj_weight, imgfeat_obj_loss_valid, \
                imgfeat_attr_label, imgfeat_attr_weight, imgfeat_attr_loss_valid, \
                itm_label, itm_loss_valid, qa_label, qa_loss_valid, text_id, img_id



    def __len__(self):
        if self.text_segment is not None:
            return self.text_segment.total_len
        elif self.on_memory:
            return len(self.data_aggr)
        else:
            return self.all_data_num
    
    def formatted_tsv_row(self, row):
        data_dict = {
            'type': row[0], 'text_set': row[1], 'text_split': row[2], 'text_id': row[3], 'text': json.loads(row[4]), 
            'lemmas': json.loads(row[5]), 'img_src': row[6], 'img_id': row[7], 'img_file': row[8], 
            'split_info': row[9], 'label': json.loads(row[10]), 'multi_label': json.loads(row[11]), 
            'tsg': json.loads(row[12]), 
        }
        return data_dict

    def formatted_tsv_row2(self, row):
        data_dict = {
            'type': row[0], 'text_set': row[1], 'text_split': row[2], 'text_id': row[3], 'text': json.loads(row[4]), 
            'lemmas': json.loads(row[5]), 'img_src': row[6], 'img_id': row[7], 'img_file': row[8], 
            'split_info': row[9], 'label': json.loads(row[10]), 'multi_label': json.loads(row[11]), 
        }
        return data_dict


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
    def load_ans_vocab():
        return json.load(open('rosita/utils/vqa/answer_vocab.json', 'r'))

    
    @staticmethod
    def filter_ans(__C, stat_ans_list, is_punct, ans_freq_thresh):
        ans_set = set()
        ans_freq_map = {}

        for stat_ans_name in stat_ans_list:
            tset = stat_ans_name.split(':')[0]
            ttype = stat_ans_name.split(':')[1]
            t_formatted_data = json.load(open(__C.DATASET_ANNO_MAP[tset], 'r'))[ttype]

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
    
    def load_tsv_with_split(self, img_src):
        tsv_files = self.__C.IMGFEAT_PATHMAP[img_src] + '/imgfeat_split*.tsv'
        split_num = len(glob.glob(tsv_files))
        img_feat_offset_map_file = self.__C.IMGFEAT_PATHMAP[img_src] + '/img_feat_offset_map.json'
        with open(img_feat_offset_map_file) as f:
            img_feat_offset_map = json.load(f)
        tsv_file_list = []
        for i in range(split_num):
            tsv_file = self.__C.IMGFEAT_PATHMAP[img_src] + '/imgfeat_split{}.tsv'.format(i)
            tsv_file_list.append(TSVFile(tsv_file))
        return tsv_file_list, img_feat_offset_map


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
        mixed_ids = text_sg['mixed']

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
        objs_ids = text_sg['objs']
        attrs_ids = text_sg['attrs']
        rels_ids = text_sg['rels']
        mixed_ids = text_sg['mixed']
        all_sg_ids = objs_ids + attrs_ids + rels_ids

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


    def masking_struct_tsg_1d(self, mlm_outside_weight, mlm_pos, tokenized_text, formatted_data, lemmas, do_mask_loss=True):
        bpe_ids = []
        for step, text in enumerate(tokenized_text):
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)
        tokens = lemmas.strip().split()
        text_sg = formatted_data['tsg']
        objs_ids = text_sg['objs']
        attr_map, attr_map_ids, rel_map, rel_map_ids = self.tsg_map_to_obj(formatted_data, tokens)

        mask_ids = []
        mask_prob = torch.zeros(len(tokens), dtype=torch.float32)
        for objs_id in objs_ids:
            if random.random() < (self.__C.MASK_STRUCT_PROB['tsg'] + 0.2):
                mask_ids.append(objs_id[0][0])
                cent = tokens[objs_id[0][0]]
                if cent in attr_map:
                    for cent_i, cent_v in enumerate(attr_map[cent]):
                        prob = self.attr_count[cent][cent_v]
                        prob = min(prob * 5., 0.5) + 0.2
                        inds = attr_map_ids[cent][cent_i]
                        for ind in inds:
                            mask_prob[ind] = max(mask_prob[ind], prob)

                if cent in rel_map:
                    for cent_i, cent_v in enumerate(rel_map[cent]):
                        prob = self.rel_count[cent][cent_v]
                        prob = min(prob * 20., 0.5) + 0.2
                        inds = rel_map_ids[cent][cent_i]
                        for ind in inds:
                            mask_prob[ind] = max(mask_prob[ind], prob)
        for i in range(mask_prob.size(0)):
            if random.random() < mask_prob[i]:
                mask_ids.append(i)

        # mask
        for mask_id in mask_ids:
            if mask_id < len(bpe_ids):
                for sent_id in bpe_ids[mask_id]:
                    mlm_pos[sent_id] = 1.
                    # mlm_outside_weight[sent_id] = 1.
                    if do_mask_loss:
                        mlm_outside_weight[sent_id] = 1.

        return mlm_outside_weight, mlm_pos


    def masking_text(self, tokenized_text, formatted_data, grain_label, boxes, lemmas, do_mask_loss=True):
        masked_tokenized_text = copy.deepcopy(tokenized_text)
        mlm_outside_weight = [0.] * len(masked_tokenized_text)
        mlm_pos = [0.] * len(masked_tokenized_text)
        
        # MLM
        for i in range(len(masked_tokenized_text)):
            if random.random() < self.__C.MASK_PROB['text']:
                mlm_pos[i] = 1.
                # append current token to output (we will predict these later)
                if do_mask_loss:
                    mlm_outside_weight[i] = 1.
        if self.__C.MASK_STRUCT['tsg']:
            mlm_outside_weight, mlm_pos = self.masking_struct_tsg_1d(mlm_outside_weight, mlm_pos, masked_tokenized_text, formatted_data, lemmas, do_mask_loss)

        # Do Mask
        ros_obj_id = torch.ones(grain_label.size(1), dtype=torch.float32)
        for i in range(len(masked_tokenized_text)):
            if mlm_pos[i] == 1.:
                masked_tokenized_text = self.masking_text_repalce(i, masked_tokenized_text)
                keep = torch.where(grain_label[i, :] > 0.9)[0]
                for k in keep.tolist():
                    if ros_obj_id[k] == 1.:
                        ros_obj_id[k] = min(grain_label[i, k], 0.98)
                    else:
                        ros_obj_id[k] = min(max(ros_obj_id[k], grain_label[i, k]), 0.98)
        ros_obj_id_t = copy.deepcopy(ros_obj_id)

        ious = torch.from_numpy(DataSet.cal_iou(boxes))
        for k in range(grain_label.size(1)):
            if ros_obj_id_t[k] != 1.:
                for ki in range(grain_label.size(1)):
                    if ki == k:
                        continue
                    if ious[k][ki] > 0.2:
                        prob = ros_obj_id_t[k] * (min((ious[k][ki] - 0.7), 0) * 0.3 + 1)
                        if ros_obj_id[ki] == 1.:
                            ros_obj_id[ki] = min(prob, 0.98)
                        else:
                            ros_obj_id[ki] = min(max(ros_obj_id[ki], prob), 0.98)

        # print('o', 1-ros_obj_id)
        return masked_tokenized_text, mlm_outside_weight, ros_obj_id


    def proc_text(self, text_input, text_outside_weight, text_mlm_label, grain_label):
        # concatenate lm labels and account for CLS, SEP
        text_input = ['[CLS]'] + text_input + ['[SEP]']
        text_mlm_label = ['[CLS]'] + text_mlm_label + ['[SEP]']

        text_input_ids = self.tokenizer.convert_tokens_to_ids(text_input)
        text_mlm_label_ids = self.tokenizer.convert_tokens_to_ids(text_mlm_label)

        # Mask & Segment Word
        text_outside_weight = [0.] + text_outside_weight + [0.]
        text_mask = [1] * len(text_input_ids)
        text_mlm_weight = [1] * len(text_input_ids)

        pad_length = self.__C.PAD_MAX['text'] - len(text_input_ids)
        if self.__C.PAD_INSIDE and len(text_input_ids) < self.__C.PAD_MAX['text']:
            text_input_ids += [0] * pad_length
            text_mlm_label_ids += [0] * pad_length
            text_outside_weight += [0.] * pad_length
            text_mask += [0] * pad_length
            text_mlm_weight += [0] * pad_length
        grain_label_padded = torch.zeros((self.__C.PAD_MAX['text'], grain_label.size(1)))
        grain_label_padded[1: -pad_length-1, :] = grain_label

        return text_input_ids, text_mask, text_outside_weight, text_mlm_label_ids, text_mlm_weight, grain_label_padded


    def rand_imgfeat_from_aggr(self, only_obj=True):
        if self.text_segment is not None:
            t_formatted_data = self.text_segment.load(random.randint(0, self.text_segment.total_len - 1))
        elif self.on_memory:
            t_formatted_data = self.data_aggr[random.randint(0, len(self.data_aggr)-1)]
        else:
            idx = random.randint(0, self.all_data_num - 1)
            for tsv_idx in range(len(self.data_start_idx)-1, -1, -1):
                if idx >= self.data_start_idx[tsv_idx]:
                    break
            tsv_data_idx = idx - self.data_start_idx[tsv_idx]
            row = self.data_tsv[tsv_idx].seek(tsv_data_idx)
            t_formatted_data = self.formatted_tsv_row(row)
        img_src = t_formatted_data['img_src']
        img_filename = t_formatted_data['img_file']
        if self.use_tsv_feat:
            tsv_file = self.tsv_files[img_src]
            img_offset_map = self.img_feat_offset_maps[img_src]
            img_idx = img_offset_map[img_filename]
            if img_src in self.datasets_with_splits:
                chunk_idx = img_idx // self.split_size
                img_idx = img_idx - chunk_idx * self.split_size
                row = tsv_file[chunk_idx].seek(img_idx)
            else:
                row = tsv_file.seek(img_idx)
            imgfeat = {}
            imgfeat['filename'] = row[0]
            num_bboxes = int(row[4])
            imgfeat['x'] = np.frombuffer(base64.b64decode(row[1]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['image_h'], imgfeat['image_w'] = int(row[2]), int(row[3])
            imgfeat['num_boxes'] = num_bboxes
            imgfeat['boxes'] = np.frombuffer(base64.b64decode(row[5]), dtype=np.float32).reshape((num_bboxes, -1))
            imgfeat['objects_id'] = np.frombuffer(base64.b64decode(row[6]), dtype=np.int64)
            imgfeat['objects_conf'] = np.frombuffer(base64.b64decode(row[7]), dtype=np.float32)
            imgfeat['attrs_id'] = np.frombuffer(base64.b64decode(row[8]), dtype=np.int64)
            imgfeat['attrs_conf'] = np.frombuffer(base64.b64decode(row[9]), dtype=np.float32)
        else:
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


    def masking_struct_bbox(self, ids, mlm_outside_weight, mlm_pos, ious, iou_thres, do_mask_loss=True):
        for i in range(mlm_outside_weight.shape[0]):
            if ids == i:
                continue

            if ious[ids][i] > iou_thres:
                if random.random() <= self.__C.MASK_STRUCT_PROB_INSIDE['bbox']:
                    mlm_pos[i] = 1.
                    if do_mask_loss:
                        mlm_outside_weight[i] = 1.

        return mlm_outside_weight, mlm_pos


    def masking_imgfeat(self, imgfeat_input, boxes, grain_label, formatted_data, tokenized_text, lemmas, do_mask_loss=True):
        masked_imgfeat_input = imgfeat_input.copy()
        mlm_outside_weight = np.zeros(imgfeat_input.shape[0], dtype=np.float32)
        mlm_pos = np.zeros(imgfeat_input.shape[0], dtype=np.float32)
        cent_pos = np.zeros(imgfeat_input.shape[0], dtype=np.float32)

        mask_grain_weight = copy.deepcopy(grain_label).transpose(0, 1)
        mask_grain_weight_sum = mask_grain_weight.sum(-1)
        mask_grain_weight_num = (mask_grain_weight > 0).float().sum(-1)
        mask_grain_weight_sum[mask_grain_weight_num > 0] = mask_grain_weight_sum[mask_grain_weight_num > 0] / mask_grain_weight_num[mask_grain_weight_num > 0]
        ious = torch.from_numpy(DataSet.cal_iou(boxes))

        # MRM
        for i in range(imgfeat_input.shape[0]):
            if random.random() < self.__C.MASK_PROB['image']:
                mlm_pos[i] = 1.
                cent_pos[i] = 1.
                # we will predict these later
                if do_mask_loss:
                    mlm_outside_weight[i] = 1.

                if self.__C.MASK_STRUCT['bbox'] and random.random() < self.__C.MASK_STRUCT_PROB['bbox']:
                    iou_thres = self.__C.OBJ_MASK_IOU_THRESH
                    mlm_outside_weight, mlm_pos = self.masking_struct_bbox(i, mlm_outside_weight, mlm_pos, ious, iou_thres, do_mask_loss=do_mask_loss)


            if random.random() < mask_grain_weight_sum[i] * self.__C.OBJ_GRAIN_RATIO:
                mlm_pos[i] = 1.
                cent_pos[i] = 1.
                # we will predict these later
                if do_mask_loss:
                    mlm_outside_weight[i] = 1.

                if self.__C.MASK_STRUCT['bbox'] and random.random() < self.__C.MASK_STRUCT_PROB['bbox']:
                    iou_thres = self.__C.OBJ_MASK_IOU_THRESH
                    mlm_outside_weight, mlm_pos = self.masking_struct_bbox(i, mlm_outside_weight, mlm_pos, ious, iou_thres, do_mask_loss=do_mask_loss)

        # Do Mask
        ros_word_id = torch.ones(grain_label.size(0), dtype=torch.float32)
        for i in range(imgfeat_input.shape[0]):
            if mlm_pos[i] == 1.:
                masked_imgfeat_input = self.masking_imgfeat_repalce(i, masked_imgfeat_input)

            if cent_pos[i] == 1.:
                keep = torch.where(grain_label[:, i] > 0.9)[0]
                for k in keep.tolist():
                    if ros_word_id[k] == 1.:
                        ros_word_id[k] = min(grain_label[k, i], 0.98)
                    else:
                        ros_word_id[k] = min(max(ros_word_id[k], grain_label[k, i]), 0.98)
        ros_word_id_t = copy.deepcopy(ros_word_id)

        bpe_ids = []
        for step, text in enumerate(tokenized_text):
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step+1)
        bpe_map = {}
        for step, bpe_id in enumerate(bpe_ids):
            for ids in bpe_id:
                bpe_map[ids] = step

        tokens = lemmas.strip().split()
        attr_map, attr_map_ids, rel_map, rel_map_ids = self.tsg_map_to_obj(formatted_data, tokens)
        for k in range(grain_label.size(0)):
            if ros_word_id_t[k] != 1.:
                cent = tokens[bpe_map[k]]
                if cent in attr_map:
                    for cent_i, cent_v in enumerate(attr_map[cent]):
                        prob = self.attr_count[cent][cent_v]
                        prob = ros_word_id_t[k] * (min(prob-0.5, 0.) * 0.3 + 1)
                        inds = attr_map_ids[cent][cent_i]
                        for ind in inds:
                            if ind < len(bpe_ids):
                                for ki in bpe_ids[ind]:
                                    if ros_word_id[ki] == 1.:
                                        ros_word_id[ki] = min(prob, 0.98)
                                    else:
                                        ros_word_id[ki] = min(max(ros_word_id[ki], prob), 0.98)

                if cent in rel_map:
                    for cent_i, cent_v in enumerate(rel_map[cent]):
                        prob = self.rel_count[cent][cent_v]
                        prob = ros_word_id_t[k] * (min(prob-0.2, 0.) * 0.75 + 1)
                        inds = rel_map_ids[cent][cent_i]
                        for ind in inds:
                            if ind < len(bpe_ids):
                                for ki in bpe_ids[ind]:
                                    if ros_word_id[ki] == 1.:
                                        ros_word_id[ki] = min(prob, 0.98)
                                    else:
                                        ros_word_id[ki] = min(max(ros_word_id[ki], prob), 0.98)

        return masked_imgfeat_input, mlm_outside_weight, ros_word_id


    def np_pad_1d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, (0, length - tensor.shape[0]), mode='constant', constant_values=value)


    def np_pad_2d(self, tensor, length, value=0):
        if tensor.shape[0] > length:
            tensor = tensor[:length]
        return np.pad(tensor, ((0, length - tensor.shape[0]), (0, 0)), mode='constant', constant_values=value)
        

    def proc_imgfeat(
            self, imgfeat_input, imgfeat_outside_weight, imgfeat_feat_label, imgfeat_feat_weight, imgfeat_obj_label,
            imgfeat_obj_weight, imgfeat_attr_label, imgfeat_attr_weight, imgfeat_bbox, length_pre):
        length_pad = self.__C.PAD_MAX['image'] + self.__C.PAD_MAX['text'] - length_pre

        imgfeat_mask = torch.ones(imgfeat_input.shape[0], dtype=torch.float32)
        imgfeat_mask = self.np_pad_1d(imgfeat_mask, length_pad)

        imgfeat_input = self.np_pad_2d(imgfeat_input, length_pad)
        imgfeat_feat_label = self.np_pad_2d(imgfeat_feat_label, length_pad)
        imgfeat_feat_weight = self.np_pad_1d(imgfeat_feat_weight, length_pad)
        imgfeat_outside_weight = self.np_pad_1d(imgfeat_outside_weight, length_pad)
        imgfeat_obj_label = self.np_pad_1d(imgfeat_obj_label, length_pad)
        imgfeat_obj_weight = self.np_pad_1d(imgfeat_obj_weight, length_pad)
        imgfeat_attr_label = self.np_pad_1d(imgfeat_attr_label, length_pad)
        imgfeat_attr_weight = self.np_pad_1d(imgfeat_attr_weight, length_pad)
        imgfeat_bbox = self.np_pad_2d(imgfeat_bbox, length_pad)


        return imgfeat_input, imgfeat_mask, imgfeat_outside_weight, imgfeat_feat_label, imgfeat_feat_weight, \
               imgfeat_obj_label, imgfeat_obj_weight, imgfeat_attr_label, imgfeat_attr_weight, imgfeat_bbox


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


    def sample_negative(self, formatted_data):
        is_repeat = True
        while is_repeat:
            if self.text_segment is not None:
                t_formatted_data = self.text_segment.load(random.randint(0, self.text_segment.total_len - 1))
            elif self.on_memory:
                t_formatted_data = self.data_aggr[random.randint(0, len(self.data_aggr) - 1)]
            else:
                idx = random.randint(0, self.all_data_num - 1)
                for tsv_idx in range(len(self.data_start_idx)-1, -1, -1):
                    if idx >= self.data_start_idx[tsv_idx]:
                        break
                tsv_data_idx = idx - self.data_start_idx[tsv_idx]
                row = self.data_tsv[tsv_idx].seek(tsv_data_idx)
                t_formatted_data = self.formatted_tsv_row(row)
            is_repeat = formatted_data['img_file'] == t_formatted_data['img_file'] and formatted_data['img_src'] == t_formatted_data['img_src']
            is_repeat = is_repeat or (formatted_data['img_src'] in ['coco'] and 'coco_id' in t_formatted_data and t_formatted_data['coco_id'] == formatted_data['img_id']) 
            is_repeat = is_repeat or (formatted_data['img_src'] in ['flickr'] and 'filckr_id' in t_formatted_data and t_formatted_data['filckr_id'] == formatted_data['img_id']) 
            is_repeat = is_repeat or ('coco_id' in formatted_data and t_formatted_data['img_src'] in ['coco'] and formatted_data['coco_id'] == t_formatted_data['img_id']) 
            is_repeat = is_repeat or ('flickr_id' in formatted_data and t_formatted_data['img_src'] in ['flickr'] and formatted_data['flickr_id'] == t_formatted_data['img_id']) 

        return t_formatted_data


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
        # box_listB_area = (box_listB[:, :, 2] - box_listB[:, :, 0] + 1.) * (box_listB[:, :, 3] - box_listB[:, :, 1] + 1.)
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
            # cos_dist = torch.exp(cos_dist)

        return cos_dist


    def grain_embedding(self, full_text, tokenized_text, formatted_data, img_objs_id, img_attrs_id):
        full_text = full_text.split()
        bpe_ids = []
        for step, text in enumerate(tokenized_text):
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)

        grain_inside_size = int(self.__C.GRAIN_EMB_SIZE / len(self.__C.GRAIN_EMB_TYPE))

        grain_embed = []
        if 'obj' in self.__C.GRAIN_EMB_TYPE:
            text_objs = []
            text_objs_id = []
            text_objs_glove = []
            for obj_id in formatted_data['tsg']['objs']:
                ids = obj_id[0][0]
                word = full_text[ids]
                word_glove = self.spacy_tool(word).vector

                if ids < len(bpe_ids) and np.sum(np.abs(word_glove)) != 0:
                    text_objs_id.append(ids)
                    text_objs.append(word)
                    text_objs_glove.append(word_glove)
            text_objs_glove = torch.tensor(text_objs_glove)

            img_objs = [self.obj_id_map[i] for i in img_objs_id]
            img_objs_glove = torch.tensor([self.spacy_tool(i).vector for i in img_objs])

            obj_grain = torch.ones((len(tokenized_text), len(img_objs_id), grain_inside_size))
            if text_objs_glove.size(0) > 0:
                cos_dist = DataSet.cal_dist_batch(text_objs_glove, img_objs_glove, type=self.__C.GRAIN_EMB)
                for step, ids in enumerate(text_objs_id):
                    for bpe_id in bpe_ids[ids]:
                        obj_grain[bpe_id] = cos_dist[step, :]

            grain_embed.append(obj_grain)

        if 'attr' in self.__C.GRAIN_EMB_TYPE:
            text_attrs = []
            text_attrs_id = []
            text_attrs_glove = []
            for attr_id in formatted_data['tsg']['attrs']:
                ids = attr_id[1][0]
                word = full_text[ids]
                word_glove = self.spacy_tool(word).vector

                if ids < len(bpe_ids) and np.sum(np.abs(word_glove)) != 0:
                    text_attrs_id.append(ids)
                    text_attrs.append(word)
                    text_attrs_glove.append(word_glove)
            text_attrs_glove = torch.tensor(text_attrs_glove)
            
            img_attrs = [self.attr_id_map[i] for i in img_attrs_id]
            img_attrs_glove = torch.tensor([self.spacy_tool(i).vector for i in img_attrs])

            attr_grain = torch.ones((len(tokenized_text), len(img_attrs_id), grain_inside_size))
            if text_attrs_glove.size(0) > 0:
                cos_dist = DataSet.cal_dist_batch(text_attrs_glove, img_attrs_glove, type=self.__C.GRAIN_EMB)
                for step, ids in enumerate(text_attrs_id):
                    for bpe_id in bpe_ids[ids]:
                        attr_grain[bpe_id] = cos_dist[step, :]

            grain_embed.append(attr_grain)

        grain_embed = torch.cat(grain_embed, dim=-1)
        assert grain_embed.size(2) == self.__C.GRAIN_EMB_SIZE

        return grain_embed


    def proc_grain(self, full_text, tokenized_text, formatted_data, img_objs_id, img_attrs_id):
        full_text = full_text.split()
        bpe_ids = []
        for step, text in enumerate(tokenized_text):
            if not text.startswith('##'):
                bpe_ids.append([])
            bpe_ids[-1].append(step)

        text_objs = []
        text_objs_id = []
        text_objs_glove = []
        for obj_id in formatted_data['tsg']['objs']:
            ids = obj_id[0][0]
            word = full_text[ids]
            word_glove = self.spacy_tool(word).vector

            if ids < len(bpe_ids) and np.sum(np.abs(word_glove)) != 0:
                text_objs_id.append(ids)
                text_objs.append(word)
                text_objs_glove.append(word_glove)
        text_objs_glove = torch.tensor(text_objs_glove)

        img_objs = [self.obj_id_map[i] for i in img_objs_id]
        img_objs_glove = torch.tensor([self.spacy_tool(i).vector for i in img_objs])
        
        obj_grain = torch.zeros((len(tokenized_text), len(img_objs_id)))
        if text_objs_glove.size(0) > 0:
            cos_dist = DataSet.cal_dist_batch(text_objs_glove, img_objs_glove, type='scalar').squeeze(-1)
            for step, ids in enumerate(text_objs_id):
                for bpe_id in bpe_ids[ids]:
                    obj_grain[bpe_id] = 1 - cos_dist[step, :]

        obj_grain[obj_grain < self.__C.OBJ_GRAIN_THRESH] = 0

        return obj_grain

