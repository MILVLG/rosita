# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import os
import logging

class Path:
    def __init__(self):
        
        self.IMGFEAT_TYPE_MAP = {}
        self.DATASET_ROOTPATH = 'datasets'
        self.OUTPATH = 'outputs'

        # Path must in multi-node shared storage
        self.TMP_RESULT_PATH = None

        self.TEXT_SEGMENT_PATH = None

        self.DATASET_PATHMAP = {
            'vqa-vqav2': 'annotations/vqa-vqav2',
            'itr-flickr': 'annotations/itr-flickr',
            'itr-coco': 'annotations/itr-coco',
            'rec-refcoco': 'annotations/rec-refcoco',
            'rec-refcocoplus': 'annotations/rec-refcocoplus',
            'rec-refcocog': 'annotations/rec-refcocog',
        }

        self.DATASET_FEATMAP = {
            'vqa-vqav2': 'coco',
            'itr-flickr': 'flickr',
            'itr-coco': 'coco',
            'rec-refcoco': 'coco',
            'rec-refcocoplus': 'coco',
            'rec-refcocog': 'coco',
        }

        self.IMGFEAT_PATHMAP = {
            'flickr': 'imgfeats/flickr_bua_r101_fix36',
            'coco': 'imgfeats/mscoco_bua_r101_fix36',
        }
    

    def proc_base_path(self, version_name):
        
        self.OUTPATH = os.path.join(self.OUTPATH, version_name)
        os.system('mkdir -p ' + self.OUTPATH)
        logging.info('DATASET_ROOTPATH: {}'.format(self.DATASET_ROOTPATH))
        logging.info('OUTPATH: {}'.format(self.OUTPATH))

        dataset_rootpath = self.DATASET_ROOTPATH
        for dataset_name in self.DATASET_PATHMAP:
            self.DATASET_PATHMAP[dataset_name] = os.path.join(dataset_rootpath, self.DATASET_PATHMAP[dataset_name])
        for featset_name in self.IMGFEAT_PATHMAP:
            self.IMGFEAT_PATHMAP[featset_name] = os.path.join(dataset_rootpath, self.IMGFEAT_PATHMAP[featset_name])

        self.DATASET_ANNO_MAP = {
            'vqa-vqav2': os.path.join(self.DATASET_PATHMAP['vqa-vqav2'], 'vqa_vqav2_annotations.json'),
            'itr-flickr': os.path.join(self.DATASET_PATHMAP['itr-flickr'], 'itr_flickr_annotations.json'),
            'itr-coco': os.path.join(self.DATASET_PATHMAP['itr-coco'], 'itr_coco_annotations.json'),
            'rec-refcoco': os.path.join(self.DATASET_PATHMAP['rec-refcoco'], 'rec_refcoco_annotations.json'),
            'rec-refcocoplus': os.path.join(self.DATASET_PATHMAP['rec-refcocoplus'], 'rec_refcocoplus_annotations.json'),
            'rec-refcocog': os.path.join(self.DATASET_PATHMAP['rec-refcocog'], 'rec_refcocog_annotations.json'),
        }

        self.LOG_PATH = os.path.join(self.OUTPATH, 'logs')
        os.system('mkdir -p ' + self.LOG_PATH)
        
        if self.RUN_MODE in ['train']:
            self.CKPT_SAVE_PATH = os.path.join(self.OUTPATH, 'ckpts')
            os.system('mkdir -p ' + self.CKPT_SAVE_PATH)

        self.BERT_VOCAB_PATH = 'rosita/utils/bert_vocabs/vocab.txt'

        if self.TEXT_SEGMENT_PATH is not None:
            self.SEGMENT_PATH = {
                'files': self.TEXT_SEGMENT_PATH,
                'sync': os.path.join(self.TEXT_SEGMENT_PATH, 'sync.txt'),
            }
            os.system('mkdir -p ' + self.SEGMENT_PATH['files'])


    def proc_imgfeat_path(self, imgfeat_list):
                
        for imgfeat_name in imgfeat_list:
            iset = imgfeat_name.split(':')[0]
            itype = imgfeat_name.split(':')[1]
            self.IMGFEAT_TYPE_MAP[iset] = itype
        pass
    
    def proc_qa_path(self):
        self.TEST_RESULT_PATH = os.path.join(self.OUTPATH, 'result_test')
        os.system('mkdir -p ' + self.TEST_RESULT_PATH)
        self.VAL_RESULT_PATH = os.path.join(self.OUTPATH, 'result_val')
        os.system('mkdir -p ' + self.VAL_RESULT_PATH)
        if self.TMP_RESULT_PATH is None:
            self.TMP_RESULT_PATH = os.path.join(self.OUTPATH, 'result_tmp')
        os.system('mkdir -p ' + self.TMP_RESULT_PATH) 

