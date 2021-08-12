import os
import logging

class Path:
    def __init__(self):
        
        self.IMGFEAT_ROOTPATH_MAP = {}
        self.IMGFEAT_TYPE_MAP = {}
        self.DATASET_ROOTPATH = None
        self.OUTPATH = None

        # Path must in multi-node shared storage
        self.TMP_RESULT_PATH = None

        self.TEXT_SEGMENT_PATH = None

        self.DATASET_PATHMAP = {
            'vqa': 'VQA',
            'flickr': 'flickr',
            'coco': 'mscoco',
            'refcoco': 'RefCOCO',
            'refcoco+': 'RefCOCO',
            'refcocog': 'RefCOCO',
        }
        self.IMGFEAT_PATHMAP = {
            'coco': 'mscoco',
            'flickr': 'flickr',
        }
        self.IMGFEAT_TYPEMAP = {
            'butd_res101_36-36_i32w_pyt': 'butd_res101_36-36_i32w_pyt',
        }
    

    def proc_base_path(self, version_name):
        
        self.OUTPATH = os.path.join(self.OUTPATH, version_name)
        os.system('mkdir -p ' + self.OUTPATH)
        logging.info('DATASET_ROOTPATH: {}'.format(self.DATASET_ROOTPATH))
        logging.info('OUTPATH: {}'.format(self.OUTPATH))

        dataset_rootpath = self.DATASET_ROOTPATH
        postfix = '_tsg'
        self.DATASET_ROOTPATH_MAP = {
            'vqa': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['vqa'] + '/formatted_data{}.json'.format(postfix))),
            'flickr': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['flickr'] + '/formatted_data{}.json'.format(postfix))),
            'flickr_feat_neg_ids': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['flickr'] + '/feat_neg_ids_map.json')),
            'coco': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['coco'] + '/formatted_data{}.json'.format(postfix))),
            'coco_feat_neg_ids': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['coco'] + '/feat_neg_ids_map.json')),
            'refcoco': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['refcoco'] + '/formatted_data_refcoco.json'.format(postfix))),
            'refcoco+': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['refcoco+'] + '/formatted_data_refcoco+.json'.format(postfix))),
            'refcocog': os.path.join(dataset_rootpath, (self.DATASET_PATHMAP['refcocog'] + '/formatted_data_refcocog.json'.format(postfix))),
        }

        self.LOG_PATH = os.path.join(self.OUTPATH, 'logs')
        os.system('mkdir -p ' + self.LOG_PATH)
        
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

            imgfeat_rootpath = os.path.join(self.DATASET_ROOTPATH, self.IMGFEAT_PATHMAP[iset], 'features', self.IMGFEAT_TYPEMAP[itype])

            self.IMGFEAT_ROOTPATH_MAP[iset] = imgfeat_rootpath
    
    def proc_qa_path(self):
        self.TEST_RESULT_PATH = os.path.join(self.OUTPATH, 'result_test')
        os.system('mkdir -p ' + self.TEST_RESULT_PATH)
        self.VAL_RESULT_PATH = os.path.join(self.OUTPATH, 'result_val')
        os.system('mkdir -p ' + self.VAL_RESULT_PATH)
        if self.TMP_RESULT_PATH is None:
            self.TMP_RESULT_PATH = os.path.join(self.OUTPATH, 'result_tmp')
        os.system('mkdir -p ' + self.TMP_RESULT_PATH) 

