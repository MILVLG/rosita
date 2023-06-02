# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Yuhao Cui and Tong-An Luo
 # -------------------------------------------------------- 

import torch, logging, os
import torch.nn as nn
from .path_cfg import Path
from types import MethodType
from typing import Dict, Any


class Cfg(Path):
    def __init__(self, world_size, args):
        super(Cfg, self).__init__()
        self.RUN_MODE = 'train'
        logging.info('Cudnn Version: {}'.format(torch.backends.cudnn.version()))

        # Set Devices
        self.WORLD_SIZE = world_size
        self.NODE_SIZE = args.NODE_SIZE
        self.N_GPU = torch.cuda.device_count() // self.WORLD_SIZE
        self.MP_STORAGE_SHR = {
            'ckpt': True,
            'screen': False,
            'tmp': False,
            'eval': True,
        }

        # Set Seed For CPU And GPUs
        self.SEED = 888

        # APEX Setting
        self.APEX_LEVEL = 'O1'
        self.BN_SYNC = False
        self.BN_FP32 = None
        
        # Version Control
        self.VERSION = 'version_name'

        # Workers and batch size
        self.NUM_WORKERS = 8
        self.BATCH_SIZE = 128
        self.EVAL_BATCH_SIZE = 128
        
        # Load CKPT
        self.CKPT_FILE = 'ckpt_file_name'
        self.RESUME_FROM_CKPT_FILE = False
        self.CKPT_EPOCH = 0
        self.CKPT_LOAD = True
        self.CKPT_LOAD_MAP = {
            'text_embeddings': 'text_embeddings',
            'backbone': 'backbone',
            'pooler': 'pooler',
            'text_mlm_head': 'text_mlm_head',
            'mm_itm_head': 'mm_itm_head',
        }

        self.CKPT_SAVE_MAP = {
            'epoch': 'epoch',
            'net_optim': 'net_optim',
            'amp': 'amp',
            'text_embeddings': 'text_embeddings',
            'visual_embeddings': 'visual_embeddings',
            'backbone': 'backbone',
            'pooler': 'pooler',
            'text_mlm_head': 'text_mlm_head',
            'imgfeat_head': 'imgfeat_head',
            'mm_itm_head': 'mm_itm_head',
            # 'mm_qa_head': 'mm_qa_head',
        }

        # Datasets
        self.DATASET_LIST = {
            'train': [
                # 'pt-coco:train',
                # 'pt-conceptual:train',
                # 'pt-sbu:train',
                # 'pt-vg:train',
                # 'genome_cap:train',
            ],
            'val': [
                'vqa-vqav2:minival',
            ],
            'test': [
                'vqa-vqav2:test',
            ],
        }

        self.ANNO_FORMAT = 'json'
        self.TSV_ON_MEMORY = False
        
        # Features
        self.USE_USG = False
        self.USG_NOISE = False
        self.USE_RELFEAT = True
        self.REL_SIZE = 64
        self.EMB_SIZE = 768
        self.IMGFEAT_SIZE = 2048
        self.IMGFEAT_OBJ_CLASSNUM = 1600
        self.IMGFEAT_ATTR_CLASSNUM = 400
        self.USE_BBOXFEAT = True
        self.BBOXFEAT_SIZE = 5

        imgfeat_type = 'bua_r101_fix36'
        self.IMGFEAT_FORMAT = 'npz'
        self.IMGFEAT_LIST = [
            'conceptual:' + imgfeat_type,
            'sbu:' + imgfeat_type,
            'coco:' + imgfeat_type,
            'genome:' + imgfeat_type,
        ]

        # VQA settings
        self.QA_CLS_WEIGHT_MACTH = False
        self.PUNCT_ANS_MAP = {
            'vqa': True,
            'genome': True,
        }

        
        
        # Network Params
        self.POS_EMB_IN_SIZE = 512
        self.TYPE_EMB_IN_SIZE = 2
        self.LAYER = 12
        self.HSIZE = 768
        self.HHEAD = 12
        self.HBASE = int(self.HSIZE / self.HHEAD)
        self.HFF = int(self.HSIZE * 4)
        self.DROPOUT_R = 0.1
        self.WEIGHT_INIT_FACTOR = 0.02


        # Training Params
        self.MASK_SIDE_PROB = 0.5  # -1 means masking both sides simultaneously, the other means the probability of text side masking
        self.MASK_PROB = {'text': 0.15, 'image': 0.15}
        self.MASK_PROB_POST = {'mask': 0.8, 'replace': 0.1}

        self.MASK_STRUCT = {'tsg': True, 'tdt': False, 'isg': False, 'bbox': True}
        self.MASK_STRUCT_PROB = {'tsg': 0.3, 'tdt': 0.3, 'isg': 0.3, 'bbox': 0.3}
        self.MASK_STRUCT_PROB_INSIDE = {'tsg': 1.0, 'tdt': 1.0, 'isg': 1.0, 'bbox': 1.0}
        self.MASK_STRUCT_DIST = {
            'tsg': [],
            'tdt': [],
            'isg': [],
            # 'bbox': [0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.35, 0.3],
            'bbox': [0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.25,
                     0.25],
        }
        self.MASK_STRUCT_PRESERVE_OBJ = False
        self.OBJ_MASK_ATTMAP_IOU_THRESH = 0.1
        self.OBJ_MASK_ATTMAP_IOU_PROB = 0.
        self.OBJ_MASK_IOU_THRESH = 0.2
        self.OBJ_GRAIN_THRESH = 0.5
        self.OBJ_GRAIN_RATIO = 0.9

        # Model Params
        self.PAD_INSIDE = True
        # self.PAD_MAX = {'text': 20, 'image': 36, }
        self.PAD_MAX = {'text': 20, 'image': 36, }
        self.MASK_IMGFEAT_WITH = 'zero' # zero, gaussian, uniform
        self.TASKS = {'text': ['mlm'], 'image': ['feat', 'obj', 'attr'], 'mm': ['itm']}

        # Loss Params
        self.OT_LAMBDA = 0.1
        self.LOSS_REDUCTION = {'text': {'mlm': 'mean'}, 'image': {'feat': 'mean', 'obj': 'mean', 'attr': 'mean'},
                               'mm': {'qa': 'mean', 'itm': 'mean'}}
        self.INSIDE_WEIGHTING = {'text': ['mlm'], 'image': ['feat', 'obj', 'attr'], 'mm': []}
        self.OUTSIDE_WEIGHTING = {'text': ['mlm'], 'image': ['feat', 'obj', 'attr'], 'mm': []}

        # self.MATCH_CONSTRAIN = ['qa']
        self.MATCH_CONSTRAIN = ['qa', 'text', 'image']

        # self.MATCH_NEG_SHUFFLE = 'text'
        self.MATCH_NEG_SHUFFLE = 'image'

        self.MULTINOMIAL_QA_LABEL = False
        self.LOSSFUNC_MAPPING = {
            'text': {'mlm': nn.CrossEntropyLoss},
            'image': {'feat': nn.SmoothL1Loss, 'obj': nn.CrossEntropyLoss, 'attr': nn.CrossEntropyLoss},
            'mm': {'qa': nn.KLDivLoss, 'itm': nn.CrossEntropyLoss}
        }
        self.LOSSFUNC_WEIGHT = {'text': {'mlm': 1.}, 'image': {'feat': 1., 'obj': 1., 'attr': 1.}, 'mm': {'qa': 1., 'itm': 1.}}

        # Optimizer Params
        self.NET_OPTIM = 'bert_adam'

        # Optimizer BERT Adam
        self.NET_LR_BASE = 0.0001
        self.NET_WEIGHT_DECAY = 0.01
        self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
        self.NET_LR_DECAY_R = 0.1
        self.NET_LR_DECAY_LIST = []
        self.OPTIM_EPOCHS = 50
        self.WARMUP_EPOCHS = 2
        # for Warmup Adam
        self.OPT_BETAS = (0.9, 0.98)
        self.OPT_EPS = 1e-9
            
        self.MAX_EPOCH = 40
        self.LOSS_TRANSFER = (-1, nn.BCEWithLogitsLoss)

        # Evaluate
        self.EVAL_EVERY_EPOCH = True
        logging.info('Eval after every epoch: {}'.format(self.EVAL_EVERY_EPOCH))

        # Memory Efficency
        self.SEGMENT_TEXT = False
        self.RE_SEGMENT = False # should be True if SEGMENT_TEXT and while runing on a dataset for the first time. 


    def set_rank(self, global_rank, local_rank):
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s][rank-{}] %(message)s".format(global_rank), datefmt = '%Y-%m-%d  %H:%M:%S %a')
        self.GRANK = global_rank
        self.LRANK = local_rank
        self.DEVICE_IDS = list(range(self.LRANK * self.N_GPU, (self.LRANK + 1) * self.N_GPU))
        print('DEVICE IDS:', self.DEVICE_IDS)
        torch.cuda.set_device(self.LRANK)


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict

    def add_args(self, args_dict):

        def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]):
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v
            return b

        for arg in args_dict:
            assert hasattr(self, arg), '\'Cfg\' object has no attribute \'{}\''.format(arg)
            if isinstance(args_dict[arg], dict):
                assert isinstance(
                    getattr(self, arg), dict
                ), "Cannot inherit key '{}' from base!".format(arg)
                merged = merge_a_into_b(args_dict[arg], getattr(self, arg))
                setattr(self, arg, merged)
            else:
                setattr(self, arg, args_dict[arg])
    
    def proc(self, resume):
        assert self.RUN_MODE in ['train', 'val', 'test']

        self.proc_base_path(self.VERSION)
        self.proc_imgfeat_path(self.IMGFEAT_LIST)
        if 'mm' in self.TASKS and 'qa' in self.TASKS['mm']:
            self.proc_qa_path()

        self.BATCH_ALL_GPUS = self.BATCH_SIZE * self.WORLD_SIZE

        if resume:
            self.CKPT_FILE = os.path.join(self.CKPT_SAVE_PATH, 'last_ckpt.pkl')
            self.RESUME_FROM_CKPT_FILE = True
            logging.info('Resume from last epoch')

        if self.RESUME_FROM_CKPT_FILE:
            # self.CKPT_LOAD_MAP['epoch'] = 'epoch'
            # self.CKPT_LOAD_MAP['net_optim'] = 'net_optim'
            self.CKPT_LOAD_MAP = self.CKPT_SAVE_MAP
        else:
            self.CKPT_LOAD_MAP.pop('epoch', '')
            self.CKPT_LOAD_MAP.pop('net_optim', '')


    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''