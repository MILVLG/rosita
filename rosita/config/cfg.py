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
        
        # Version Control
        self.VERSION = 'version_name'

        # Workers and batch size
        self.NUM_WORKERS = 4
        self.NUM_WORKERS_NEG = 4
        self.BATCH_SIZE = 32
        self.EVAL_BATCH_SIZE = 32
        
        # Load CKPT
        self.CKPT_FILE = 'ckpt_file_name'
        self.RESUME_FROM_CKPT_FILE = False
        self.CKPT_EPOCH = 0
        self.CKPT_LOAD = True
        self.CKPT_LOAD_MAP = {
            'text_embeddings': 'text_embeddings',
            'visual_embeddings': 'visual_embeddings',
            'backbone': 'backbone',
            'pooler': 'pooler',
        }

        self.CKPT_SAVE_MAP = {
            'epoch': 'epoch',
            'net_optim': 'net_optim',
            'text_embeddings': 'text_embeddings',
            'visual_embeddings': 'visual_embeddings',
            'backbone': 'backbone',
            'pooler': 'pooler',
        }

        # Datasets
        self.DATASET_LIST = {
            'train': [],
            'val': [],
            'test': [],
        }
        
        # Features
        self.IMGFEAT_SIZE = 2048
        self.IMGFEAT_OBJ_CLASSNUM = 1600
        self.IMGFEAT_ATTR_CLASSNUM = 400
        self.USE_BBOXFEAT = True
        self.BBOXFEAT_SIZE = 5

        imgfeat_type = 'bua_r101_fix36'
        self.IMGFEAT_FORMAT = 'tsv'
        self.IMGFEAT_LIST = [
            'coco:' + imgfeat_type,
            'flickr:' + imgfeat_type,
        ]

        # VQA settings
        self.QA_CLS_WEIGHT_MACTH = False
        self.PUNCT_ANS_MAP = {
            'vqa': True,
        }

        # ITR Params
        self.NEG_BATCHSIZE = 25
        self.NEG_RANDSIZE = 128
        self.NEG_HARDSIZE = 5
        self.NEG_NEPOCH = 1
        self.NEG_START_EPOCH = 0

        # REC Params
        self.BBOX_NORM = True
        self.BBOX_NORM_MEANS = (0.0, 0.0, 0.0, 0.0)
        self.BBOX_NORM_STDS = (0.1, 0.1, 0.2, 0.2)
        self.OVERLAP_THRESHOLD = 0.5
        
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

        # Model Params
        self.PAD_INSIDE = True
        self.PAD_MAX = {'text': 30, 'image': 36, }
        self.TASKS = {'mm': ['qa', 'refs', 'itm-tri']}

        # Loss Params
        self.LOSS_REDUCTION = {'mm': {'qa': 'mean', 'itm-tri': 'mean', 'refs-rank': 'mean', 'refs-reg': 'mean'}}
        self.INSIDE_WEIGHTING = {'mm': ['itm-tri', 'refs-rank', 'refs-reg']}
        self.MULTINOMIAL_QA_LABEL = False
        self.LOSSFUNC_MAPPING = {
            'mm': {'qa': nn.KLDivLoss, 'itm-tri': nn.CrossEntropyLoss, 'refs-rank': nn.KLDivLoss, 'refs-reg': nn.SmoothL1Loss}
        }
        self.LOSSFUNC_WEIGHT = {'mm': {'qa': 1., 'itm-tri': 1., 'refs-rank': 1., 'refs-reg': 0.5}}

        # Optimizer Params
        self.NET_OPTIM = 'warmup_adam'
        if self.NET_OPTIM in ['warmup_adam']:
            # Optimizer Warmup Adam
            self.NET_OPTIM_WARMUP = True
            self.NET_LR_BASE = 0.00004
            self.NET_WEIGHT_DECAY = 0
            self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.NET_LR_DECAY_R = 0.1
            self.NET_LR_DECAY_LIST = [2, 4]
            self.OPT_BETAS = (0.9, 0.98)
            self.OPT_EPS = 1e-9
            self.WARMUP_EPOCHS = 1

        elif self.NET_OPTIM in ['bert_adam']:
            # Optimizer BERT Adam
            self.NET_LR_BASE = 0.0001
            self.NET_WEIGHT_DECAY = 0.01
            self.NET_GRAD_CLIP = 1.  # GRAD_CLIP = -1: means not use grad_norm_clip
            self.NET_LR_DECAY_R = 0.1
            self.NET_LR_DECAY_LIST = []
            self.OPTIM_EPOCHS = 40
            self.WARMUP_EPOCHS = 2
            
        self.MAX_EPOCH = 15
        self.LOSS_TRANSFER = (3, nn.BCEWithLogitsLoss)

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
            self.CKPT_LOAD_MAP['epoch'] = 'epoch'
            self.CKPT_LOAD_MAP['net_optim'] = 'net_optim'
        else:
            self.CKPT_LOAD_MAP.pop('epoch', '')
            self.CKPT_LOAD_MAP.pop('net_optim', '')


    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''