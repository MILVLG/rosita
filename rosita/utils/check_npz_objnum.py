from glob import glob
from os import path as osp
import numpy as np
from tqdm import tqdm

imgfeat_root = 'datasets/imgfeats'
datasets = ['conceptual', 'mscoco', 'visualgenome', 'sbu']
dataset = datasets[0]

imgfeats = glob(osp.join(imgfeat_root, '{}_bua_r101_fix36/npz_files/*.npz'.format(dataset)))
# print(osp.join(imgfeat_root, '{}_bua_r101_fix36/npz_files/*.npz'.format(dataset)))
print(len(imgfeats))

for imgfeat in tqdm(imgfeats):
    npz = np.load(imgfeat)
    objs = npz['objects_id']
    if len(objs) != 36:
        print(imgfeat)
        print(len(objs))