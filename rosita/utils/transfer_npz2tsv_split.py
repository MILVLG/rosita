# -------------------------------------------------------- 
# ROSITA
# Licensed under The Apache License 2.0 [see LICENSE for details] 
# Written by Tong-An Luo
# -------------------------------------------------------- 

import os
import os.path as op
import json
import numpy as np
import base64
from tqdm import tqdm
import argparse

from tsv_file import tsv_writer

def transfer_npz2tsv(npz_path, tsv_path, split_size):
    # To transfer npz files to a tsv file:
    img_feat_offset_map_file = os.path.join(tsv_path, 'img_feat_offset_map.json')

    npz_list = os.listdir(npz_path)
    print('Starting transfer {} npz files'.format(len(npz_list)))
    
    idx = 0
    img_feat_offset_map = {}
    split_num = len(npz_list) // split_size
    if len(npz_list) % split_size > 0:
        split_num += 1

    print('Total {} part.'.format(split_num))
    
    for i in range(split_num):
        print('Start transfer part', i)
        rows = []
        tsv_file = os.path.join(tsv_path, 'imgfeat_split{}.tsv'.format(i))
        npz_split_list = npz_list[i*split_size:(i+1)*split_size]

        for npz_file in tqdm(npz_split_list):
            npz = np.load(op.join(npz_path, npz_file))
            filename = str(npz['filename'])
            img_feat = npz['x']
            img_h = int(npz['image_h'])
            img_w = int(npz['image_w'])
            num_boxes = int(npz['num_boxes'])
            boxes = npz['boxes']
            objects_id = npz['objects_id']
            objects_conf = npz['objects_conf']
            attrs_id = npz['attrs_id']
            attrs_conf = npz['attrs_conf']

            img_feat_offset_map[filename] = idx
            img_feat_encoded = base64.b64encode(img_feat)
            boxes_encoded = base64.b64encode(boxes)
            objects_id_encoded = base64.b64encode(objects_id)
            objects_conf_encoded = base64.b64encode(objects_conf)
            attrs_id_encoded = base64.b64encode(attrs_id)
            attrs_conf_encoded = base64.b64encode(attrs_conf)

            row = [filename, img_feat_encoded, img_h, img_w, num_boxes, boxes_encoded, 
                objects_id_encoded, objects_conf_encoded, attrs_id_encoded, attrs_conf_encoded]
            rows.append(row)

            idx += 1

        tsv_writer(rows, tsv_file)
    with open(img_feat_offset_map_file, 'w') as f:
        json.dump(img_feat_offset_map, f)

# def parse_args():
#     parser = argparse.ArgumentParser(description='Multi-Node Args')
#     parser.add_argument('--npz-dir', dest='path_to_npz_files', type=str)
#     parser.add_argument('--tsv-dir', dest='path_to_tsv_files', type=str)
#     args = parser.parse_args()
#     return args


if __name__ == '__main__':
    # args = parse_args()
    # transfer_npz2tsv(args.path_to_npz_files, args.path_to_tsv_files)
    # path_to_npz_files = '/data-ssd/luota/rosita_datasets/imgfeats/sbu_bua_r101_fix36/npz_files'
    # path_to_tsv_files = '/data/luota/features/sbu/tsv'
    path_to_npz_files = '/data/luota/rosita_datasets/imgfeats/conceptual_bua_r101_fix36/npz_files'
    path_to_tsv_files = '/data/luota/rosita_datasets/imgfeats/conceptual_bua_r101_fix36'
    split_size = 800000
    transfer_npz2tsv(path_to_npz_files, path_to_tsv_files, split_size)
