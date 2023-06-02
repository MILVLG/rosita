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
import ray
from ray.actor import ActorHandle
from tsv_file import tsv_writer
from progress_bar import ProgressBar


@ray.remote(num_cpus=1)
def transfer_npz2row(npz_path, ray_split_num, ray_split_idx, split_size, split_idx, npz_list, actor: ActorHandle):
    num_npzs = len(npz_list)
    print('Number of npzs on ray split{}: {}.'.format(ray_split_idx, num_npzs))

    rows = []
    img_feat_offset_map = {}

    for i, npz_file in enumerate(npz_list):
        idx = i * ray_split_num + ray_split_idx + split_idx * split_size
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

        actor.update.remote(1)
    
    return rows, img_feat_offset_map


def transfer_npz2tsv(npz_path, tsv_path, ray_split_num, split_size):
    # To transfer npz files to a tsv file

    npz_list = os.listdir(npz_path)
    print('Starting transfer {} npz files'.format(len(npz_list)))

    ray.init()
    split_num = len(npz_list) // split_size
    if len(npz_list) % split_size:
        split_num += 1
    print('Number of splits:', split_num)

    img_feat_offset_map = {}
    for split_idx in range(split_num):
        npz_split_list = npz_list[split_idx*split_size:(split_idx+1)*split_size]
        npz_lists = [npz_split_list[i::ray_split_num] for i in range(ray_split_num)]
        split_npzs = len(npz_split_list)
        print('Starting transfer split {}.'.format(split_idx))
        print('Number of npzs in this split:', split_npzs)
        pb = ProgressBar(split_npzs)
        actor = pb.actor

        print('Number of ray splits: {}'.format(ray_split_num))
        transfer_row_list = []
        for i in range(ray_split_num):
            transfer_row_list.append(transfer_npz2row.remote(npz_path, ray_split_num, i, split_size, split_idx, npz_lists[i], actor))
        
        pb.print_until_done()
        rows_and_maps = ray.get(transfer_row_list)
        ray.get(actor.get_counter.remote())

        rows = []
        for i in range(len(npz_list)):
            part_idx = i % ray_split_num
            rows.append(rows_and_maps[part_idx][0][0])
            del rows_and_maps[part_idx][0][0]

        for _ in range(ray_split_num):
            img_feat_offset_map.update(rows_and_maps[0][1])
            del rows_and_maps[0]

        tsv_file = os.path.join(tsv_path, 'imgfeat_split{}.tsv'.format(split_idx))
        tsv_writer(rows, tsv_file)
    # end for

    img_feat_offset_map_file = os.path.join(tsv_path, 'img_feat_offset_map.json')
    with open(img_feat_offset_map_file, 'w') as f:
        json.dump(img_feat_offset_map, f)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--npz-dir', dest='path_to_npz_files', type=str)
    parser.add_argument('--tsv-dir', dest='path_to_tsv_files', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args = parse_args()
    # transfer_npz2tsv(args.path_to_npz_files, args.path_to_tsv_files)
    path_to_npz_files = '/data/luota/rosita_datasets/imgfeats/sbu_bua_r101_fix36/npz_files'
    path_to_tsv_files = '/data/luota/rosita_datasets/imgfeats/sbu_bua_r101_fix36'
    ray_split_num = 16
    split_size = 450000
    transfer_npz2tsv(path_to_npz_files, path_to_tsv_files, ray_split_num, split_size)