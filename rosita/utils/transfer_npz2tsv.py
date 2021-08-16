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

from tsv_file import tsv_writer

# To transfer npz files to a tsv file:
npz_path = "../../datasets/mscoco/features/butd_res101_36-36_i32w_pyt"
tsv_file = "../../datasets/mscoco/features/butd_res101_36-36_i32w_pyt_tsv/imgfeat.tsv"
img_feat_offset_map_file = "../../datasets/mscoco/features/butd_res101_36-36_i32w_pyt_tsv/img_feat_offset_map.json"

npz_list = os.listdir(npz_path)

rows = []
idx = 0
img_feat_offset_map = {}

for npz_file in tqdm(npz_list):
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
