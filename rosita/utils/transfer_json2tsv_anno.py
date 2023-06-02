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

def transfer(anno_file, tsv_file):
    '''
    column0     column1     column2     column3     column4     column5     column6
    type        text_set    text_split  text_id     text        lemmas      img_src
    column7     column8     column9     column10    column11    column12    column13    column14
    img_id      img_file    split_info  label       multi_label gt_boxes    image_h     image_w
    '''
    # RefCOCO / plus / g
    # column_key_name = {
    #     0: 'type', 1: 'text_set', 2: 'text_split', 3: 'text_id', 4: 'text', 5: 'lemmas', 6: 'img_src', 
    #     7: 'img_id', 8: 'img_file', 9: 'split_info', 10: 'label', 11: 'multi_label', 12: 'gt_boxes', 
    #     13: 'image_h', 14: 'image_w'
    # }
    # VQAv2 / ITR-Flickr / ITR-COCO
    # column_key_name = {
    #     0: 'type', 1: 'text_set', 2: 'text_split', 3: 'text_id', 4: 'text', 5: 'lemmas', 6: 'img_src', 
    #     7: 'img_id', 8: 'img_file', 9: 'split_info', 10: 'label', 11: 'multi_label'
    # }
    # PT-COCO / sbu / conceptual
    column_key_name = {
        0: 'type', 1: 'text_set', 2: 'text_split', 3: 'text_id', 4: 'text', 5: 'lemmas', 6: 'img_src', 
        7: 'img_id', 8: 'img_file', 9: 'split_info', 10: 'label', 11: 'multi_label', 12: 'tsg', 
    }
    # PT-VG
    # column_key_name = {
    #     0: 'type', 1: 'text_set', 2: 'text_split', 3: 'text_id', 4: 'text', 5: 'lemmas', 6: 'img_src', 
    #     7: 'img_id', 8: 'img_file', 9: 'split_info', 10: 'label', 11: 'multi_label', 12: 'tsg', 
    #     13: 'region_id', 
    # }
    need_json_dumps = [4, 5, 10, 11,]
    with open(anno_file, 'r') as f:
        json_anno = json.load(f)
    splits_name = list(json_anno.keys())
    for split_name in splits_name:
        print('starting transfer', split_name)
        annos = json_anno[split_name]
        save_tsv_file = tsv_file.replace('.tsv', '_{}.tsv'.format(split_name))
        rows = []
        for anno in tqdm(annos):
            row = []
            for i in range(len(column_key_name)):
                if i in need_json_dumps or column_key_name[i] == 'tsg':
                    row.append(json.dumps(anno[column_key_name[i]]))
                elif column_key_name[i] == 'gt_boxes':
                    gt_boxes_encoded = base64.b64encode(np.array(anno[column_key_name[i]], dtype=np.float32))
                    row.append(gt_boxes_encoded)
                else:
                    row.append(anno[column_key_name[i]])
            rows.append(row)
        tsv_writer(rows, save_tsv_file)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--anno-file', dest='anno_file', type=str)
    parser.add_argument('--tsv-file', dest='tsv_file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # args = parse_args()
    # transfer(args.anno_file, args.tsv_file)
    # REC
    # anno_file = 'datasets/annotations/rec-refcoco/rec_refcoco_annotations.json'
    # tsv_file = 'datasets/annotations/rec-refcoco/rec_refcoco_annotations.tsv'
    # anno_file = 'datasets/annotations/rec-refcocoplus/rec_refcocoplus_annotations.json'
    # tsv_file = 'datasets/annotations/rec-refcocoplus/rec_refcocoplus_annotations.tsv'
    # anno_file = 'datasets/annotations/rec-refcocog/rec_refcocog_annotations.json'
    # tsv_file = 'datasets/annotations/rec-refcocog/rec_refcocog_annotations.tsv'
    # VQA
    # anno_file = 'datasets/annotations/vqa-vqav2/vqa_vqav2_annotations.json'
    # tsv_file = 'datasets/annotations/vqa-vqav2/vqa_vqav2_annotations.tsv'
    # ITR
    # anno_file = 'datasets/annotations/itr-flickr/itr_flickr_annotations.json'
    # tsv_file = 'datasets/annotations/itr-flickr/itr_flickr_annotations.tsv'
    # Pre-Train
    # anno_file = 'datasets/annotations/pt-coco/pt_coco_annotations.json'
    # tsv_file = 'datasets/annotations/pt-coco/pt_coco_annotations.tsv'
    # anno_file = 'datasets/annotations/pt-vg/pt_vg_annotations.json'
    # tsv_file = 'datasets/annotations/pt-vg/pt_vg_annotations.tsv'
    # anno_file = 'datasets/annotations/pt-sbu/pt_sbu_annotations.json'
    # tsv_file = 'datasets/annotations/pt-sbu/pt_sbu_annotations.tsv'
    anno_file = 'datasets/annotations/pt-conceptual/pt_conceptual_annotations.json'
    tsv_file = 'datasets/annotations/pt-conceptual/pt_conceptual_annotations.tsv'
    transfer(anno_file, tsv_file)
