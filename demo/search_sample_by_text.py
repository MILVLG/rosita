# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Tong-An Luo
 # -------------------------------------------------------- 

import json, argparse

text = [
    'a person on a skate board about to descend a ramp', 
    'an older woman standing next to two children', 
    'a silver bowl contain many carrots with green stems', 
    'a cat standing under a chair placed in a yard', 
    'a black train is driving under a bridge', 
    'a man blowing out candles on a cup cake', 
]


def search_by_text(formatted_data_file, sub_set_name):
    with open(formatted_data_file) as f:
        formatted_data = json.load(f)

    sub_set = formatted_data[sub_set_name]

    for id, data in enumerate(sub_set):
        if data['text'] in text:
            print('{}: {}'.format(id, data['text']))


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--data-file', dest='formatted_data_file', 
        default='datasets/annotations/itr-coco/itr_coco_annotations.json', type=str)
    parser.add_argument('--sub-set', dest='sub_set_name', default='dev', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    search_by_text(args.formatted_data_file, args.sub_set_name)