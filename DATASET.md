# Datasets

We provide the extracted image region features, and original text annotations for each downstream tasks.

## Image Region Features
The image features are extracted using the [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) strategy, with each image being represented as a fix number (k=36) of 2048-D features. We store the features for each image in a `.npz` file, and convert them to a `.tsv` file for COCO and Flickr. You can prepare the visual features by yourself or download the extracted features. The downloaded files contains two files: [**mscoco_bua_r101_fix36.tar.gz**](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETVLb0s1EltHizf4mqk16a4BFzujqng5ffAIrAP48egZKQ?download=1) and [**flickr_bua_r101_fix36.tar.gz**](), corresponding to the features of the MS-COCO/Flickr images respectively. 