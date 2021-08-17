# Datasets

The datasets consist of the **Image Features** and **Text Annotations** parts. The `datasets` folder forms the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- ...

```

We provide the extracted image region features, and formatted text annotations for each downstream tasks. 

## Image Features
We use pre-extracted region region features for each image. For the finetuning tasks in this repository, two image datasets `COCO` and `Flickr` are used. 

The image features are extracted using the commonly-used [bottom-up-attention](https://arxiv.org/abs/1707.07998) manner, with each image being represented as a fixed number (k=36) of 2048-D features. You can **download the extracted features** or **extract the visual features** by yourself.

---
### Download the extracted features

We provide the extracted image features for two datasets in `.tsv` format, namely [**mscoco_bua_r101_fix36.tar.gz**](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETVLb0s1EltHizf4mqk16a4BFzujqng5ffAIrAP48egZKQ?download=1) and [**flickr_bua_r101_fix36.tar.gz**](), corresponding to the features for `COCO` and `Flickr`, respectively. Using the command below to unzip the downloaded files to the proper places:
```bash
$ tar -xzvf *_bua_r101_fix36.tar.gz datasets/imgfeats/
```
Each zipped file contain three files: `imgfeat.tsv`, `imgfeat.lineidx`, and `img_feat_offset_map.json`  

The `datasets/` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- mscoco_bua_r101_fix36
    |   |   |-- imgfeat.tsv
    |   |   |-- imgfeat.lineidx
    |   |   |-- img_feat_offset_map.json
    |   |-- flickr_bua_r101_fix36
    |       |-- imgfeat.tsv
    |       |-- imgfeat.lineidx
    |       |-- img_feat_offset_map.json
    |-- annotations
        |-- ...

```

---

### Extract the visual features

Alternatively, the above image features can be extracted by using our [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) repository. To make a fair comparison to other VLP methods,  we use the standard [Faster R-CNN with R101-fix36](https://github.com/MILVLG/bottom-up-attention.pytorch/blob/master/configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml) model ([ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1)) to extract the features. 

The following command below extract features for each image in `$IMAGE_DIR` and output corresponding features in `.npz` format to `$OUT_DIR`:

```bash
$ python extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode roi_feats \
         --min-max-boxes 36, 36 \
         --config-file configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml \
         --image-dir $IMAGE_DIR \
         --out-dir $OUT_DIR \
         --resume
```

The `$IMAGE_DIR` refers to a folder contains `.jpg` format images, and the `$OUT_DIR` refers to the folder that stores extracted `.npz` format features. 

After obtaining the `.npz` features for the whole dataset, you can use [transfer_npz2tsv.py](https://github.com/MILVLG/rosita/blob/main/rosita/utils/transfer_npz2tsv.py) to convert these `.npz` features into one `.tsv` file as follows:

```bash
$ python transfer_npz2tsv.py \
         -- npz-dir $NPZ_DIR \
         -- tsv-dir $TSV_DIR
```

The `$NPZ_DIR` is the folder contains `.npz` format features, and the `$TSV_DIR` folder stores the converted `.tsv` format features, it will contain three files: `imgfeat.tsv`, `imgfeat.lineidx`, and `img_feat_offset_map.json`. 

After preparing the visual features, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- $COCO_TSV_DIR
    |   |   |-- imgfeat.tsv
    |   |   |-- imgfeat.lineidx
    |   |   |-- img_feat_offset_map.json
    |   |-- $Flickr_TSV_DIR
    |       |-- imgfeat.tsv
    |       |-- imgfeat.lineidx
    |       |-- img_feat_offset_map.json
    |-- annotations
        |-- ...

```

The `$COCO_TSV_DIR` and the `$Flickr_TSV_DIR` are the `$TSV_DIR` folders for COCO and Flickr, respectively. Make sure that the paths are consistent with the settings in the config files. For simplicity, we recommend setting `$COCO_TSV_DIR` and `$Flickr_TSV_DIR` to `mscoco_bua_r101_fix36` and `flickr_bua_r101_fix36`, respectively. 



## Text Annotations

For each downstream task (i.e., dataset), we provide the formatted annotation files as follows. 

### Visual Question Answering (VQA)

For the VQA task, we use VQAv2 datasets. You can download the formatted annotations [here]() and run the following command to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf vqa-vqav2.tar.gz
```
After unzipping, there will be a folder `vqa-vqav2`  containing several files. The  `vqa_vqav2_annotations.json` file, which is the primary annotation file for VQAv2 including the `train`, `val`, and `test` splits. To perform offline validation on the `val` and `minival` splits, we additionally provide four files:

- v2_OpenEnded_mscoco_val2014_questions.json
- v2_mscoco_val2014_annotations.json
- v2_OpenEnded_mscoco_minival2014_questions.json
- v2_mscoco_minival2014_annotations.json

After that, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- vqa-vqav2
        |   |-- vqa_vqav2_annotations.json
        |   |-- v2_OpenEnded_mscoco_val2014_questions.json
        |   |-- v2_mscoco_val2014_annotations.json
        |   |-- v2_OpenEnded_mscoco_minival2014_questions.json
        |   |-- v2_mscoco_minival2014_annotations.json
        |-- ...

```


### Referring Expression Comprehension (REC)

For the REC task, we use RefCOCO, RefCOCOplus, and RefCOCOg datasets. You can download the formatted annotations here ([RefCOCO](), [RefCOCOplus](), [RefCOCOg]()) and run the following commands to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf rec-refcoco.tar.gz
$ tar -xzvf rec-refcocoplus.tar.gz
$ tar -xzvf rec-refcocog.tar.gz
```
 Similarly, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- rec-refcoco
        |   |-- rec_refcoco_annotations.json
        |-- rec-refcocoplus
        |   |-- rec_refcocoplus_annotations.json
        |-- rec-refcocog
        |   |-- rec_refcocog_annotations.json
        |-- ...

```

### Image-Text Retrieval (ITR)

For the ITR task, we use ITR-COCO and ITR-Flickr Datasets. You can download the formatted annotations here ([ITR-COCO](), [ITR-Flickr]()) and run the following commands to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf itr-coco.tar.gz
$ tar -xzvf itr-flickr.tar.gz
```
After unzipping, you will obtain a `itr-coco` folder and a `itr-flickr` folder. Each folder contains a `*_annotations.json` file and a `img_text_map.json` file. The latter file is used to store the different mappings between image-text pairs. 

After that, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- itr-coco
        |   |-- itr_coco_annotations.json
        |   |-- img_text_map.json
        |-- itr-flickr
        |   |-- itr_flickr_annotations.json
        |   |-- img_text_map.json
        |-- ...

```

## The Final Structure

Finally, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- mscoco_bua_r101_fix36
    |   |   |-- imgfeat.tsv
    |   |   |-- imgfeat.lineidx
    |   |   |-- img_feat_offset_map.json
    |   |-- flickr_bua_r101_fix36
    |       |-- imgfeat.tsv
    |       |-- imgfeat.lineidx
    |       |-- img_feat_offset_map.json
    |-- annotations
        |-- vqa-vqav2
        |   |-- vqa_vqav2_annotations.json
        |   |-- v2_mscoco_val2014_annotations.json
        |   |-- v2_OpenEnded_mscoco_val2014_questions.json
        |   |-- v2_mscoco_minival2014_annotations.json
        |   |-- v2_OpenEnded_mscoco_minival2014_questions.json
        |-- rec-refcoco
        |   |-- rec_refcoco_annotations.json
        |-- rec-refcocoplus
        |   |-- rec_refcocoplus_annotations.json
        |-- rec-refcocog
        |   |-- rec_refcocog_annotations.json
        |-- itr-coco
        |   |-- itr_coco_annotations.json
        |   |-- img_text_map.json
        |-- itr-flickr
            |-- itr_flickr_annotations.json
            |-- img_text_map.json
```