# Datasets

The Datasets contains **Image Region Features** and **Text Annotations** two parts. And the `datasets` folder has the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- ...

```

We provide the extracted image region features, and original text annotations for each downstream tasks. 

## Image Region Features
We use the image region features of COCO and Flickr, the COCO features will be used on VQA task, REC task, and ITR-COCO task, the Flickr features will be only used on ITR-Flickr task. 

The image features are extracted using the [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) strategy, with each image being represented as a fixed number (k=36) of 2048-D features. We store the features for each image in a `.npz` file, and convert them to a `.tsv` file for COCO and Flickr respectively. You can **prepare the visual features** by yourself or **download the extracted features**. 

---
### Prepare the visual features
We use the [Faster R-CNN-k36](https://github.com/MILVLG/bottom-up-attention.pytorch/blob/master/configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml) model ([ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1)) to extract the region features of both COCO and Flickr. 

To extract features, you should first setup the **bottom-up-attention.pytorch** follow the [Installation](https://github.com/MILVLG/bottom-up-attention.pytorch#installation). And then using the command below:
```bash
$ python extract_features.py --mode caffe \
         --num-cpus 32 --gpu '0,1,2,3' \
         --extract-mode roi_feats \
         --min-max-boxes 36,36 \
         --config-file configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml \
         --image-dir {$IMAGE_DIR} \
         --out-dir {$OUT_DIR} \
         --resume
```
The `{$IMAGE_DIR}` should be a folder contains `.jpg` format images, and the `{$OUT_DIR}` will store the extracted `.npz` format features. 

You can use `.npz` features directly, but we recommend to use `.tsv` features. We use [transfer_npz2tsv.py](https://github.com/MILVLG/rosita/blob/main/rosita/utils/transfer_npz2tsv.py) to convert `.npz` features to `.tsv` features. To convert features, using the command below:
```bash
$ python transfer_npz2tsv.py \
         -- npz-dir {$NPZ_DIR} \
         -- tsv-dir {$TSV_DIR}
```
The `{$NPZ_DIR}` should be a folder contains `.npz` format features, and the `{$TSV_DIR}` will store the converted `.tsv` format features, it will contain a `imgfeat.tsv` file, a `imgfeat.lineidx` file, and a `img_feat_offset_map.json` file. 

After preparing the visual features, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- {$COCO_TSV_DIR}
    |   |   |-- imgfeat.tsv
    |   |   |-- imgfeat.lineidx
    |   |   |-- img_feat_offset_map.json
    |   |-- {$Flickr_TSV_DIR}
    |       |-- imgfeat.tsv
    |       |-- imgfeat.lineidx
    |       |-- img_feat_offset_map.json
    |-- annotations
        |-- ...

```

The `{$COCO_TSV_DIR}` and the `{$Flickr_TSV_DIR}` are the `{$TSV_DIR}` folders for COCO and Flickr respectively. And you need to make sure that the config files are using these folders. We use `mscoco_bua_r101_fix36` and `flickr_bua_r101_fix36` for `{$COCO_TSV_DIR}` and `{$Flickr_TSV_DIR}` by default. 

---
### Download the extracted features

The downloaded files contains two files: [**mscoco_bua_r101_fix36.tar.gz**](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETVLb0s1EltHizf4mqk16a4BFzujqng5ffAIrAP48egZKQ?download=1) and [**flickr_bua_r101_fix36.tar.gz**](), corresponding to the features of the MS-COCO/Flickr images respectively. Using the command below to unzip files:
```bash
$ tar -xzvf *_bua_r101_fix36.tar.gz {$DATASETS_DIR}
```
The `{$DATASETS_DIR}` should be the path to `datasets/imgfeats/` folder. After unzip, a `*_bua_r101_fix36` dir would be there, and this dir would contain a `imgfeat.tsv` file, a `imgfeat.lineidx` file, and a `img_feat_offset_map.json` file. 

After that, the `datasets` folder will have the following structure:
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

## Text Annotations

We provide the formatted text annotations for each downstream tasks. 

### VQA

In VQA task, we use VQAv2 datasets. You can download the formatted annotations [here]() and run the following command to unzip the annotations:
```bash
$ tar -xzvf vqa-vqav2.tar.gz {$ANNO_DIR}
```
The `{$ANNO_DIR}` is the path to `datasets/annotations` folder. After unzip, there will be a `vqa_vqav2_annotations.json` file, which is the text annotation of the VQAv2 dataset, a `v2_mscoco_val2014_annotations.json` file, a `v2_OpenEnded_mscoco_val2014_questions.json` file, a `v2_mscoco_minival2014_annotations.json` file, and a `v2_OpenEnded_mscoco_minival2014_questions.json` file in `vqa-vqav2` folder. 

The last 4 files are used to do evaluation on val and minival split. 

After that, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- vqa-vqav2
        |   |-- vqa_vqav2_annotations.json
        |   |-- v2_mscoco_val2014_annotations.json
        |   |-- v2_OpenEnded_mscoco_val2014_questions.json
        |   |-- v2_mscoco_minival2014_annotations.json
        |   |-- v2_OpenEnded_mscoco_minival2014_questions.json
        |-- ...

```


### REC

In REC task, we use RefCOCO, RefCOCOplus, and RefCOCOg datasets. You can download the formatted annotations here ([RefCOCO](), [RefCOCOplus](), [RefCOCOg]()) and run the following commands to unzip the annotations:
```bash
$ tar -xzvf rec-refcoco.tar.gz {$ANNO_DIR}
$ tar -xzvf rec-refcocoplus.tar.gz {$ANNO_DIR}
$ tar -xzvf rec-refcocog.tar.gz {$ANNO_DIR}
```
The `{$ANNO_DIR}` is the path to `datasets/annotations` folder. After unzip, there will be:
- a `rec_refcoco_annotations.json` file, which is the text annotation of the RefCOCO dataset, in `rec-refcoco` folder. 
- a `rec_refcocoplus_annotations.json` file, which is the text annotation of the RefCOCOplus dataset, in `rec-refcocoplus` folder. 
- a `rec_refcocog_annotations.json` file, which is the text annotation of the RefCOCOg dataset, in `rec-refcocog` folder. 

After that, the `datasets` folder will have the following structure:
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

### ITR

In ITR task, we use ITR-COCO and ITR-Flickr Datasets. You can download the formatted annotations here ([ITR-COCO](), [ITR-Flickr]()) and run the following commands to unzip the annotations:
```bash
$ tar -xzvf itr-coco.tar.gz {$ANNO_DIR}
$ tar -xzvf itr-flickr.tar.gz {$ANNO_DIR}
```
The `{$ANNO_DIR}` is the path to `datasets/annotations` folder. After unzip, there will be:
- a `itr_coco_annotations.json` file, which is the text annotation of the ITR-COCO dataset, and a `img_text_map.json` file in `itr-coco` folder. 
- a `itr_flickr_annotations.json` file, which is the text annotation of the ITR-Flickr dataset, and a `img_text_map.json` file in `itr-flickr` folder. 

The `img_text_map.json` files are used to:
- map the image name to the image idx
- map the image idx to the image name
- map the text annotation id to the image idx
- map the image idx to a text annotation id

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

## Structure

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