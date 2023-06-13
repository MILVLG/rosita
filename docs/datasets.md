# Datasets

The datasets of both pretraining stage and finetuning stage consist of the **Image Features** and **Text Annotations** parts. The `datasets` folder forms the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- ...

```

We provide the method to extract image region features and formatted text annotations for pretraining. For each downstream tasks, we provide the extracted image region features and formatted text annotations. 

## Pretraining

### Image Features
We use pre-extracted region region features for each image. For the pretraining stage in this repository, four image datasets `COCO`,  `VG` , `SBU` and `Conceptual` are used. 

The image features are extracted using the commonly-used [bottom-up-attention](https://arxiv.org/abs/1707.07998) manner, with each image being represented as a fixed number (k=36) of 2048-D features. You can **extract the visual features** by yourself. 

---
#### Extract the visual features

Image features can be extracted by using our [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) repository. To make a fair comparison to other VLP methods,  we use the standard [Faster R-CNN with R101-fix36](https://github.com/MILVLG/bottom-up-attention.pytorch/blob/master/configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml) model ([ckpt](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUKhQ3hSRv9JrrW64qpNLSIBGoOjEGCkF8zvgBP9gKax-w?download=1)) to extract the features. 

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

After preparing the visual features, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- $COCO_NPZ_DIR
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- $VG_NPZ_DIR
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- $SBU_NPZ_DIR
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- $CONCEPTUAL_NPZ_DIR
    |       |-- npz_files
    |            |-- ***.npz
    |            |-- ...
    |-- annotations
        |-- ...

```

The `$COCO_NPZ_DIR`,  `$VG_NPZ_DIR`, `$SBU_NPZ_DIR`and the `$CONCEPTUAL_NPZ_DIR` are the `$NPZ_DIR` folders for COCO, VG, SBU and Conceptual, respectively. Make sure that the paths are consistent with the settings in the config files. For simplicity, we recommend setting`$COCO_NPZ_DIR`,  `$VG_NPZ_DIR`, `$SBU_NPZ_DIR`and `$CONCEPTUAL_NPZ_DIR` to `mscoco_bua_r101_fix36`,  `visualgenome_bua_r101_fix36`,  `sbu_bua_r101_fix36` and `conceptual_bua_r101_fix36`, respectively. 



### Text Annotations

For each pretraining dataset, we provide the formatted annotation files in tsv format as follows. 

#### MSCOCO

You can download the formatted annotations [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EV40L1mEU2JHhHBKNlXBT0EB5crLR_pdBX9py3dmo4vbUQ?download=1) and run the following command to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf pt-coco.tar.gz
```
After unzipping, there will be a folder `pt-coco`  containing several files, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- pt-coco
        |   |-- text_piror_coco_vg_cc_sbu.json
        |   |-- pt_coco_annotations_train.tsv
        |   |-- pt_coco_annotations_train.lineidx
        |   |-- pt_coco_annotations_test.tsv
        |   |-- pt_coco_annotations_test.lineidx
        |   |-- pt_coco_annotations_testall.tsv
        |   |-- pt_coco_annotations_testall.lineidx
        |   |-- pt_coco_annotations_dev.tsv
        |   |-- pt_coco_annotations_dev.lineidx
        |-- ...

```


#### Visual Genome

You can download the formatted annotations [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ESET6YQxB_hHioTOF8jlwKABYQyLCY0QkL6cYBlamAbxXQ?download=1) and run the following command to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf pt-vg.tar.gz
```
After unzipping, there will be a folder `pt-vg`  containing several files, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- pt-vg
        |   |-- pt_vg_annotations_train.tsv
        |   |-- pt_vg_annotations_train.lineidx
        |-- ...

```


#### SBU

You can download the formatted annotations [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EattQurnnZRKo8rGfFp9pjkBzRk6C4fH50tiXkETdoSa8w?download=1) and run the following command to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf pt-sbu.tar.gz
```
After unzipping, there will be a folder `pt-sbu`  containing several files, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- pt-sbu
        |   |-- pt_sbu_annotations_train.tsv
        |   |-- pt_sbu_annotations_train.lineidx
        |-- ...

```

#### Conceptual

You can download the formatted annotations [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXqy0wuOZOpHvquweLnxDu8B8P2WHfGfc2jzNqZasMZekg?download=1) and run the following command to unzip the annotations:
```bash
$ cd datasets/annotations
$ tar -xzvf pt-conceptual.tar.gz
```
After unzipping, there will be a folder `pt-conceptual`  containing several files, the `datasets` folder will have the following structure:

```
|-- datasets
    |-- imgfeats
    |   |-- ...
    |-- annotations
        |-- pt-conceptual
        |   |-- pt_conceptual_annotations_train.tsv
        |   |-- pt_conceptual_annotations_train.lineidx
        |   |-- pt_conceptual_annotations_val.tsv
        |   |-- pt_conceptual_annotations_val.lineidx
        |-- ...

```

### The Final Structure

Finally, the `datasets` folder will have the following structure:
```
|-- datasets
    |-- imgfeats
    |   |-- mscoco_bua_r101_fix36
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- visualgenome_bua_r101_fix36
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- sbu_bua_r101_fix36
    |   |   |-- npz_files
    |   |        |-- ***.npz
    |   |        |-- ...
    |   |-- conceptual_bua_r101_fix36
    |       |-- npz_files
    |            |-- ***.npz
    |            |-- ...
    |-- annotations
        |-- pt-coco
        |   |-- text_piror_coco_vg_cc_sbu.json
        |   |-- pt_coco_annotations_train.tsv
        |   |-- pt_coco_annotations_train.lineidx
        |   |-- pt_coco_annotations_test.tsv
        |   |-- pt_coco_annotations_test.lineidx
        |   |-- pt_coco_annotations_testall.tsv
        |   |-- pt_coco_annotations_testall.lineidx
        |   |-- pt_coco_annotations_dev.tsv
        |   |-- pt_coco_annotations_dev.lineidx
        |-- pt-vg
        |   |-- pt_vg_annotations_train.tsv
        |   |-- pt_vg_annotations_train.lineidx
        |-- pt-sbu
        |   |-- pt_sbu_annotations_train.tsv
        |   |-- pt_sbu_annotations_train.lineidx
        |-- pt-conceptual
            |-- pt_conceptual_annotations_train.tsv
            |-- pt_conceptual_annotations_train.lineidx
            |-- pt_conceptual_annotations_val.tsv
            |-- pt_conceptual_annotations_val.lineidx
```



## Finetuning

We provide the extracted image region features, and formatted text annotations for each downstream tasks. 

### Image Features
We use pre-extracted region region features for each image. For the finetuning tasks in this repository, two image datasets `COCO` and `Flickr` are used. 

The image features are extracted using the commonly-used [bottom-up-attention](https://arxiv.org/abs/1707.07998) manner, with each image being represented as a fixed number (k=36) of 2048-D features. You can **download the extracted features** or **extract the visual features** by yourself.

---
#### Download the extracted features

We provide the extracted image features for two datasets in `.tsv` format, namely [**mscoco_bua_r101_fix36.tar.gz**](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETVLb0s1EltHizf4mqk16a4BFzujqng5ffAIrAP48egZKQ?download=1) and [**flickr_bua_r101_fix36.tar.gz**](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EdBx9IhZWChGn3aOCgkTfFMBJxEL8g6DhUrIF47n-zQP5Q?download=1), corresponding to the features for `COCO` and `Flickr`, respectively. Using the command below to unzip the downloaded files to the proper places:
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

#### Extract the visual features

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



### Text Annotations

For each downstream task (i.e., dataset), we provide the formatted annotation files as follows. 

#### Visual Question Answering (VQA)

For the VQA task, we use VQAv2 datasets. You can download the formatted annotations [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Ed3QUwvH5SpJkb7lm2bOEXgBqG5OmWExbF0rUq3Es9fmYg?download=1) and run the following command to unzip the annotations:
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


#### Referring Expression Comprehension (REC)

For the REC task, we use RefCOCO, RefCOCOplus, and RefCOCOg datasets. You can download the formatted annotations here ([RefCOCO](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EdhT3chPrkpBsXQrVspn0zUBdH2_hp1ee3Umo11Q5oGsGw?download=1), [RefCOCOplus](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaEE3QzS9qZMu_sAdB2T1lMB1gNjrnGcjEF6uf04dF6RIQ?download=1), [RefCOCOg](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYiM7AKo7IhGgyfDHh-VrfkBS18CWCmppQl_S8UtPNlnNg?download=1)) and run the following commands to unzip the annotations:
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

#### Image-Text Retrieval (ITR)

For the ITR task, we use ITR-COCO and ITR-Flickr Datasets. You can download the formatted annotations here ([ITR-COCO](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EWtpywySxi9IpR2TPJNqHzoBxj06VBQ0jY9vqJZ8RFeQvg?download=1), [ITR-Flickr](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EdVASEDfmIxClmSzNS6SKCIBJecUrGVKTYa5zBCJMKsepQ?download=1)) and run the following commands to unzip the annotations:
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

### The Final Structure

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
