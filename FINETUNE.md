# Finetuning

We provide the scripts as well as the trained models on the downstream tasks as follows. Note that before you run the finetuning scripts, you need to download the corresponding pretrained ROSITA model [here](PRETRAIN.md).



## Table of Contents

- <a href='#VQA'>Visual Question Answering (VQA)</a>
- <a href='#REC'>Referring Expression Comhrehension (REC)</a>
- <a href='#Image/Text Retrieval'>Image-Text Retrieval (ITR)</a>



## VQA

The following script run the finetuning on the `train+trainval`split of VQAv2.
```bash
$ bash scripts/train-vqa-vqav2.sh
```

As the training stage complete, you may run the following script to run evaluation on the `test` split and generate a `result.json` file under the output folder. This file can be submitted to the online server to obtain the performance.  

```bash
$ bash scripts/test-vqa-vqav2.sh
```

We also provide the checkpoint model to reproduce the following results on `test-dev` and `test-std` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th align="center" rowspan="2">Name</th>
<th align="center" colspan="4">Test-dev</th>
<th align="center" colspan="4">Test-std</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">All</th>
<th align="center" valign="middle">Y/N</th>
<th align="center" valign="middle">Num</th>
<th align="center" valign="middle">Other</th>
<th align="center" valign="middle">All</th>
<th align="center" valign="middle">Y/N</th>
<th align="center" valign="middle">Num</th>
<th align="center" valign="middle">Other</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">73.97</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
</tr>



## REC

Script to finetune for ROSITA base model on RefCOCO.
```bash
$ bash scripts/train-rec-refcoco.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [rec-refcoco-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETr_J0Ak8L9Phc8JPQG_pZYBMqG35EdwfuFTZUet1vrKSQ?e=tWPNId?download=1)

Script to test for ROSITA base model on RefCOCO.
```bash
bash scripts/test-rec-refcoco.sh
```

Script to finetune for ROSITA base model on RefCOCOplus.
```bash
bash scripts/train-rec-refcoco+.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [rec-refcocoplus-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ERCWoT4cpVNJr9OOXoNjdHQB6WelAAtAMj9rbE8DAoec0w?e=EhjHzT?download=1)

Script to test for ROSITA base model on RefCOCOplus.
```bash
bash scripts/test-rec-refcoco+.sh
```

Script to finetune for ROSITA base model on RefCOCOg.
```bash
bash scripts/train-rec-refcocog.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [rec-refcocog-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXxpfotSwO1Jgu5POVbLQ24BNUPHWfdCS07UyNJNHWP4vQ?e=VzcuhU?download=1)

Script to test for ROSITA base model on RefCOCOg.
```bash
bash scripts/test-rec-refcocog.sh
```



## Image/Text Retrieval

Script to finetune for ROSITA base model on ITR-COCO.
```bash
bash scripts/train-itr-coco.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [itr-coco-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Eet3nwx-RIVLt3v17tzsIhIBOnsTapsUVGR5HI2Hg_VKNQ?e=O2S19T?download=1)

Script to test for ROSITA base model on ITR-COCO.
```bash
bash scripts/test-itr-coco.sh
```

Script to finetune for ROSITA base model on ITR-Flickr.
```bash
bash scripts/train-itr-flickr.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [itr-flickr-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYsabcbcrflOinC4LyuAfzYBaCucZZ6wv7e7k1QgTG32JA?e=jgYBOR?download=1)

Script to test for ROSITA base model on ITR-Flickr.
```bash
bash scripts/test-itr-flickr.sh
```
