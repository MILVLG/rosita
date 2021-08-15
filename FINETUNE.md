# Finetuning

We provide the scripts as well as the trained models on the downstream tasks as follows. Note that before you run the finetuning scripts, you need to download the corresponding pretrained ROSITA model [here](PRETRAIN.md).



## Table of Contents

- <a href='#VQA'>Visual Question Answering (VQA)</a>
- <a href='#REC'>Referring Expression Comhrehension (REC)</a>
- <a href='#ITR'>Image-Text Retrieval (ITR)</a>



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
<td align="center" valign="middle">89.91</td>
<td align="center" valign="middle">56.07</td>
<td align="center" valign="middle">64.29</td>
<td align="center" valign="middle">73.97</td>
<td align="center" valign="middle">89.76</td>
<td align="center" valign="middle">55.81</td>
<td align="center" valign="middle">64.39</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?download=1">model</a></td>
</tr>
</table>


## REC

### RefCOCO

The following script run the finetuning on the `train` split of RefCOCO.
```bash
$ bash scripts/train-rec-refcoco.sh
```


As the training stage complete, you may run the following script to run evaluation on the `val/testA/testB` split of RefCOCO. 
```bash
bash scripts/test-rec-refcoco.sh
```

We also provide the checkpoint model to reproduce the following results on `val` and `testA/B` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->

<th align="center" rowspan="2">Name</th>
<th align="center" colspan="3">RefCOCO</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">val</th>
<th align="center" valign="middle">testA</th>
<th align="center" valign="middle">testB</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">84.79</td>
<td align="center" valign="middle">87.99</td>
<td align="center" valign="middle">78.28</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ETr_J0Ak8L9Phc8JPQG_pZYBMqG35EdwfuFTZUet1vrKSQ?download=1">model</a></td>
</tr>
</table>

### RefCOCOpuls

The following script run the finetuning on the `train` split of RefCOCOplus.
```bash
bash scripts/train-rec-refcoco+.sh
```


As the training stage complete, you may run the following script to run evaluation on the `val/testA/testB` split of RefCOCOplus. 
```bash
bash scripts/test-rec-refcoco+.sh
```
We also provide the checkpoint model to reproduce the following results on `val` and `testA/B` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->

<th align="center" rowspan="2">Name</th>
<th align="center" colspan="3">RefCOCOplus</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">val</th>
<th align="center" valign="middle">testA</th>
<th align="center" valign="middle">testB</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">76.06</td>
<td align="center" valign="middle">82.01</td>
<td align="center" valign="middle">67.40</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ERCWoT4cpVNJr9OOXoNjdHQB6WelAAtAMj9rbE8DAoec0w?download=1">model</a></td>
</tr>
</table>

### RefCOCOg

The following script run the finetuning on the `train` split of RefCOCOg.
```bash
bash scripts/train-rec-refcocog.sh
```


As the training stage complete, you may run the following script to run evaluation on the `val/test` split of RefCOCOg. 
```bash
bash scripts/test-rec-refcocog.sh
```
We also provide the checkpoint model to reproduce the following results on `val` and `testA/B` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->

<th align="center" rowspan="2">Name</th>
<th align="center" colspan="2">RefCOCOg</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">val</th>
<th align="center" valign="middle">test</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">78.23</td>
<td align="center" valign="middle">78.25</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EXxpfotSwO1Jgu5POVbLQ24BNUPHWfdCS07UyNJNHWP4vQ?download=1">model</a></td>
</tr>
</table>


## ITR

### ITR-COCO

The following script run the finetuning on the `train` split of ITR-COCO.
```bash
bash scripts/train-itr-coco.sh
```

As the training stage complete, you may run the following script to run evaluation on the `testall` split of ITR-COCO. 
```bash
bash scripts/test-itr-coco.sh
```
We also provide the checkpoint model to reproduce the following results on `testall` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->

<th align="center" rowspan="2">Name</th>
<th align="center" colspan="3">IR-COCO</th>
<th align="center" colspan="3">TR-COCO</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">R@1</th>
<th align="center" valign="middle">R@5</th>
<th align="center" valign="middle">R@10</th>
<th align="center" valign="middle">R@1</th>
<th align="center" valign="middle">R@5</th>
<th align="center" valign="middle">R@10</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">54.40</td>
<td align="center" valign="middle">80.92</td>
<td align="center" valign="middle">88.60</td>
<td align="center" valign="middle">71.26</td>
<td align="center" valign="middle">91.62</td>
<td align="center" valign="middle">95.58</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/Eet3nwx-RIVLt3v17tzsIhIBOnsTapsUVGR5HI2Hg_VKNQ?download=1">model</a></td>
</tr>
</table>

### ITR-Flickr

The following script run the finetuning on the `train` split of ITR-Flickr.
```bash
bash scripts/train-itr-flickr.sh
```

As the training stage complete, you may run the following script to run evaluation on the `test` split of ITR-Flickr.
```bash
bash scripts/test-itr-flickr.sh
```
We also provide the checkpoint model to reproduce the following results on `test` split using the testing script above.

<table><tbody>
<!-- TABLE HEADER -->

<th align="center" rowspan="2">Name</th>
<th align="center" colspan="3">IR-Flickr</th>
<th align="center" colspan="3">TR-Flickr</th>
<th align="center" rowspan="2">Downloads</th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle">R@1</th>
<th align="center" valign="middle">R@5</th>
<th align="center" valign="middle">R@10</th>
<th align="center" valign="middle">R@1</th>
<th align="center" valign="middle">R@5</th>
<th align="center" valign="middle">R@10</th>
</tr>
<tr>
<td align="center" nowrap>ROSITA-base</td>
<td align="center" valign="middle">74.08</td>
<td align="center" valign="middle">92.44</td>
<td align="center" valign="middle">96.08</td>
<td align="center" valign="middle">88.90</td>
<td align="center" valign="middle">98.10</td>
<td align="center" valign="middle">99.30</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYsabcbcrflOinC4LyuAfzYBaCucZZ6wv7e7k1QgTG32JA?download=1">model</a></td>
</tr>
</table>
