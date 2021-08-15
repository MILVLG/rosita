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
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
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
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
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
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
</tr>
</table>


## Image/Text Retrieval

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
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
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
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">73.91</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle">xx.xx</td>
<td align="center" valign="middle"><a href="https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1">model</a></td>
</tr>
</table>