# <img src="misc/rosita_logo.png" width="100" align="right">ROSITA



## Introduction

This repository contains source code necessary to reproduce the results presented in our paper [ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration](https://arxiv.org/abs/2004.06165).

<img src="misc\rosita.png" width="900"> 

## Performance

<table><tbody>
<!-- TABLE HEADER -->
<!-- Info: we use wrap text in <sup><sub></sub><sup> to make is small -->
<th align="center"><sup><sub>Tasks</sub></sup></th>
<th align="center"><sup><sub>VQA</sub></sup></th>
<th align="center" colspan="3"><sup><sub>REC</sub></sup></th>
<th align="center" colspan="4"><sup><sub>ITR</sub></sup></th>
<!-- TABLE BODY -->
<tr>
<th align="center" valign="middle"><sup><sub>Datasets</sub></sup></th>
<th align="center" valign="middle"><sup><sub>VQAv2<br/>dev | std</sub></sup></th>
<th align="center" valign="middle"><sup><sub>RefCOCO<br/>val | testA | testB</sub></sup></th>
<th align="center" valign="middle"><sup><sub>RefCOCO+<br/>val | testA | testB</sub></sup></th>
<th align="center" valign="middle"><sup><sub>RefCOCOg<br/>val | test</sub></sup></th>
<th align="center" valign="middle"><sup><sub>IR-COCO<br/>R@1 | R@5 | R@10</sub></sup></th>
<th align="center" valign="middle"><sup><sub>TR-COCO<br/>R@1 | R@5 | R@10</sub></sup></th>
<th align="center" valign="middle"><sup><sub>IR-Flickr<br/>R@1 | R@5 | R@10</sub></sup></th>
<th align="center" valign="middle"><sup><sub>TR-Flickr<br/>R@1 | R@5 | R@10</sub></sup></th>
</tr>
<tr>
<td align="center" nowrap><sup><sub>ROSITA</sub></sup></td>
<td align="center" nowrap><sup><sub>73.91 | 73.97</sub></sup></td>
<td align="center" nowrap><sup><sub>84.79 | 87.99 | 78.28</sub></sup></td>
<td align="center" nowrap><sup><sub>76.06 | 82.01 | 67.40</sub></sup></td>
<td align="center" nowrap><sup><sub>78.23 | 78.25</sub></sup></td>
<td align="center" nowrap><sup><sub>54.40 | 80.92 | 88.60</sub></sup></td>
<td align="center" nowrap><sup><sub>71.26 | 91.62 | 95.58</sub></sup></td>
<td align="center" nowrap><sup><sub>74.08 | 92.44 | 96.08</sub></sup></td>
<td align="center" nowrap><sup><sub>88.90 | 98.10 | 99.30</sub></sup></td>
</tr>
<tr>
<td align="center" nowrap><sup><sub>SoTA-base</sub></sup></td>
<td align="center" nowrap><sup><sub>73.59 | 73.67</sub></sup></td>
<td align="center" nowrap><sup><sub>81.56 | 87.40 | 74.48</sub></sup></td>
<td align="center" nowrap><sup><sub>76.05 | 81.65 | 65.70</sub></sup></td>
<td align="center" nowrap><sup><sub>75.90 | 75.93</sub></sup></td>
<td align="center" nowrap><sup><sub>54.00 | 80.80 | 88.50</sub></sup></td>
<td align="center" nowrap><sup><sub>70.00 | 91.10 | 95.50</sub></sup></td>
<td align="center" nowrap><sup><sub>74.74 | 92.86 | 95.82</sub></sup></td>
<td align="center" nowrap><sup><sub>86.60 | 97.90 | 99.20</sub></sup></td>
</tr>

</tbody></table>





## Installation

### Requirements
- Pytorch 1.4
- torchvision 0.5.0
- Cython

### Setup
```bash
# git clone

cd rosita/rosita/utils/rec

python setup.py build

cp build/lib*/bbox.cpython*.so .
```


## Dataset Setup


To download the required datasets for ROSITA, please check [DATASET.md](DATASET.md) for details. 

## Pretrain

Check [PRETRAIN.md](PRETRAIN.md) for the provided pretrained checkpoints to run the finetuning on downstream tasks. We will provide the scripts to run the pretraining tasks later. 

## Finetune

Check [FINETUNE.md](FINETUNE.md) for the scripts and provided checkpoints to run the finetuning on downstream tasks.


## Citations

Please consider citing this paper if you use the code:

```
@inProceedings{cui2021rosita,
  title={ROSITA: Enhancing Vision-and-Language Semantic Alignments
via Cross- and Intra-modal Knowledge Integration},
  author={Cui, Yuhao and Yu, Zhou and Wang, Chunqi and Zhao, Zhongzhou and Zhang, Ji and Wang, Meng and Yu, Jun},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```
