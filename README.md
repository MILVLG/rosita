# <img src="misc/rosita_logo.png" width="100" align="right">ROSITA



## Introduction

This repository contains source code necessary to reproduce the results presented in our paper [ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration](https://arxiv.org/abs/2004.06165).

<img src="misc\rosita.PNG" width="900"> 

## Performance

<table><tbody>
<!-- START END-TO-END KEYPOINTS TABLE -->
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
<td align="center" nowrap><sup><sub>SOTA-O</sub></sup></td>
<td align="center" nowrap><sup><sub>73.16 | 73.44</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | -</sub></sup></td>
<td align="center" nowrap><sup><sub>54.00 | 80.80 | 88.50</sub></sup></td>
<td align="center" nowrap><sup><sub>70.00 | 91.10 | 95.50</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
</tr>
<tr>
<td align="center" nowrap><sup><sub>SOTA-V</sub></sup></td>
<td align="center" nowrap><sup><sub>73.59 | 73.67</sub></sup></td>
<td align="center" nowrap><sup><sub>81.56 | 87.40 | 74.48</sub></sup></td>
<td align="center" nowrap><sup><sub>76.05 | 81.65 | 65.70</sub></sup></td>
<td align="center" nowrap><sup><sub>75.90 | 75.93</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>74.74 | 92.86 | 95.82</sub></sup></td>
<td align="center" nowrap><sup><sub>86.60 | 97.90 | 99.20</sub></sup></td>
</tr>
<tr>
<td align="center" nowrap><sup><sub>SOTA-E</sub></sup></td>
<td align="center" nowrap><sup><sub>72.62 | 72.85</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>74.02 | 80.33 | 64.74</sub></sup></td>
<td align="center" nowrap><sup><sub>- | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>- | - | -</sub></sup></td>
<td align="center" nowrap><sup><sub>74.44 | 92.72 | 95.94</sub></sup></td>
<td align="center" nowrap><sup><sub>86.70 | 97.80 | 99.00</sub></sup></td>
</tr>
<tr>
<td align="center" nowrap><sup><sub>SOTA-U</sub></sup></td>
<td align="center" nowrap><sup><sub>72.70 | 72.91</sub></sup></td>
<td align="center" nowrap><sup><sub>81.24 | 86.48 | 73.94</sub></sup></td>
<td align="center" nowrap><sup><sub>75.31 | 81.30 | 65.68</sub></sup></td>
<td align="center" nowrap><sup><sub>74.31 | 74.51</sub></sup></td>
<td align="center" nowrap><sup><sub>50.33 | 78.52 | 87.16</sub></sup></td>
<td align="center" nowrap><sup><sub>64.40 | 87.40 | 93.08</sub></sup></td>
<td align="center" nowrap><sup><sub>72.52 | 92.36 | 96.08</sub></sup></td>
<td align="center" nowrap><sup><sub>85.90 | 97.10 | 98.80</sub></sup></td>
</tr>



<!-- END END-TO-END KEYPOINTS TABLE -->
</tbody></table>





## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset Setup


To download the required datasets for ROSITA, please check [DATASET.md](DATASET.md) for details. 

## Model Zoo

Check [MODEL_ZOO.md](MODEL_ZOO.md) for the scripts and provided checkpoints to run the finetuning on downstream tasks.


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
