# <img src="misc/rosita_logo.png" width="100" align="right">ROSITA



## News & Update

**(15/08/2021)**
- Release the basic framework for ROSITA, including the pretrained base ROSITA model, as well as the scripts to run the fine-tuning and evaluation on three downstream tasks (i.e., VQA, REC, ITR) over six datasets.

## Introduction

This repository contains source code necessary to reproduce the results presented in our ACM MM paper [ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration](https://arxiv.org/abs/2108.07073), which encodes the c**ROS**s- and **I**n**T**r**A**-model prior knowledge in a in a unified scene graph to perform knowledge-guided vision-and-language pretraining. Compared with existing counterparts, ROSITA learns better **fine-grained semantic alignments** across different modalities, thus improving the capability of the pretrained model. 

<div align="center">
  <img src="misc\rosita.png"/>
</div>

## Performance

We compare ROSITA against existing state-of-the-art VLP methods on three downstream tasks. All methods use the base model of Transformer for a fair comparison. The trained checkpoints to reproduce these results are provided in [finetune.md](docs/finetune.md). 

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
    <td align="center" nowrap><sup><sub><b>73.91</b> | <b>73.97</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>84.79</b> | <b>87.99</b> | <b>78.28</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>76.06</b> | <b>82.01</b> | <b>67.40</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>78.23</b> | <b>78.25</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>54.40</b> | <b>80.92</b> | <b>88.60</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>71.26</b> | <b>91.62</b> | <b>95.58</b></sub></sup></td>
<td align="center" nowrap><sup><sub>74.08 | 92.44 | <b>96.08</b></sub></sup></td>
<td align="center" nowrap><sup><sub><b>88.90</b> | <b>98.10</b> | <b>99.30</b></sub></sup></td>
</tr>
<tr>
<td align="center" nowrap><sup><sub>SoTA-base</sub></sup></td>
<td align="center" nowrap><sup><sub>73.59 | 73.67</sub></sup></td>
<td align="center" nowrap><sup><sub>81.56 | 87.40 | 74.48</sub></sup></td>
<td align="center" nowrap><sup><sub>76.05 | 81.65 | 65.70</sub></sup></td>
<td align="center" nowrap><sup><sub>75.90 | 75.93</sub></sup></td>
<td align="center" nowrap><sup><sub>54.00 | 80.80 | 88.50</sub></sup></td>
<td align="center" nowrap><sup><sub>70.00 | 91.10 | 95.50</sub></sup></td>
<td align="center" nowrap><sup><sub><b>74.74</b> | <b>92.86</b> | 95.82</sub></sup></td>
<td align="center" nowrap><sup><sub>86.60 | 97.90 | 99.20</sub></sup></td>
</tr>

</tbody></table>



## Installation

#### Software and Hardware Requirements

We recommand a workstation with **4 GPU (>= 24GB, e.g., RTX 3090 or V100)**, **120GB memory** and **50GB free disk space**. We strongly recommend to use a SSD drive to guarantee high-speed I/O. Also, you should first install some necessary package as follows:

- Python >= 3.6
- PyTorch >= 1.4 with Cuda >=10.2
- torchvision >= 0.5.0
- Cython

```bash
# git clone
$ git clone https://github.com/MILVLG/rosita.git 

# build essential utils
$ cd rosita/rosita/utils/rec
$ python setup.py build
$ cp build/lib*/bbox.cpython*.so .
```



## Dataset Setup


To download the required datasets to run this project, please check [datasets.md](docs/datasets.md) for details. 

## Pretraining

Please check [pretrain.md](docs/pretrain.md) for the details for ROSITA pretraining. **We currently only provide the pretrained model to run finetuning on downstream tasks. The codes to run pretraining will be released later**.  

## Finetuning

Please check [finetune.md](docs/finetune.md) for the details for finetuning on downstream tasks. Scripts to run finetuning on downstream tasks are provided. Also, we provide trained models that can be directly evaluated to reproduce the results.  

## Acknowledgment

We appreciate the well-known open-source projects such as [LXMERT](https://github.com/airsplay/lxmert), [UNITER](https://github.com/ChenRocks/UNITER), [OSCAR](https://github.com/microsoft/Oscar), and [Huggingface](https://github.com/huggingface/transformers), which help us a lot when writing our codes. 

Yuhao Cui ([@cuiyuhao1996](https://github.com/cuiyuhao1996)) and Tong-An Luo ([@Zoroaster97](https://github.com/Zoroaster97)) are the main contributors to this repository. Please kindly contact them if you find any issue.


## Citations

Please consider citing this paper if you use the code:

```
@inProceedings{cui2021rosita,
  title={ROSITA: Enhancing Vision-and-Language Semantic Alignments via Cross- and Intra-modal Knowledge Integration},
  author={Cui, Yuhao and Yu, Zhou and Wang, Chunqi and Zhao, Zhongzhou and Zhang, Ji and Wang, Meng and Yu, Jun},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```
