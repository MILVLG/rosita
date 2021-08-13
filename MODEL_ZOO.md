## Table of Contents
- <a href='#VQA'>VQA</a>
- <a href='#REC'>REC</a>
- <a href='#Image/Text Retrieval'>Image/Text Retrieval</a>

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


## Pre-trained Models
We provide pre-trained Rosita models.
- [rosita-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYTZaPGm3DRBsbWDSJA8IQMB_-me1J7JAIqyuxzzs1dMyw?e=RfFU56?download=1)

## VQA
Script to finetune for ROSITA base model on VQAv2.
```bash
bash scripts/train-vqa-vqav2.sh
```

We also provide the checkpoints finetuned on downstream tasks for run evaluation directly.
- [vqa-vqav2-base](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EVuxUtRFkRZJhjKTg9w8sesBKlM3hgcbZxE2nzSRbbAhRA?e=XNAH9v?download=1)
Script to test for ROSITA base model on VQAv2.
```bash
bash scripts/test-vqa-vqav2.sh
```

## REC
Script to finetune for ROSITA base model on RefCOCO.
```bash
bash scripts/train-rec-refcoco.sh
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