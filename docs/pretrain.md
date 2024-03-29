# Pretraining

We provide the pretraining script as well as the pretrained model to support the finetuning on downstream tasks.  More models will be updated continuously.



## Pretraining Scripts

The following script run the pretraining on the MSCOCO, VG, SBU, Conceptual datasets. 

```bash
$ bash scripts/train-pt.sh
```

As the pretraining stage complete, you may run the [finetuning scripts](finetune.md) on downstream tasks. 



## Pretrained Models

|    name     | model size |                           download                           |
| :---------: | :--------: | :----------------------------------------------------------: |
| ROSITA-base |    116M    | [model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EYTZaPGm3DRBsbWDSJA8IQMB_-me1J7JAIqyuxzzs1dMyw?download=1) |

