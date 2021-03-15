# Zero-Shot Cross-Lingual Dependency Parsing through Contextual Embedding Transformation

If you find anything useful in this work, please cite our [paper](https://arxiv.org/abs/2103.02212):
```
@article{xu2021zero,
  title={Zero-Shot Cross-Lingual Dependency Parsing through Contextual Embedding Transformation},
  author={Xu, Haoran and Koehn, Philipp},
  journal={arXiv preprint arXiv:2103.02212},
  year={2021}
}
```

<div align=center><img src="https://github.com/fe1ixxu/ZeroShot-CrossLing-Parsing/blob/master/model.png"/></div>

## Prerequisites
First install the virtual environmemt including required packages.
```
conda create --name clce python=3.7
conda activate clce
pip install -r requirements.txt
```

## Pre-Trained Model and Mapping
To reproduce the number in the paper, please find our pre-trained model and mappings in the following table. Note that the pre-trained model and mappings go through the iterative normalization prepreocessing and in a near-isotropic space.

### Pre-Trained Parser
Pre-trained English parser: [model.zip](https://drive.google.com/file/d/1WvNyKOJyKfAFcLRK7bimXatppZT1bCOR/view?usp=sharing)

### Pre-Trained Cross-Lingual Space Mapping:
|Language |    word-level mapping  | sense-level mapping| 
| ---------- | ---------- | ---------- | 
| es | [iter-norm-mean_es-en.th](https://drive.google.com/file/d/1TIvP4ZXenhRbMWk4T9uBpQq6ShuHSCCT/view?usp=sharing) | [iter-norm-multi_es-en.th](https://drive.google.com/file/d/1X_v-Du_fLgIBiRWhtohWda2Y1xdeTA2U/view?usp=sharing) |
| pt | [iter-norm-mean_pt-en.th](https://drive.google.com/file/d/12vZT_ti_2Rv619ENqZfsC1uSCekWcUZP/view?usp=sharing) | [iter-norm-multi_pt-en.th](https://drive.google.com/file/d/1tEMQbw7u4NA0cKsXrOxm_dF8I2V86sCb/view?usp=sharing) |
| ro | [iter-norm-mean_ro-en.th](https://drive.google.com/file/d/1eZeR9KU2f7wUu6KTSkpPB5c6_QfDgYnG/view?usp=sharing) | [iter-norm-multi_ro-en.th](https://drive.google.com/file/d/1_MYE8Ze4eU3DKQwPXDoftmG4hu5ciOcr/view?usp=sharing) |
| pl | [iter-norm-mean_pl-en.th](https://drive.google.com/file/d/1SM0z8fzESZ1HaeGxfNdd5vuRtFQNjqev/view?usp=sharing) | [iter-norm-multi_pl-en.th](https://drive.google.com/file/d/1ZFpR6jduv1c1iTEn5Zr31W3Y8Veyu181/view?usp=sharing) |
| fi | [iter-norm-mean_fi-en.th](https://drive.google.com/file/d/1OnxYguqyIUlPbsNNT0e0lr3-DoP0dwhU/view?usp=sharing) | [iter-norm-multi_fi-en.th](https://drive.google.com/file/d/19_q-alWsBVefAA2HiitJjrHBt__3u6aH/view?usp=sharing) |
| el | [iter-norm-mean_el-en.th](https://drive.google.com/file/d/13RA79A3n-AsDgSWQjrVhcng22dXrjUm_/view?usp=sharing) | [iter-norm-multi_el-en.th](https://drive.google.com/file/d/1O_wC2lHVr8MCL-wjoBmWfaU3peNuPxuK/view?usp=sharing) |

## Data
The zero-shot depdency parsing task is evaluated on [Universal Dependencies treebank 2.6](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226), which is available for free download.

## Zero-Shot Dependency Parsing
Before using English pre-trained model to parse treebanks in other languages, you have to point out the path for the pre-trained model, pre-trained mappings, and treebanks in `evaluate.sh`. After that, you can easily run:
```
./evaluate.sh lang   # e.g., ./evaluate.sh fi
```

## If you want to train a English parser yourself
You may need to change the path location for the train and dev dataset in the config file `allen_configs/enbert_IN.jsonnet`.
```
allennlp train allen_configs/enbert_IN.jsonnet -s PATH/TO/STORE/MODEL  --include-package src
```

## If you want to derive your own cross-lingual mappings
Please follow the instrcution [here](https://github.com/fe1ixxu/ZeroShot-CrossLing-Parsing/tree/master/mapper).
