![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [ECCV 2020] Semantic Line Detection Using Mirror Attention and Comparative Ranking and Matching
#### Dongkwon Jin, Jun-Tae Lee, and Chang-Su Kim

<!--
![IVOS Image](Overall_Network.png)

\\[[Project page]](https://openreview.net/forum?id=bo_lWt_aA)
\\[[arXiv]](https://arxiv.org/abs/2007.08139)
-->

Official implementation for **"Semantic Line Detection Using Mirror Attention and Comparative Ranking and Matching"**.

### Requirements
- PyTorch 1.3.1
- CUDA 10.0
- CuDNN 7.6.5
- python 3.6

### Installation
Create conda environment:
```
    $ conda create -n DRM python=3.6 anaconda
    $ conda activate DRM
    $ pip install opencv-python==3.4.2.16
    $ conda install pytorch==1.3.1 torchvision cudatoolkit=10.0 -c pytorch
```

Download repository:
```
    $ git clone https://github.com/dongkwonjin/Semantic-Line-DRM.git
```

### Instruction

1. Download the following datasets. ```SEL``` and ```SEL_Hard``` datasets for semantic line detection. Other datasets for application.

|        Dataset      |            Custom          |      Original path     |
|:-------------------:|:--------------------------:|:----------------------:|
|          SEL        |          [Download]        |          [here]        |
|       SEL_Hard      |          [Download]        |                        |
|         ICCV        |          [Download]        |          [here]        |
|          NYU        |          [Download]        |          [here]        |
|       SYM_Hard      |          [Download]        |                        |
|    AVA landscape    |          [Download]        |          [here]        |


2. Download our [network parameters](https://drive.google.com/file/d/1SSGpOfhDKzoZl9jXeTvACLUUshBS1rNz/view?usp=sharing) if you want to get the performance of the paper.

3. Edit `config.py`. Please modify ```dataset_dir``` and ```paper_weight_dir```. We provide specific description of configuration in ```config.txt``` file.

4. Run with 
```
cd Semantic-Line-SLNet-master
python main.py
```

### Reference
```
@Inproceedings{
Jin2020DRM,
title={Semantic Line Detection Using Mirror Attention and Comparative Ranking and Matching},
author={Dongkwon Jin, Jun-Tae Lee, and Chang-Su Kim},
booktitle={ECCV},
year={2020}
}
```
