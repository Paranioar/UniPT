# CLIP-ViL: How Much Can CLIP Benefit Vision-and-Language Tasks?

## Introduction
This is code and checkpoints for the vision-and-language pre-training model in our paper "How Much Can CLIP Benefit Vision-and-Language Tasks?" ([Link](https://arxiv.org/abs/2107.06383)). CLIP-ViL with pre-training sets new single-model state-of-the-arts on benchmarks such as VQA v2.0 (76.70 on test-std).

The code is adopted from both the [CLIP](https://github.com/openai/CLIP) repo and the [LXMERT](https://github.com/airsplay/lxmert) repo. Many thanks to the authors of these repos~


## Data & Files Required

### Annotation files

1. Download [data file](https://pan.baidu.com/s/1x5SG-FSF2pi2WyMsByI9Pw) with [0dhw] and save them as `data/` file. Now, **gpa** and **mscoco** files are incomplete. 

### Image Data

2. Download COCO images and unzip them in `data/mscoco` file:
    ```bash
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/test2015.zip -P data/mscoco

    unzip data/mscoco/train2014.zip -d data/mscoco/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/ && rm data/mscoco/val2014.zip
    unzip data/mscoco/test2015.zip -d data/mscoco/ && rm data/mscoco/test2015.zip
    ```

3. Download original [GQA dataset](https://cs.stanford.edu/people/dorarad/gqa/download.html), including Scene Graphs (ver 1.1 / 42.7MB), Questions (ver 1.2 / 1.4GB), Images (20.3G), and unzip them in in `data/gpa` file.

4. Please refer to `data/shot_for_check.jpg` to check the download.

## Environment Setup

1. Run `pip install -r requirement.text` to install the exactly same dependencies.

2. Or use `conda-pack` command to install the environment downloaded from [here](https://pan.baidu.com/s/1BiiQWYr1HX1BNi2nl-EQhw) with [0dhw]:
    ```bash
    pip install conda-pack
    mkdir -p [path_to_conda_env]    # (e.g., ~/anaconda/envs/ENV_NAME)
    tar -zxvf [ENV_NAME].tar.gz -C [path_to_conda_env]
    ```

## Fine-Tuning

Caveats: 
To reduce CPU memory cost, we use shared memory to share annotation files across data readers. Be sure to delete any file with the prefix `sharearray_` under `/dev/shm/` after you finish training.

1. Training (Load [checkpoint](https://pan.baidu.com/s/1cz9RRjLZe7T_kzMKc1alnQ) with [0dhw] to `snap/pretrained/CLIP_VL_RN50x4/Epoch11_LXRT.pth`):
    ```bash
    ./scripts/[train_side_xxx.sh] 0,1,2,3 [model_name] 9590 4
    ```
    When the model finishes training, you will get `snap/vqa or gqa/[model_name]/BEST.pth`.

2. Testing:
    ```bash
    ./scripts/[test_side_xxx.sh] 0,1,2,3 [model_name] 9590 4
    ```
    It will generate `snap/vqa/[model_name]/test_predict.json` for vqa or `snap/gqa/[model_name]/submit_predict.json` for gqa, which could submited to the [VQA leaderboard](https://eval.ai/web/challenges/challenge-page/830/submission) or [GQA leaderboard](https://eval.ai/web/challenges/challenge-page/225/submission) for Dev and Std results.

3. One can download our best checkpoints of [vqa](https://pan.baidu.com/s/1v2hG3R6o3EkOTQLai4qOFQ) and [gqa](https://pan.baidu.com/s/100d9lzfmAYS85eOwMWOSZA) with [0dhw].