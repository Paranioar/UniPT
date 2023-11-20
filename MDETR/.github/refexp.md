# Referring Expression Comprehension

### Usage

1. Run `pip install -r requirement.text` to install the exactly same dependencies.

2. Or use `conda-pack` command to install the environment downloaded from [here](https://pan.baidu.com/s/10UhHqilYNxOt9bTBcxfftQ) with [0dhw]:
    ```bash
    pip install conda-pack
    mkdir -p [path_to_conda_env]    # (e.g., ~/anaconda/envs/ENV_NAME)
    tar -zxvf [ENV_NAME].tar.gz -C [path_to_conda_env]
    ```

### Data preparation
There are three datasets which have the same structure, to be used with three config files: configs/refcoco, configs/refcoco+ and configs/refcocog. Here we show instructions for refcoco but the same applies for the others. The config for this dataset can be found in configs/refcoco.json and is also shown below:

```json
{
    "combine_datasets": ["refexp"],
    "combine_datasets_val": ["refexp"],
    "refexp_dataset_name": "refcoco",
    "coco_path": "",
    "refexp_ann_path": "mdetr_annotations/",
}
```

The images for this dataset come from the COCO 2014 train split which can be downloaded from : [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the "coco_path" to the folder containing the downloaded images.
Download our [pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and place them in a folder called "mdetr_annotations". The `refexp_ann_path` should point to this folder.

### Script to reproduce results

Firstly, you should download the pre-trained [R101 checkpoint](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1) into `pretrained_weights/`.

#### Training

Run the script `./train_xxxx.sh` and you will obtain the model checkpoint in `runs/`.


#### Testing

Adjust the path and run the script `./test_xxxx.sh`

#### Our checkpoints

One can download our best checkpoints of [refcoco](https://pan.baidu.com/s/1g9UREAhU1e_YDhaLvz2M5Q), [refcoco+](https://pan.baidu.com/s/16PmF9Wj3JIoYy-q7yd6NzA), and [refcocog](https://pan.baidu.com/s/1HubpIX5Yz73zdOo1e0YAcg) with [0dhw].
