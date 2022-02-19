## [Occulsion Aware Unsupervised Learning of Optical Flow from Video](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11605/116050T/Occlusion-aware-unsupervised-learning-of-optical-flow-from-video/10.1117/12.2588381.short)

## Introduction
We proposed a new method of dealing with occulsion problem in unsupervised learning of optical flow by calculating occlusion mask. 
Compared with UnFlow(AAAI 2018) and OAFlow(CVPR 2018), we achieved more precise results in KITTI dataset.
|method |KITTI 2012| KITTI 2015 train set| KITTI 2015 test set|
|------ |----------|-----------|------------|
|UnFlow| 3.78 | 8.80 | 23.27% |
|OAFlow| 3.55 | 8.88| 31.2% |
|Ours | **2.67** | **7.1** | 22% |

## Installation
The code is based on Python3.6. You could use either virtualenv or conda to setup a specified environment. And then run:
```
pip install -r requirements.txt
```

## Run experiments

### Prepare training data:
1. Download KITTI raw dataset using the <a href="http://www.cvlibs.net/download.php?file=raw_data_downloader.zip">script</a> provided on the official website. You also need to download <a href="http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow">KITTI 2015 dataset</a> to evaluate the predicted optical flow. 

### Training:
1. Modify the configuration file in the ./config directory to set up your path. The config file contains the important paths and default hyper-parameters used in the training process.

```bash
1. python train.py --config_file ./config/kitti.yaml --gpu [gpu_id] --mode flow --prepared_save_dir [name_of_your_prepared_dataset] --model_dir [your/directory/to/save/training/models]
```
If you are running experiments on the dataset for the first time, it would first process data and save in the [prepared_base_dir] path defined in your config file. 

### Evaluation:
the network weights after training on kitti raw data is [here](https://drive.google.com/file/d/1_sXqSUysOy56JiUjVmwdM1Z-1D3B54HH/view?usp=sharing)

1. To evaluate the optical flow estimation on KITTI 2015, run:
```bash
python test.py --config_file ./config/kitti.yaml --gpu [gpu_id] --mode flow --task kitti_flow --pretrained_model [path/to/your/model]
```

### Acknowledgement
We implemented our idea based on <a href="https://github.com/B1ueber2y/TrianFlow">TrainFlow</a>

### Cite
@inproceedings{10.1117/12.2588381,
author = {Jianfeng Li and Junqiao Zhao and Shuangfu Song and Tiantian Feng},
title = {{Occlusion aware unsupervised learning of optical flow from video}},
volume = {11605},
booktitle = {Thirteenth International Conference on Machine Vision},
editor = {Wolfgang Osten and Dmitry P. Nikolaev and Jianhong Zhou},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {224 -- 231},
year = {2021},
doi = {10.1117/12.2588381},
URL = {https://doi.org/10.1117/12.2588381}
}

### one more thing
This unsupervised optical flow estimation project is intergrated and updated into https://github.com/jianfenglihg/Unsupervised_depth_flow_egomotion.
