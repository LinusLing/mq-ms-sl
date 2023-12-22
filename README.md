# Context-weighted Multi-Query Partially Relevant Video Retrieval

<img src="https://github.com/LinusLing/mq-ms-sl/blob/main/figures/MQPRVR.png" width="1100px">

## Table of Contents

* [Environments](#environments)
* [MQ-MS-SL on Activitynet](#MQ-MS-SL-on-activitynet)
  * [Required Data](#Required-Data-1)
  * [Model Training](#Training-1)
  * [Model Evaluation](#Evaluation-1)
  * [Expected Performance](#Expected-Performance-1)
* [MS-SL on Charades-STA](#MQ-MS-SL-on-Charades-STA)
  * [Required Data](#Required-Data-2)
  * [Model Training](#Training-2)
  * [Model Evaluation](#Evaluation-2)
  * [Expected Performance](#Expected-Performance-2)
* [Reference](#Reference)
* [Acknowledgement](#Acknowledgement)

## Environments 
* **python 3.9**
* **pytorch 2.1.0**
* **torchvision 0.16.0**
* **tensorboard 2.6.0**
* **tqdm 4.62.0**
* **easydict 1.9**
* **h5py 2.10.0**
* **cuda 12.2**

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install the required packages.
```
conda create --name mq_ms_sl python=3.9
conda activate mq_ms_sl
git clone https://github.com/LinusLing/mq-ms-sl.git
cd mq-ms-sl
pip install -r requirements.txt
conda deactivate
```

## MQ-MS-SL on Activitynet
### Required Data
Run the following script to download the video feature and text feature of the Activitynet dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
wget http://8.210.46.84:8787/prvr/data/activitynet.tar
tar -xvf activitynet.tar
```

### Training
Run the following script to train `MQ-MS-SL` network on Activitynet.
```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate mq-ms-sl

ROOTPATH=$HOME/VisualSearch
RUN_ID=runs_0
GPU_DEVICE_ID=0

./do_activitynet.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=activitynet-runs_0-2022_07_11_20_27_02

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

We also provide the trained checkpoint on Activitynet, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.baidu.com/s/1Owodx7F55CPsOz8vsEU5sg?pwd=6aax).
```
DATASET=activitynet
FEATURE=i3d
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_activitynet

wget http://8.210.46.84:8787/prvr/checkpoints/checkpoint_activitynet.tar
tar -xvf checkpoint_activitynet.tar -C $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1 | R@5  |  R@10  | R@100 | SumR  |
| :---------: |:---:|:----:|:------:|:-----:|:-----:|
| Text-to-Video | 7.6 | 24.0 | 35.8 | 74.7  | 142.1 |

## MQ-MS-SL on Charades-STA

### Required Data
Run the following script to download the video feature and text feature of the Charades-STA dataset and place them in the specified path. The data can also be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4). Please refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for more description of the dataset.

```
ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH
wget http://8.210.46.84:8787/prvr/data/charades.tar
tar -xvf charades.tar
```

### Training
Run the following script to train `MQ-MS-SL` network on Charades-STA.

```
#Add project root to PYTHONPATH (Note that you need to do this each time you start a new session.)
source setup.sh

conda activate mq-ms-sl

ROOTPATH=$HOME/VisualSearch
RUN_ID=runs_0
GPU_DEVICE_ID=0

./do_charades.sh $RUN_ID $ROOTPATH $GPU_DEVICE_ID
```

### Evaluation
The model is placed in the directory $ROOTPATH/$DATASET/results/$MODELDIR after training. To evaluate it, please run the following script:
```
DATASET=charades
FEATURE=i3d_rgb_lgi
ROOTPATH=$HOME/VisualSearch
MODELDIR=charades-runs_0-2022_07_11_20_27_02

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```
We also provide the trained checkpoint on Charades-STA, run the following script to evaluate it. The model can also be downloaded from [Here](https://pan.baidu.com/s/1WPtRm5Lg52bwg3Ya9S3sSA?pwd=6ccw).
```
DATASET=charades
FEATURE=i3d_rgb_lgi
ROOTPATH=$HOME/VisualSearch
MODELDIR=checkpoint_charades

wget http://8.210.46.84:8787/prvr/checkpoints/checkpoint_charades.tar
tar -xvf checkpoint_charades.tar -C $ROOTPATH/$DATASET/results

./do_test.sh $DATASET $FEATURE $ROOTPATH $MODELDIR
```

### Expected performance 

|             | R@1 | R@5 | R@10 | R@100 | SumR |
| :---------: |:---:|:---:|:----:|:-----:|:----:|
| Text-to-Video | 1.6 | 6.7 | 11.9 | 49.2  | 69.5 |

## Reference

To be added.

## Acknowledgement
The codes are modified from [TVRetrieval](https://github.com/jayleicn/TVRetrieval), [ReLoCLNet](https://github.com/IsaacChanghau/ReLoCLNet) and [MS_SL](https://github.com/HuiGuanLab/ms-sl).

This work was supported by the National Natural Science Foundation of China(No. 61902347) and Scientific Research Fund of Zhejiang Provincial Education Department (No. Y202351804).
