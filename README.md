# LocalSampleNet
Created by Amit Shomer and Royi Avron.\
Advisors: Prof. Shai Avidan and Mr Itai Lang.\
Electrical Engineerig Faculty, Tel-Aviv University.

## Introduction
The popularity of 3D sensing devices like LiDAR and Stereo increased in the recent years. Point cloud is a set of points, produced by these
devices, represents a visual scene, and can be used for different tasks. By sampling a smaller number of points it’s possible to reduce cost 
and process time and still complete the task with high results.

LocalSampleNet(LSN) purpose is to extend  <a href="https://arxiv.org/pdf/1912.03663.pdf">SampleNet</a>, a point cloud sampling Neural network, by incorporate local information in the sampling process.
 It is based on <a href="https://arxiv.org/pdf/1706.02413.pdf">PointNet++</a>  architecture which introduced hierarchical neural network that extract local features from a small geometric structure
 neighborhood. By localizing the sampling network, we achieved better generalization ability for point clouds structures, compared to the global 
sampling of SampleNet (SN). 

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/inter.PNG)

## Method 
An overview of our LSN architecture is described in Figure 1. First, the classifier task was pre-trained on ModelNet40 dataset and its wights were frozen. 
Afterwards LSN trained on an input size N, in our case 1024, from ModelNet10 dataset (subset data of ModelNet40), and down sampled it to a smaller set 
of size P, that was fed to the task. We examined each implementation with sample ratio from 1 to 128 compared to the original input data. 

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/method.PNG)

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/Airplane.gif)

## Results
### Sample Method Comparison
LSN was evaluated on the disjoint sets MD10, MD30 which are subsets of MD40. Compared to SN, LSN generalization is better, higher by 6.8% at sample
 ratio 32 for MD30. LSN (concat feature vector) achieves high performance for structures that it had been trained on (MD10), with the same SR the accuracy
 is 86.2%, only 3.4% lower than SN.

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD10.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD30.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD40.PNG)

### Dropout Augmentation for Classification Task
A huge performance increase can be achieved with simple augmentation on the input data. Randomly reduced each point cloud by 0% tyo 87% while training the task classifier is enhance the accuracy for all sampling methods. 

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD10_dropout.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD30_dropout.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/docs/MD40_dropout.PNG)

## Installation and usage
This Code was tested under Pytorch 1.6.0, CUDA 10.2 on Ubuntu 20.04.1. You can find `requirement.txt` file in the main folder.

### Data preparation
Download sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. 
Move the uncompressed data folder to `data/modelnet40_normal_resampled` .

### Classification Task (PointNet)
Classification task already have been tranied and it's weights can be found in `/log/pointnet_cls_task/weight/`\
In case you want train the task network on your on, It can be clone and train it from <a href="https://github.com/yanx27/Pointnet_Pointnet2_pytorch">Pointnet_Pointnet2_pytorch</a>.

### Train LocalSampleNet
While traning LocalSampleNet task weights are being freeze and piped as described in Fig.3.\
For example, traning LSN on MD30 dataset with sample ration 32, 32 patches and 32 points per patch:  
```
python train_localsamplenet.py -modelnet 10 -num_out_points 32 -npatches 32 -n_sper_patch 32
```
Weight will be saved in this following format: \
 `log/LocalSamplenet/<YYYY-MM-DD_HH-MM>/checkpoints/sampler_cls_2609.pth `

### Evalute LocalSampleNet
Make sure that all configurations are identical to the train setup. Choose the model to evalute by date and time for example: 

```
python test_localsamplenet.py -modelnet 10 -num_out_points 32 -npatches 32 -n_sper_patch 32 -weights 2021-05-11_13-53
```


In order to reproduce results graphs as above and evalute the model with MD10, MD30 or MD40 use:
```
python test_localsamplenet.py -modelnet 30 -num_out_points 32 -npatches 32 -n_sper_patch 32 -weights 2021-05-11_13-53
``` 

Additional configurations where you may to train and evalute the model as reflected in the code arguments as: one_feture_vec, one_mlp_feture and reduce_to_8 (default not in use) can be found in our <a href="https://github.com/amitshomer/Local_samplenet/blob/master/docs/LocalSampleNet_Book_v3.pdf">Project Book</a>


## Acknowledgment
This code builds upon the code provided in <a href="https://github.com/yanx27/Pointnet_Pointnet2_pytorch">Pointnet_Pointnet2_pytorch</a>, <a href="https://github.com/itailang/SampleNet/tree/master/registration">SampleNet</a> and <a href="https://github.com/unlimblue/KNN_CUDA">KNN_CUDA</a>. We thank the authors for sharing their code.
