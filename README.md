# LocalSampleNet
Created by Amit Shomer and Royi Avron. 
Project supervisors: Prof. Shai Avidan and Mr Itai lang.
Tel-Aviv University

## Introduction
The popularity of 3D sensing devices like LiDAR and Stereo increased in the recent years. Point cloud is a set of points, produced by these
devices, represents a visual scene, and can be used for different tasks. By sampling a smaller number of points it’s possible to reduce cost 
and process time and still complete the task with high results.

LocalSampleNet(LSN) purpose is to extend  <a href="https://arxiv.org/pdf/1912.03663.pdf">SampleNet</a>, a point cloud sampling Neural network, by incorporate local information in the sampling process.
 It is based on <a href="https://arxiv.org/pdf/1706.02413.pdf">PointNet++</a>  architecture which introduced hierarchical neural network that extract local features from a small geometric structure
 neighborhood. By localizing the sampling network, we achieved better generalization ability for point clouds structures, compared to the global 
sampling of SampleNet (SN). 

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/inter.PNG)

## Method 
An overview of our LSN architecture is described in Figure 1. First, the classifier task was pre-trained on ModelNet40 dataset and its wights were frozen. 
Afterwards LSN trained on an input size N, in our case 1024, from ModelNet10 dataset (subset data of ModelNet40), and down sampled it to a smaller set 
of size P, that was fed to the task. We examined each implementation with sample ratio from 1 to 128 compared to the original input data. 

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/method.PNG)
All MLP and Fully Connected layers are followed by ReLU and batch-normalization layer except for the output layer.

## Results
LSN was evaluated on the disjoint sets MD10, MD30 which are subsets of MD40. Compared to SN, LSN generalization is better, higher by 6.8% at sample
 ratio 32 for MD30. LSN (concat feature vector) achieves high performance for structures that it had been trained on (MD10), with the same SR the accuracy
 is 86.2%, only 3.4% lower than SN.

![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/MD10.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/MD30.PNG)
![teaser](https://github.com/amitshomer/Local_samplenet/blob/master/MD40.PNG)
## Installation and usage
```
Will be completed 
```
.
##Acknowledgment

Will be completed 
