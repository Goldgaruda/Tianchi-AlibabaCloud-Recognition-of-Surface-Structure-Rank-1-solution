
Alibaba Cloud TIANCHI NLP Competition The introduction of the activity and whole dataset are available:
https://tianchi.aliyun.com/competition/entrance/531872/submission/735#
## Competition background
The competition questions are based on computer vision, requiring the contestants to use a given aerial image to train the model and complete the task of identifying buildings on the surface. 

## Problem description and data explanation
Remote sensing technology has become the most effective means to obtain land cover information, and remote sensing technology has been successfully applied to land cover detection, vegetation area detection and building detection tasks. This question uses aerial photography data, and contestants are required to complete surface building identification, and divide the surface aerial image pixels into two types: buildings with and without buildings.
## 运行环境

*一些常用的包*

pytorch 1.7.1

albumentations 0.5.2

segmentation_models_pytorch



## 解题思路

此题目是经典的语义分割问题，baseline用的fcn + resnet50。

略加跑通之后，大概82,3 左右。之后用数据增强来到85,86.

感觉基本到头了，于是调用smp包，换了经典的用于医学的Unet模型，骨干网络换了effientb4。

稍微加上TTA的预测，外加Unet加持下，突破了90.

由于担心过拟合，外加只训练了一个fold4，于是又加了一个unetplusplus 进来，然后换了一个fold0来训练。

最后经过多轮训练，单个模型都超过91.3。 联合之后，通过调节两个模型之间比例，0.9vs1.1时候，达到了91.48的A榜第二成绩。 由于算力有限，就没有训练其他的模型了。最后b榜时候，联合两个模型+tta,最后结果比a榜还要好。经过调参，反复测试，最终获得第一。

之前也考虑过deeplab等，但是要不就是效果一般，要不就是太大，训练慢，没仔细测试就放弃了。

## 运行方法

1. 将原始数据拷贝到对应目录下，可在天池比赛对应链接中下载。
2. 训练，运行python ./train_unet.py 训练unet, 运行python ./train_upp.py训练unetplusplus。
3. 运行 python ./ref_mixtta.py即可预测。 需要把训练好的两个模型的地址设置好，目前不同模型位于test下面分数upp和unet目录。或者运行对应的sh文件--也需要设置好目录和环境配置。

- 注：因进行多轮训练，消耗资源较大。如果只为学习，可较少数据量和训练折数

  

