# Re-ID-of-Mechanical-Components based on Faster R-CNN
## 1. Introduction
This is the network built for my master thesis "Object Re-identification for Assembly Parts to Aid Manual Assembly" 
This Re-ID system has a siamese structure and the locating part is based on Faster R-CNN
I will load the pretrained model. I will try to modify it later to make it friendly for other datasets.
## 2. Dependencies
GPU only
Anaconda
Pytorch
## 3. Performance
Cross Validation is adapted to generate the metrics and teh threshold to distinguish same or not
The mAP will be added later...

### 3.1 Best Performance on validation set
|     m.Accuracy     |   AUC  |   Best Threshold  |
| :----------------: | :----: |  :--------------: |
|      0.82     |  0.92  | 0.35

![image](https://user-images.githubusercontent.com/78811701/117831484-e69e0c00-b274-11eb-827a-0be99321dfba.png)

Examples

![image](https://user-images.githubusercontent.com/78811701/117832069-73e16080-b275-11eb-85b1-37253065f578.png)
![image](https://user-images.githubusercontent.com/78811701/117832092-76dc5100-b275-11eb-8efa-6d1fa42864da.png)
![image](https://user-images.githubusercontent.com/78811701/117831990-61ffbd80-b275-11eb-9d54-1f54dd8cff5d.png)

### 3.2 speed
![image](https://user-images.githubusercontent.com/78811701/117832156-878cc700-b275-11eb-81e7-7936986d1d97.png)
![image](https://user-images.githubusercontent.com/78811701/117832167-8bb8e480-b275-11eb-97d2-bc629b95d312.png)

## Citation
https://github.com/chenyuntc/simple-faster-rcnn-pytorch
