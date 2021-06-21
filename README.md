# Re-ID-of-Mechanical-Components based on Faster R-CNN
## 1. Introduction
This is the network built for my master thesis "Object Re-identification for Assembly Parts to Aid Manual Assembly" 
This Re-ID system has a siamese structure and the locating part is based on Faster R-CNN.
5 losses are to be trained in the learning process.
The model is modified to fit the specific datasets and usage of the Institute https://www.ifl.kit.edu/index.php. It was trained with synthetic data and applied in the real world.
I will load the pretrained pth file later. 
It is currently not friendly to be used directly, but i will try to modify it if i had time...
![image](https://user-images.githubusercontent.com/78811701/117865317-21646c00-b296-11eb-86c4-43d92d19cdc2.png)

## 2. Dependencies
***GPU only
***Anaconda
***Pytorch
## 3. Performance
Cross Validation is adapted to generate the metrics and thresholds to distinguish same or not.
The mAP will be added later...

### 3.1 Best Performance on validation set
|     m.Accuracy     |   AUC  |   Best Threshold  |
| :----------------: | :----: |  :--------------: |
|      0.82     |  0.92  | 0.35

![image](https://user-images.githubusercontent.com/78811701/117831484-e69e0c00-b274-11eb-827a-0be99321dfba.png)

Examples

![image](https://user-images.githubusercontent.com/78811701/117832069-73e16080-b275-11eb-85b1-37253065f578.png)
![image](https://user-images.githubusercontent.com/78811701/117833228-71cbd180-b276-11eb-821d-0dd13431028b.png)
![image](https://user-images.githubusercontent.com/78811701/117833288-7f815700-b276-11eb-827c-25bd5691a19f.png)

### 3.2 speed
![image](https://user-images.githubusercontent.com/78811701/117832156-878cc700-b275-11eb-81e7-7936986d1d97.png)
![image](https://user-images.githubusercontent.com/78811701/117832167-8bb8e480-b275-11eb-97d2-bc629b95d312.png)

## Citation
https://github.com/chenyuntc/simple-faster-rcnn-pytorch
