# kitti-tracking-dynamic-objects (work in progress)
those scripts are used to generate dynamic objects for kitti tracking dataset

## preparation:
1. obtain instance mask of potential moving objects from a instance segmentation framework, such as [Detectron2](https://github.com/facebookresearch/detectron2). We use [Detectron2](https://github.com/facebookresearch/detectron2) with [config file](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) and [pretrained model](detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl) to obtain the instance mask. We only use the instance mask of class 0 to 8 from COCO dataset as potential moving objects. The mask should be saved as numpy array with shape (instance_num, height, width). We provide the script ```detectron2.py``` as an example to obtain the instance numpy array. 
2. download the kitti tracking labels from [here](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip). We provide the labels of the tracking [here](./data_tracking_label_2/)
3. obtain camera pose from a visual odometry method, such as [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). We provide the camera pose of the tracking sqeuence 0000-0020 in [here](./orbslam3_poses/). The poses are obtained by ORB-SLAM3 stereo mode. All the poses are in the real scale.
4. install the requirements: `pip install -r requirements.txt`
5. [optional] download the kitti tracking dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) for visualization.

## results visualization:
![result sequence 0000](gif/example_0000.gif)
![result sequence 0002](gif/example_0002.gif)
## usage:

To generate the mask in form of numpy array and visualize the mask at each image, run:
```
python vis.py 
```


## data structure
```
├── data_tracking_label_2
│   ├── training
│        ├── label_02
│             ├── 0000.txt
│             ├── 0001.txt
│             ...
├── mask_from_detectron
│   ├── 0000
│        ├── 000000.npy
│        ├── 000001.npy
│        ...
│   ├── 0001
│        ├── 000000.npy
│        ├── 000001.npy
│        ...
├── orbslam3_poses
│   ├── 0000
│        ├── 0000.yaml
│        ├── CameraTrajectory.txt
│   ├── 0001
│        ├── 0001.yaml
│        ├── CameraTrajectory.txt
│   ...   


