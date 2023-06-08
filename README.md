# kitti-tracking-dynamic.objects
those scripts are used to generate dynamic objects for kitti tracking dataset

## preparation:
1. download the kitti tracking dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
2. download the kitti tracking labels from [here](http://www.cvlibs.net/download.php?file=data_tracking_label_2.zip). We provide the labels of the tracking [here](./data_tracking_label_2/)
3. obtain camera pose from a visual odometry method, such as [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3). We provide the camera pose of the tracking sqeuence 0000-0020 in [here](./orbslam3_poses/). The poses are obtained by ORB-SLAM3 stereo mode. All the poses are in the real scale.
4. install the requirements: `pip install -r requirements.txt`

## usage:

## visualize the dynamic objects
