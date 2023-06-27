import numpy as np
import os


class SeqmapLoader:
    @staticmethod
    def load_seqmap(seqmap_filename):
        print("Loading seqmap...")
        seqmap = []
        max_frames = {}
        with open(seqmap_filename, "r") as fh:
            for i, l in enumerate(fh):
                fields = l.split(" ")
                seq = "%04d" % int(fields[0])
                seqmap.append(seq)
                max_frames[seq] = int(fields[3])
        return seqmap, max_frames


class GeometryUtils:
    @staticmethod
    def roty(t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])

    @staticmethod
    def convert_3dbox_to_8corner_world(bbox3d, pose):
        ''' Takes an object's 3D box with the representation of [h,w,l, x,y,z,theta] and
            convert it to the 8 corners of the 3D box

            Returns:
                corners_3d: (8,3) array in in rect world coord
        '''
        # compute rotational matrix around yaw axis
        R = GeometryUtils.roty(bbox3d[6])

        # 3d bounding box dimensions
        l = bbox3d[2]
        w = bbox3d[1]
        h = bbox3d[0]

        # 3d bounding box corners
        x_corners = [l / 2, l / 2, -l / 2, -
                     l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2,
                     w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # rotate and translate 3d bounding box
        corners_3d = np.dot(R, np.vstack(
            [x_corners, y_corners, z_corners]))  # np.vstack([x_corners,y_corners,z_corners])
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + bbox3d[3]  # x
        corners_3d[1, :] = corners_3d[1, :] + bbox3d[4]  # y
        corners_3d[2, :] = corners_3d[2, :] + bbox3d[5]  # z

        homo_corners_3d = np.ones((4, 8))
        homo_corners_3d[:3, :] = corners_3d

        return pose@homo_corners_3d

    @staticmethod
    def Kabsch_solver(X, Y):
        """
        Solve Y = RX + t

        Input:
            X: Nx3 numpy array of N points in camera coordinate
            Y: Nx3 numpy array of N points in world coordinate
        Returns:
            R: 3x3 numpy array describing camera orientation in the world (R_wc)
            t: 1x3 numpy array describing camera translation in the world (t_wc)

        """
        # equation (2)
        cx, cy = X.sum(axis=0) / Y.shape[0], Y.mean(axis=0)

        # equation (6)
        x, y = np.subtract(X, cx), np.subtract(Y, cy)

        # equation (13)
        w = x.transpose() @ y

        # equation (14)
        u, s, vh = np.linalg.svd(w)

        # equation (20)
        ide = np.eye(3)
        ide[2][2] = np.linalg.det(vh.transpose() @ u.transpose())
        R = vh.transpose() @ ide @ u.transpose()

        # compute equation (4)
        t = cy - R @ cx

        return R, np.array([t])


class PoseLoader:
    @staticmethod
    def load_poses(pose_path):
        """Load ground truth poses (T_w_cam0) from file."""
        pose_file = os.path.join(pose_path)

        # Read and parse the poses
        poses = []
        try:
            with open(pose_file, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble for sequence.')

        return poses


class MovingObjectDetector:
    @staticmethod
    def read_tracking_label(label_file, pose_file, moving_threshold=0.1):
        """Read the tracking label file and return the dict of moving objects
        """
        # dict -> obj -> frame to get the location diff of one obj in different frame
        poses = PoseLoader.load_poses(pose_file)

        obj_dict = {}
        moving_id_dict = dict()
        obj_info_dict = dict()

        # read the KITTI tracking label file and process it line by line
        with open(label_file) as f:
            for line in f.readlines():
                line = line.split()
                this_name = line[2]
                frame_id = int(line[0])
                ob_id = int(line[1])
                # if the frame id is not in the dict, add it
                if frame_id not in moving_id_dict:
                    moving_id_dict[frame_id] = []
                if frame_id not in obj_info_dict:
                    obj_info_dict[frame_id] = dict()

                if this_name != "DontCare":
                    # get the global location of the object by relative location of the object in current frame and the pose of the current frame
                    location = np.array(line[13:16], np.float32)
                    location = poses[frame_id][:3, :3]@location + \
                        poses[frame_id][:3, 3]
                    bbox_3d = GeometryUtils.convert_3dbox_to_8corner_world(
                        np.array(line[10:17], np.float32), poses[frame_id])
                    bbox_2d = np.array(line[6:10], np.float32)

                    line = [frame_id, bbox_2d, bbox_3d, location]

                    transformation = np.eye(4)

                    # if the object is in the dict
                    if ob_id in obj_dict.keys():
                        # get the info of the last frame of the same object
                        last_frame_id = obj_dict[ob_id][-1][0]
                        last_frame_bbox_3d = obj_dict[ob_id][-1][2]
                        R, T = GeometryUtils.Kabsch_solver(
                            last_frame_bbox_3d[:3, :].T, bbox_3d[:3, :].T)

                        transformation[:3, :3] = R
                        transformation[:3, 3] = T

                        line.append(transformation)  # 4
                        # calculate the global location difference between the current frame and the last frame of the same object
                        # ignore the y axis movement
                        location_diff = [location[0]-obj_dict[ob_id][-1]
                                         [3][0], location[2]-obj_dict[ob_id][-1][3][2]]
                        line.append(np.linalg.norm(location_diff))  # 5
                        # line.append(np.linalg.norm(location-obj_dict[ob_id][-1][3])) # 5
                        obj_dict[ob_id].append(line)

                        # print info if the object is not be seen in the consecutive frame
                        if last_frame_id != frame_id-1:
                            print("last frame is {}, but this frame is {} for obj {}".format(
                                last_frame_id, frame_id, ob_id))

                    else:
                        # add new obj if it is not in the dict
                        line.append(transformation)  # 4
                        line.append(0)  # 5
                        obj_dict[ob_id] = [line]

                    obj_info_dict[frame_id][ob_id] = line

                    # add the moving id to the dict if the location diff is larger than the threshold
                    if line[5] > moving_threshold and (this_name == "Pedestrian" or this_name == "Person"):
                        moving_id_dict[frame_id].append(ob_id)
                        # handle the first frame of the moving obj
                        if len(obj_dict[ob_id]) == 2:
                            # obj_dict[ob_id][0][0] is the frame id of the first frame for ob_id
                            moving_id_dict[obj_dict[ob_id][0][0]].append(ob_id)
                    # threshold for car
                    if line[5] > (moving_threshold+0.1) and not (this_name == "Pedestrian" or this_name == "Person"):
                        moving_id_dict[frame_id].append(ob_id)
                        # handle the first frame of the moving obj
                        if len(obj_dict[ob_id]) == 2:
                            # obj_dict[ob_id][0][0] is the frame id of the first frame for ob_id
                            moving_id_dict[obj_dict[ob_id][0][0]].append(ob_id)
        return moving_id_dict, obj_info_dict

    @staticmethod
    def find_moving_id_from_tracking_label(label_file, pose_file, moving_threshold):
        """find the moving id from the KITTI tracking label file
        Args:
            label_file: the path of the label file
            pose_file: the path of the pose file
            moving_threshold: the threshold to determine whether the object is moving
        Returns:
            moving_id_dict: a dict with frame_id as key and moving id list as value
            obj_info_dict: a dict with frame_id as key and obj info dict as value
        """
        moving_id_dict, obj_info_dict = MovingObjectDetector.read_tracking_label(
            label_file, pose_file, moving_threshold)

        # count the number of frames that the moving obj appears
        moving_obj_cnt_dict = dict()
        for frame_id in moving_id_dict:
            for ob_id in moving_id_dict[frame_id]:
                if ob_id not in moving_obj_cnt_dict:
                    moving_obj_cnt_dict[ob_id] = 1
                else:
                    moving_obj_cnt_dict[ob_id] += 1

        # we only keep the moving obj that appears in more than 3 frames
        for frame_id in moving_id_dict:
            for ob_id in moving_id_dict[frame_id]:
                if moving_obj_cnt_dict[ob_id] < 4:
                    print("the {} in frame {} is removed due to frame counter".format(
                        ob_id, frame_id))
                    moving_id_dict[frame_id].remove(ob_id)

        # handle the fp in the moving obj, if the moving obj only in 1 frame for every 3 frames
        for frame_id in moving_id_dict:
            for ob_id in moving_id_dict[frame_id]:
                if frame_id-1 in moving_id_dict and frame_id+1 in moving_id_dict and ob_id not in moving_id_dict[frame_id-1] and ob_id not in moving_id_dict[frame_id+1]:
                    try:
                        print("the {} in frame {} is removed due to nearst 2 frames".format(
                            ob_id, frame_id))
                        moving_id_dict[frame_id].remove(ob_id)
                    except:
                        print("the {} in frame {} cannot be removed due to previous frame counter".format(
                            ob_id, frame_id))

        return moving_id_dict, obj_info_dict

    @staticmethod
    def find_all_moving_obj_id(label_folder, pose_folder, list_sequences, moving_threshold):
        """find all moving obj id from the tracking label and pose file
        Args:
            label_folder: the folder that contains the tracking label
            pose_folder: the folder that contains the pose file
            list_sequences: the list of sequences that we want to process
            moving_threshold: the threshold to determine whether the obj is moving or not
        Returns:
            all_obj_info_dict: the dict that contains the info of all obj in all frames
            all_moving_id: the dict that contains the moving obj id in all frames
        """
        all_moving_id = dict()
        all_obj_info_dict = dict()
        print("processing sequences: {}".format(list_sequences))
        for sequence in list_sequences:
            print("processing {}".format(sequence))
            pose_file = pose_folder + '/' + sequence + '/' + "CameraTrajectory.txt"
            label_file = label_folder + '/' + sequence + ".txt"

            moving_id, obj_info_dict = MovingObjectDetector.find_moving_id_from_tracking_label(
                label_file, pose_file, moving_threshold)
            all_moving_id[sequence] = moving_id
            all_obj_info_dict[sequence] = obj_info_dict

        return all_obj_info_dict, all_moving_id

    @staticmethod
    def generate_obj_detection(label_folder, seqmap_filename, offset):
        """generate the detection file from the tracking label file
        Args:
            label_folder: the folder that contains the tracking label
            seqmap_filename: the seqmap file that contains the sequence name
            offset: the offset of the frame id that we don't want to consider
        Returns:    
            all_detectio_dict: the detection dict
        """
        list_sequences, max_frames = SeqmapLoader.load_seqmap(seqmap_filename)
        all_detectio_dict = dict()
        for sequence in list_sequences:
            # max_frame is the number of the frames, not just the max idx
            max_frame = max_frames[sequence]+1
            label_file = label_folder + '/' + sequence + ".txt"

            detectio_dict = MovingObjectDetector.generate_obj_detection_file(
                label_file, offset, max_frame)
            all_detectio_dict[sequence] = detectio_dict

        return all_detectio_dict

    @staticmethod
    def generate_obj_detection_file(label_file, offset, max_frame):
        """generate the detection file from the tracking label file
        Args:
            label_file: the tracking label file
            offset: the offset of the frame id that we don't want to consider
            max_frame: the max frame id
        Returns:
            detectio_dict: the detection dict
        """

        curr_max_frame = max_frame-offset
        detectio_dict = dict()

        with open(label_file) as f:
            for line in f.readlines():
                line = line.split()
                frame_id = int(line[0])
                # offset 5->0 1 2 3 4 drop
                # max_frame 800->curr_max_frame 795-> 799 798 797 796 795
                if frame_id < offset or frame_id >= curr_max_frame:
                    continue
                if frame_id not in detectio_dict:
                    detectio_dict[frame_id] = []

                detectio_dict[frame_id].append(" ".join(line[2:]))

        return detectio_dict
