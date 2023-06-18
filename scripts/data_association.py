import numpy as np
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as rletools
import os
import json


# https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/utils.py
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


# for detectron
def get_moving_mask_dict_detectron(all_moving_id_dict, detectron_img_folder, all_obj_info_dict, association_threshold):
    """generate moving mask from detectron segmentation mask and moving id dict.
    Here we use hungarian algorithm to associate moving id and segmentation mask.
    """
    os.makedirs(os.path.join("moving_mask_detectron_index",), exist_ok=True)
    # reference: https://github.com/xinshuoweng/AB3DMOT/blob/6698c95606d819c03c9215b42f6f652b635a324d/AB3DMOT_libs/matching.py
    for seq_num in all_moving_id_dict.keys():
        print("start seq{}".format(seq_num))
        moving_mask_index = dict()

        os.makedirs(os.path.join(
            "moving_mask_detectron", seq_num), exist_ok=True)

        for frame_num in all_moving_id_dict[seq_num]:

            decoded_mots_mask_list = list()
            detectron_mask_path = os.path.join(
                detectron_img_folder[seq_num], f"{(frame_num):06d}.npy")
            detectron_mask = np.load(detectron_mask_path)
            for single_detectron_mask in detectron_mask:
                decoded_mots_mask_list.append(single_detectron_mask*1)

            moving_bbox_list = list()
            for moving_id in all_moving_id_dict[seq_num][frame_num]:
                moving_bbox_xy = all_obj_info_dict[seq_num][frame_num][moving_id][1]
                moving_bbox_yx = [
                    moving_bbox_xy[1], moving_bbox_xy[0], moving_bbox_xy[3], moving_bbox_xy[2]]
                moving_bbox_list.append(moving_bbox_yx)

            if len(decoded_mots_mask_list) != 0 and len(moving_bbox_list) != 0:

                decoded_mots_masks = np.stack(decoded_mots_mask_list, axis=2)
                seg_obj_2d_bboxs = extract_bboxes(decoded_mots_masks)

                all_moving_bboxs = np.stack(moving_bbox_list, axis=0)

                aff_matrix = compute_overlaps(
                    all_moving_bboxs, seg_obj_2d_bboxs)

                # hougarian algorithm
                row_ind, col_ind = linear_sum_assignment(-aff_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)

                matches_mask_ind = []
                for m in matched_indices:
                    if (aff_matrix[m[0], m[1]] < association_threshold):
                        pass
                    else:
                        matches_mask_ind.append(m[1])
                if len(matches_mask_ind) == 0:
                    moving_mask_index[frame_num] = False
                else:
                    moving_mask = sum(
                        decoded_mots_mask_list[i] for i in matches_mask_ind)
                    moving_mask_path = os.path.join(
                        "moving_mask_detectron", seq_num, f"{(frame_num):06d}")
                    np.save(moving_mask_path, moving_mask)
                    moving_mask_index[frame_num] = True

          
        with open(os.path.join("moving_mask_detectron_index", seq_num+'.json'), 'w') as f:
            json.dump(moving_mask_index, f)

    return
