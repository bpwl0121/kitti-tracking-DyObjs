import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
from detectron2.utils.logger import setup_logger

setup_logger()


MV_OBJ_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8]


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


if __name__ == '__main__':
    fulltrain = r"fulltrain.seqmap"
    seqmap, max_frames = load_seqmap(fulltrain)

    model_file_path = "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    model_weights = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    cfg = get_cfg()
    cfg.merge_from_file(model_file_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_weights
    predictor = DefaultPredictor(cfg)

    for seq_num, frame_nums in max_frames.items():
        print(seq_num)
        input_folder_path = os.path.join(
            "../data/data_tracking_image_2/training/image_02/", seq_num)

        os.makedirs(os.path.join(
            "../data/mask_from_detectron", seq_num), exist_ok=True)
        output_path = os.path.join("../data/mask_from_detectron", seq_num)
        for frame_num in range(frame_nums+1):

            img_path = os.path.join(
                "../data/data_tracking_image_2/training/image_02/", seq_num, f"{(frame_num):06d}.png")

            img = cv2.imread(img_path)
            pred = predictor(img)['instances']

            use_instance = pred.get("pred_classes") == MV_OBJ_INDICES[0]
            for i in MV_OBJ_INDICES[1:]:
                use_instance |= pred.get("pred_classes") == i
            mask = pred.get("pred_masks")[
                use_instance, :, :].cpu().detach().numpy()

            mask_path = os.path.join(output_path, f"{(frame_num):06d}")
            np.save(mask_path, mask)
