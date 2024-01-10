# kepoint_inference.py
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

import numpy as np
import cv2

class Keypoint_Net():
    def __init__(self):
        register_coco_instances("fruit", {}, "/home/suke/Videos/6D_pose/dataset/config/tomato_keypoint-3.json", "/home/suke/Videos/6D_pose/dataset/image2")
        self.metadata = MetadataCatalog.get("fruit")
        dataset_dicts = DatasetCatalog.get("fruit")

        keypoint_names = ['calyx','center','bottom']
        keypoint_flip_map = []
        keypoint_connection_rules = [('calyx','center',(128,128,128)),('center','bottom',(28,28,28))]        
        MetadataCatalog.get("fruit").thing_classes = ["fruit"]          #各インスタンス/モノのカテゴリの名前のリスト
        MetadataCatalog.get("fruit").thing_dataset_id_to_contiguous_id = {3:0} #データセット内のインスタンス クラス ID から範囲 [0, #class) の連続する ID へのマッピング
        MetadataCatalog.get("fruit").keypoint_names = keypoint_names           #キーポイント検出で使用される、各キーポイントの名前のリスト
        MetadataCatalog.get("fruit").keypoint_flip_map = keypoint_flip_map     #画像が水平方向に反転される拡張中に反転される2つのキーポイント
        MetadataCatalog.get("fruit").keypoint_connection_rules = keypoint_connection_rules #接続されているキーポイントのペアと、視覚化するときにそれらの間の線に使用する色

        cfg = get_cfg()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 3
        cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((3,1),dtype=float).tolist()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.WEIGHTS = "/home/suke/Videos/6D_pose/output/model_final.pth"

        #推論
        self.predictor = DefaultPredictor(cfg)
        

    def infor(self,img):
        outputs   = self.predictor(img)
        v = Visualizer(img[:, :, ::-1],metadata=self.metadata,scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        h,w,_ = img.shape
        result_img = cv2.resize(v.get_image()[:, :, ::-1], (int(w),int(h)))
        cv2.imshow('Results',result_img)
        cv2.waitKey(1) 
        keypoints = outputs["instances"]._fields["pred_keypoints"].cpu().numpy() 
        return keypoints

    
    def postprocess(self,keypoints):
        key = np.empty((0,3))
        for keypoint in keypoints :
            point0 = keypoint[0]
            point1 = keypoint[1]
            point2 = keypoint[2]
            l = (point1[1] - point0[1]) / (point1[0] - point0[0])
            if l < -1 :
                dir = -30
            if l > 1 :
                dir = 30
            else :
                dir = 0
            new_element = np.array([point1[0],point1[1],dir])
            key = np.vstack((key,new_element))
        return key