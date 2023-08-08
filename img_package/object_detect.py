import cv2
import json
import numpy as np
import os
import random
import cv2
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from std_msgs.msg import Int16
from cv_bridge import CvBridge, CvBridgeError
from rclpy.utilities import remove_ros_args
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import ReentrantCallbackGroup




class Segmantation_Model():
    def __init__(self, xyxy=(0.0, 0.0, 0.0, 0.0), name='', conf=0.0):
        self.setup()
    def setup(self):
        cfg = get_cfg()
        cfg.CUDA = 'cuda:0'
        register_coco_instances("tomato", {}, "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/train.json", "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/laboro_big/train/")
        self.metadata = MetadataCatalog.get("tomato")
        dataset_dicts = DatasetCatalog.get("tomato")
        cfg.MODEL.DEVICE = "cuda" #cpu or cuda
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = "/home/suke/toms_ws/instanse_seg_tools/detectron_tool/output/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(cfg)
        #detectron sample data
        # self.cfg = get_cfg()
        # self.cfg.MODEL.DEVICE = "cuda"
        # self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # self.predictor = DefaultPredictor(self.cfg)

    def detact(self,img):
        outputs = self.predictor(img)
        box_result=self.obj_box(outputs)
        bool_array = outputs["instances"]._fields["pred_masks"]
        v = Visualizer(img[:, :, ::-1],metadata=self.metadata, scale=1.0)
        outputs = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        outputs=outputs.get_image()[:, :, ::-1]
        #objects = outputs["instances"]._fields["pred_classes"].tensor.cpu().numpy():
        return outputs,box_result,bool_array
       
    def obj_box(self,outputs):
        bounding_boxes = outputs["instances"]._fields["pred_boxes"].tensor.cpu().numpy()
        box_result=[]
        for i in range(len(bounding_boxes)):
            """
            #box_arrray

              left<-------->right
            top   ----------
              ↑   |        |
              |   |        |
              |   |        |
              ↓   ----------
            bottom

            box_left   : bounding_boxes[i][0]
            box_right  : bounding_boxes[i][2]
            box_top    : bounding_boxes[i][1]
            box_bottom : bounding_boxes[i][3]
            """
            box_result.append([bounding_boxes[i][0],bounding_boxes[i][2],bounding_boxes[i][1],bounding_boxes[i][3]])
        return box_result


class Object_Detect(Node):
    def __init__(self):
        super().__init__('object_detect')
        self.seg_tools = Segmantation_Model()
        
        #param
        self.declare_parameter('image_topic',"/camera/color/image_raw")
        self.declare_parameter('depth_topic',"/camera/depth/image_rect_raw")
        self.declare_parameter('debug',False)
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        #publisher
        self.box_result_pub_ = self.create_publisher(Float32MultiArray, '/Box_Result', 10)
        if self.debug :
            self.seg_img_pub_ = self.create_publisher(Image, '/Segmentation_Image', 10)
        #subscriber
        self.callback_group = ReentrantCallbackGroup()  
        #self.sub_info = Subscriber(self, CameraInfo, 'camera/aligned_depth_to_color/camera_info',callback_group=self.callback_group)
        self.sub_color = Subscriber(self, Image, self.image_topic ,callback_group=self.callback_group)
        self.sub_depth = Subscriber(self, Image, self.depth_topic ,callback_group=self.callback_group)
        self.ts = ApproximateTimeSynchronizer([self.sub_color, self.sub_depth], 10, 0.1) #同期処理　[]内のトピックが同期して処理される
        self.ts.registerCallback(self.images_callback) #ApproximateTimeSynchronizerで登録したtopicすべてのコールバックを行う
    

    def images_callback(self, msg_color, msg_depth):
        try:
            #ros→opencv format
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8') 
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]: #size check
            self.get_logger().warn('Different image sizes for color and depth')
            return

        outputs,box_result,bool_array = self.seg_tools.detact(img_color)
        box_result=self._numpy2multiarray(np.array(box_result))
        self.box_result_pub_.publish(box_result)
        cv2.imshow('segmantation', outputs)
        cv2.waitKey(1)  
        try:
            if self.debug :
                bridge = CvBridge()
                mat_image = bridge.cv2_to_imgmsg(outputs, encoding="bgr8")
                self.seg_img_pub_.publish(mat_image)
        except CvBridgeError as e:
            print("Failure to convert") 


    def _numpy2multiarray(self, np_array):
        """Convert numpy.ndarray to multiarray"""
        multiarray = Float32MultiArray()
        layout = MultiArrayLayout()
        for i in range(np_array.ndim):  
            dim=MultiArrayDimension()
            dim.label="rows"
            dim.size=np_array.shape[i]
            dim.stride=np_array.shape[i]*np_array.dtype.itemsize
            layout.dim.append(dim)
        multiarray.layout=layout
        multiarray.data = np_array.flatten().tolist()
        return multiarray


def main(args=None):
    rclpy.init(args=args)
    node = Object_Detect()
    try :
        rclpy.spin(node) 
    except KeyboardInterrupt :
        print("Ctrl+C has been typed")  
        print("End of Program")  
    rclpy.shutdown() 


if __name__ == '__main__':
    main()
