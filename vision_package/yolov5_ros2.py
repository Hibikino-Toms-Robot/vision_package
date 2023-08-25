import os
import sys
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn


#ros2_lib
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension

#yolo_lib
sys.path.append("/home/suke/hibikino_toms_ws/src/yolov5")
from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_imshow, check_requirements,non_max_suppression, print_args, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


class Yolov5(Node):
    def __init__(self):
        super().__init__('yolov5_node')

        """
        yolov5_setup
        """
        check_requirements(exclude=('tensorboard', 'thop'))
        
        # Load model       
        self.declare_parameter('device',"") 
        device = self.get_parameter('device').get_parameter_value().string_value
        self.device = select_device(device)
        
        #yolo model param
        dnn=False # use OpenCV DNN for ONNX inference
        self.declare_parameter('weights',"") 
        self.declare_parameter('data',"") 
        weights =  self.get_parameter('weights').get_parameter_value().string_value # weight path
        data =  self.get_parameter('data').get_parameter_value().string_value # dataset.yaml path
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        self.stride = self.model.stride
        self.names = self.model.names
        self.pt = self.model.pt
        self.jit = self.model.jit
        self.onnx = self.model.onnx
        self.engine = self.model.engine
        #self.declare_parameter('imgsz',640) 
        #imgsz =  self.get_parameter('imgsz').get_parameter_value().integer_value # image size
        imgsz = (640, 640)
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
  
        #Result Param
        self.declare_parameter('conf_thres',0.8) 
        self.declare_parameter('iou_thres',0.45) 
        self.conf_thres =  self.get_parameter('conf_thres').get_parameter_value().double_value  #Confidence threshold
        self.iou_thres =  self.get_parameter('iou_thres').get_parameter_value().double_value    # NMS IOU threshold

        # view param 
        self.declare_parameter('line_thickness',3) 
        self.declare_parameter('view_img',False) 
        self.declare_parameter('hide_labels',False) 
        self.declare_parameter('hide_conf',False) 
        self.line_thickness = self.get_parameter('line_thickness').get_parameter_value().integer_value  # bounding box thickness (pixels)
        self.view_img = self.get_parameter('view_img').get_parameter_value().bool_value        # show results
        self.hide_labels = self.get_parameter('hide_labels').get_parameter_value().bool_value     # hide labels
        self.hide_conf = self.get_parameter('hide_conf').get_parameter_value().bool_value       # hide confidences

        #etc... 
        self.classes = None       # filter by class: --class 0, or --class 0 2 3
        self.augment = False      # augmented inference
        self.visualize = False    # visualize features
        self.agnostic_nms = False # class-agnostic NMS
        self.max_det = 1000       # maximum detections per image
        self.half = False         # use FP16 half-precision inference

        # Half
        # FP16 supported on limited backends with CUDA
        self.half &= ((self.pt or self.jit or self.onnx or self.engine)and self.device.type != 'cpu')
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()

        # Dataloader
        self.view_img = True  # check_imshow()
        # set True to speed up constant image size inference
        cudnn.benchmark = True
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup
        

        """
        ros2_setup
        """
        self.bridge = CvBridge()
        self.result_pub_ = self.create_publisher(BoundingBoxes, "Yolov5_Result", 10)
        self.detect_img_pub_ = self.create_publisher(Image, '/Detect_Image', 10)
        self.subscriber_ = self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        

    def image_callback(self,image:Image):
        try:
            #ros→opencv format
            image_raw = self.bridge.imgmsg_to_cv2(image, 'bgr8') 
            outputs, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = self.detect(image_raw)    
            msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=image.header)
            self.result_pub_.publish(msg)
            cv2.imshow('object_detect', outputs)   
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return
        try:
            mat_image = self.bridge.cv2_to_imgmsg(outputs, encoding="bgr8")
            self.detect_img_pub_.publish(mat_image)
        except CvBridgeError as e:
            print("Failure to convert") 

        
    def yolovFive2bboxes_msgs(self, bboxes:list, scores:list, cls:list, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header
        i = 0
        for score in scores:
            one_box = BoundingBox()
            center_x = int((bboxes[0][i]+bboxes[2][i])/2)
            center_y = int((bboxes[1][i]+bboxes[3][i])/2)
            img_wight = int(bboxes[2][i]-bboxes[0][i])
            img_hight = int(bboxes[3][i]-bboxes[1][i])
            one_box.tom_pos = [center_x,center_y,img_wight,img_hight]
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1
        return bboxes_msg
    
        
        return box_result
    @torch.no_grad()
    def detect(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img[None]  # expand for batch dim
        #estimate
        pred = self.model(img, augment=self.augment, visualize=self.visualize)
        #NMS(bounding_boxの重なりを除去)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,self.agnostic_nms, max_det=self.max_det)
        det = pred[0]
        s = '%gx%g ' % img.shape[2:]  # print string
        torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(img0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # add to string
                s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
            # Write results
            class_list = []
            confidence_list = []
            x_min_list = []
            y_min_list = []
            x_max_list = []
            y_max_list = []
            for *xyxy, conf, cls in reversed(det):
                class_list.append(self.names[int(cls)])
                confidence_list.append(conf)
                x_min_list.append(xyxy[0])
                y_min_list.append(xyxy[1])
                x_max_list.append(xyxy[2])
                y_max_list.append(xyxy[3])
                if self.view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_conf else (
                            f'{self.names[c]} {conf:.2f}'))
                    annotator.box_label(xyxy, label, color=colors(c, True))

            return img0, class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list
        else:
            return img0, [], [], [], [], [], []

def main():
    rclpy.init()
    node = Yolov5()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Ctrl+Cが入力されました")  
        print("プログラム終了")  
        node.destroy_node()
        rclpy.shutdown()
    rclpy.shutdown()
