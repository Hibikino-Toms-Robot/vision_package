import rclpy  # ROS2のPythonモジュールをインポート
from rclpy.node import Node 
from toms_msg.msg import BoundingBoxes, BoundingBox
from toms_msg.srv import VisionService
#from toms_msg.msg import TomatoPos, TomatoData

import sys
sys.path.append("/home/suke/hibikino_toms_ws/src/vision_package/vision_package/modules")
from realsense_module import Realsense_Module
from seg_tools import Seg_Module
from yolo_tools import Yolov5
from harvest_order import Harvest_Order
import cv2 

"""
@autor yoshida keisuke  
-----------------------------------------
vision service node
画像処理司令を受け取って,要求された情報を返すノード

[パターン1]
受信 : task == "detect_check"
返信 : トマトが画角に写っているか

[パターン1]
受信 : task == "req_tomato_pos"
返信 : 画角に映るトマトの座標情報(収穫順番に並び変えた情報)

"""

class Vision_Service(Node):  
    def __init__(self):
        super().__init__('vision_service') 

        #service
        self.vision_host_server = self.create_service(VisionService,"vision_service", self.vision_host_server)

        #library
        self.realsense = Realsense_Module()
        self.yolo_setup()
        self.segmentation = Seg_Module()
        self.harvest_order = Harvest_Order()

        # timer
        timer_period = 0.01  #(s)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        color_img,depth_img,depth_frame = self.realsense.get_image() 
        if color_img is not None :
            yolo_result = self.yolov5.infor(color_img)  
            seg_img = self.segmentation.infor(color_img)
            if yolo_result is not None:
                result_pos =self.realsense.imgpoint_to_3dpoint(depth_frame,yolo_result)
                self.get_logger().info(f"{result_pos}")        
                target_coordinates = self.harvest_order.order_decision(seg_img,depth_img,result_pos)
                self.get_logger().info(f"{target_coordinates}")
        
        # detect = False
 

    def yolo_setup(self):
        # infor_device       
        self.declare_parameter('device',"") 
        device = self.get_parameter('device').get_parameter_value().string_value
    
        self.declare_parameter('weights',"") 
        self.declare_parameter('data',"") 
        weights =  self.get_parameter('weights').get_parameter_value().string_value # weight path
        data =  self.get_parameter('data').get_parameter_value().string_value # dataset.yaml path
        
        #Result Param
        self.declare_parameter('conf_thres',0.8) 
        self.declare_parameter('iou_thres',0.45) 
        conf_thres =  self.get_parameter('conf_thres').get_parameter_value().double_value  #Confidence threshold
        iou_thres =  self.get_parameter('iou_thres').get_parameter_value().double_value    # NMS IOU threshold

        # view param 
        self.declare_parameter('line_thickness',3) 
        self.declare_parameter('view_img',False) 
        self.declare_parameter('hide_labels',False) 
        self.declare_parameter('hide_conf',False) 
        line_thickness = self.get_parameter('line_thickness').get_parameter_value().integer_value  # bounding box thickness (pixels)
        view_img = self.get_parameter('view_img').get_parameter_value().bool_value        # show results
        hide_labels = self.get_parameter('hide_labels').get_parameter_value().bool_value     # hide labels
        hide_conf = self.get_parameter('hide_conf').get_parameter_value().bool_value       # hide confidences

        self.yolov5 = Yolov5(device,weights,data,conf_thres,iou_thres ,line_thickness,view_img,hide_labels,hide_conf)

    def detect_check(self,msg):        
        color_image,depth_image,depth_frame = self.realsense.get_image() 
        yolo_result = self.yolov5.infor(color_image)  
        return yolo_result

    def harvest_order_callback(self,msg):
        color_image,depth_image,depth_frame = self.realsense.get_image()  
        # yolo_result = self.yolov5.infor(color_image)  
        # index_image = self.segmentation.infor(color_image)
        # detect = False
        tomato_pos = 1
        return tomato_pos

    
    def vision_host_server(self,request, response):
        if request.task == "detect_check" :
            yolo_result = self.detect_check()
            if len(yolo_result)!=0:
                response.detect_check = True
            else :
                response.detect_check = False
        else :
            pass
            #response.target_pos = self.tomato_pos
        return response

def main():
    rclpy.init() 
    node=Vision_Service() 
    try :
        rclpy.spin(node)       
    except KeyboardInterrupt :
        print("Ctrl+C has been typed")  
        print("End of Program")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()