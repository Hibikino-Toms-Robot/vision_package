import rclpy  # ROS2のPythonモジュールをインポート
from rclpy.node import Node 
from std_msgs.msg import String ,Bool
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension


from cv_bridge import CvBridge, CvBridgeError
import math
import sys
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
sys.path.append("/home/suke/toms_ws/src/image_node/image_node")
from realsense_setup import Realsense_Module
from utils import Coordinate_Transformation,Maturity_Judgment
from instanse_seg  import Segmantation_Model
from command_service.srv import DetectComand,AnalyzeComand


class Vision_Service(Node):  
    def __init__(self):
        super().__init__('vision_service') 

        
        #subscriber
        self.subscriber_ = self.create_subscription(BoundingBoxes, "Yolov5_Result",self.yolo_callback,10)
        self.subscriber_ = self.create_subscription(BoundingBoxes, "Yolov5_Result",self.yolo_callback,10)

        #service
        self.detect_srv = self.create_service(DetectComand,"tmt_detect", self.detect_check)
        self.analyze_srv = self.create_service(AnalyzeComand,"analyze_pos", self.get_target_pose) 
    
    def yolo_callback(self,msg):
        pass


    def detect_check(self,request, response):
        self.process()
        response.detect=self.tmt_detect_
        return response

    def get_target_pose(self,request, response):
        response.target_pos = self._numpy2multiarray(numpy_array)
        return response


            
def main():
    rclpy.init() 
    node=Image_Processing() 
    try :
        rclpy.spin(node) 
        
    except KeyboardInterrupt :
        print("Ctrl+Cが入力されました")  
        print("プログラム終了")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
