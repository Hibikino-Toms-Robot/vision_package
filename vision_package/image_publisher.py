#python tools
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import sys
import torch

#ros2_tools
import rclpy  
from rclpy.node import Node 
from sensor_msgs.msg import Image
from std_msgs.msg import Int64
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo

#my package
import sys
sys.path.append("/home/suke/hibikino_toms_ws/src/vision_package/vision_package")
from img_tools import ClearViewProcessor,Realsense_Module,Midas


class Image_Publisher(Node):  
    def __init__(self):
        super().__init__('image_publisher') 
        self.filter = ClearViewProcessor()
        #param
        self.declare_parameter('cam_num',1)
        self.declare_parameter('cam_mode',"realsense")
        self.declare_parameter('debug',False)
        self.cam_num = self.get_parameter('cam_num').get_parameter_value().integer_value
        self.cam_mode = self.get_parameter('cam_mode').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        #ros2 moduls setup
        self.bridge = CvBridge()
        timer_period = 0.01  #(s)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.color_pub_ = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub_ = self.create_publisher(Image, '/camera/depth/image_rect_raw', 10)
        
        
        if self.cam_mode == "realsense":
            if self.cam_num == 2 :
                self.device = "218622271154"
                self.realsense = Realsense_Module()
                fx,fy,cx,cy,width,height = self.realsense.get_cam_param()
            else :
                self.realsense = Realsense_Module()
                fx,fy,cx,cy,width,height = self.realsense.get_cam_param()
        else : 
            self.midas = Midas()
            self.cam = cv2.VideoCapture(-1) 

        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.header.frame_id = 'camera_frame'
        self.camera_info_msg.width = width
        self.camera_info_msg.height = height
        self.camera_info_msg.distortion_model = 'plumb_bob'
        self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # 歪み係数
        self.camera_info_msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]  # カメラ行列
        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # 回転行列(世界座標系⇨カメラ座標系)
        self.camera_info_msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]  # 投影行列
        self.camera_info_pub_ = self.create_publisher(CameraInfo, 'camera_info_topic', 10)  


    def timer_callback(self):
        if self.cam_mode == "realsense":
            color_image,depth_image = self.realsense.get_image()  
        else : 
            ret, color_image = self.cam.read()
            if not ret:
                return None,None
            depth_image = self.midas.estmate(color_image)
            depth_image = self.midas.normalize_depth(depth_image)       
        try:
            #filter
            color_image = self.filter.adjust_white_balance(color_image)
            #publish
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='mono16')
            self.color_pub_.publish(color_msg)
            self.depth_pub_.publish(depth_msg)
            cv2.imshow('color', color_image)
            cv2.imshow('depth', depth_image)
            cv2.waitKey(1)  
        except CvBridgeError as e:
            print("Failure to convert") 
  
        self.camera_info_pub_.publish(self.camera_info_msg)

    def limit_area(self,color_image,depth_image,left=0,right=600,top=0,bottom=500):
        lim_colorimage=color_image[left:right,top:bottom,:]
        lim_depth_image=depth_image[left:right,top:bottom]
        return lim_colorimage,lim_depth_image


def main():
    rclpy.init() 
    node=Image_Publisher() 
    try :
        rclpy.spin(node) 
    except KeyboardInterrupt :
        print("Ctrl+C has been typed")  
        print("End of Program")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()

