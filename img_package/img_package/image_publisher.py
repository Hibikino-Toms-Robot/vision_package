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


#my package
import sys
sys.path.append("/home/suke/hibikino_toms_ws/src/img_package/img_package/")
from img_tools import ClearViewProcessor

class Midas(Node):  
    def __init__(self):
        model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
    
    def estmate(self,color_image):
        rgb_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb_img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=color_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_image = prediction.cpu().numpy()
        return depth_image

    def normalize_depth(self,depth):
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*2))-1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.type)
        return out.astype("uint16")



class Realsense_Module():
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    def __init__(self,device = None) :
        self.setup(device)

    def setup(self,device=None):
        self.conf = rs.config()
        if device is not None:
            self.conf.enable_device(device)
        self.conf.enable_stream(rs.stream.color, self.WIDTH, self.HEIGHT, rs.format.bgr8, self.FPS)
        self.conf.enable_stream(rs.stream.depth, self.WIDTH, self.HEIGHT, rs.format.z16, self.FPS)
        #start_stream
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(self.conf)
        #Align_objetc
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        #get_camera_param
        self.depth_intrinsics = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth)).get_intrinsics()
        self.color_intrinsics = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics()

    def get_image(self) :
        try :
            #waiting for a frame
            frames = self.pipe.wait_for_frames()
            #get_frame_data
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = self.depth_filter(depth_frame)
            if not depth_frame or not color_frame:
                return
            #conversion unit16⇨numpy
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image,depth_image,
        except Exception as e :
            print(e)
            color_image=None
            depth_image=None
            return color_image,depth_image

    def depth_filter(self,depth_frame):
        #TODO recursive median filterを入れる
        # decimarion_filter param
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 1)
        # spatial_filter param
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 1)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
        spatial.set_option(rs.option.filter_smooth_delta, 50)
        # hole_filling_filter param
        hole_filling = rs.hole_filling_filter()
        # disparity
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        # filter
        filter_frame = decimate.process(depth_frame)
        filter_frame = depth_to_disparity.process(filter_frame)
        filter_frame = spatial.process(filter_frame)
        filter_frame = disparity_to_depth.process(filter_frame)
        filter_frame = hole_filling.process(filter_frame)
        result_frame = filter_frame.as_depth_frame()
        return result_frame

class Image_Publisher(Node):  
    def __init__(self):
        super().__init__('image_publisher') 
        self.filter = ClearViewProcessor()
        #param
        self.declare_parameter('cam_num',1)
        self.declare_parameter('cam_mode',"")
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
                self.realsense = Realsense_Module()
            else :
                self.realsense = Realsense_Module()
        else : 
            self.midas = Midas()
            self.cam = cv2.VideoCapture(-1)  
            
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

