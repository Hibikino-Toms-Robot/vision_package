import os
import time
import cv2 
import numpy as np
from skimage import img_as_float64
#python tools
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math
import sys
import torch

class ClearViewProcessor():
    def __init__(self):
        self.sigmas = [15, 80, 250]
        self.alpha = 125.0
        self.beta = 46.0
        self.G = 5.0
        self.OFFSET = 25.0

    def msrcr(self,img):
        """
        MSRCR (Multi-scale retinex with color restoration)

        Parameters :

        img : input image
        sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        G : final gain
        b : offset
        """
        img = img_as_float64(img)+1
        img_msr = self.multiScale(img)    
        img_color = self.crf(img)    
        img_msrcr = self.G * (img_msr*img_color + self.OFFSET)
        img_msrcr = (img_msrcr - np.min(img_msrcr, axis=(0, 1))) / (np.max(img_msrcr, axis=(0, 1)) - np.min(img_msrcr, axis=(0, 1))) * 255 #normalization　and change range to 0~255
        img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255)) #Processing to keep within 0<RGB<255
        return img_msrcr

    def singleScale(self,img,sigma):
        """
        Single-scale Retinex
        
        Parameters :

        img : input image
        sigma : the standard deviation in the X and Y directions, for Gaussian filter
        """
        #ssr = np.log10(img) - np.log10(cv2.GaussianBlur(img,(0,0),sigma))
        ssr = np.log10(img) - np.log10(cv2.blur(img, (sigma, sigma))) #高速化用
        return ssr

    def multiScale(self,img):
        """
        Multi-scale Retinex
        
        Parameters :

        img : input image
        sigma : list of all standard deviations in the X and Y directions, for Gaussian filter
        """
        retinex = np.zeros_like(img)
        for s in self.sigmas:
            retinex += self.singleScale(img,s)
        msr = retinex/len(self.sigmas)
        return msr

    def crf(self,img):
        """
        CRF (Color restoration function)

        Parameters :

        img : input image
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        """
        img_sum = np.sum(img,axis=2,keepdims=True)

        color_rest = self.beta * (np.log10(self.alpha*img) - np.log10(img_sum))
        return color_rest

    def adjust_white_balance(self,image: np.ndarray) :
        # white balance adjustment for strong neutral white
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(image[:, :, 1])
        avg_b = np.average(image[:, :, 2])
        image[:, :, 1] = image[:, :, 1] - (
            (avg_a - 128) * (image[:, :, 0] / 255.0) * 1.1
        )
        image[:, :, 2] = image[:, :, 2] - (
            (avg_b - 128) * (image[:, :, 0] / 255.0) * 1.1
        )
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        return image


class Realsense_Module():
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    def __init__(self,device = None) :
        self.setup(device)

        #camera param
        cam_pos=np.array([0,0,0])
        arm_pos=np.array([0,0,0])
        world_pos=np.array([0,0,0])
        self.cam_rotation_matrix=np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.cam_translation_vector=world_pos-cam_pos
        self.arm_rotation_matrix=np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.arm_translation_vector=arm_pos-world_pos

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

    def transformation(self,tom_pos_matrix):
        target_coordinates = np.array([])
        for tom_pos in tom_pos_matrix :
            u,v,Z = tom_pos[0],tom_pos[1],tom_pos[2]
            #img→cam
            camera_coordinate = rs.rs2_deproject_pixel_to_point(self.color_intrinsics , [u,v],Z) #カメラ座標のx,y取得
            #image→arm
            world_coordinate=self.camera_to_world(camera_coordinate,self.cam_rotation_matrix,self.cam_translation_vector)
            #target coordinates
            target_coordinate = self.world_to_arm(world_coordinate,self.arm_rotation_matrix,self.arm_translation_vector)
            target_coordinates = np.vstack((target_coordinates,target_coordinate))
        return target_coordinates

    def camera_to_world(self,camera_coordinates,rotation_matrix,translation_vector):
        world_coordinates = rotation_matrix @ camera_coordinates + translation_vector
        x,y,z = world_coordinates
        return  np.array([z,x,y])

    def world_to_arm(self,camera_coordinates,rotation_matrix,translation_vector):
        target_coordinates = rotation_matrix @ camera_coordinates + translation_vector
        return target_coordinates

# 座標変換ツール
class Transform():
    def __init__(self) :
        #焦点距離
        self.fx = 0
        self.fy = 0
        #カメラ中心座標
        self.cx = 0
        self.cy = 0
        #カメラ高さ
        self.HIGHT=0
        cam_pos=np.array([-5.5, 107.0, -4.0])
        arm_pos=np.array([64.0,-40.0,79.5])
        world_pos=np.array([0, 0, 0])
        self.cam_rotation_matrix=np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.cam_translation_vector=world_pos-cam_pos
        self.arm_rotation_matrix=np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.arm_translation_vector=arm_pos-world_pos

    def transformation(self,tom_pos_matrix):
        target_coordinates = np.array([])
        for tom_pos in tom_pos_matrix :
            u,v,Z = tom_pos[0],tom_pos[1],tom_pos[2]
            #img→cam
            camera_coordinate = self.image_to_camera(u,v,Z)
            #image→arm
            world_coordinate = self.camera_to_world(camera_coordinate,self.cam_rotation_matrix,self.cam_translation_vector)
            #target coordinates
            target_coordinate = self.world_to_arm(world_coordinate,self.arm_rotation_matrix,self.arm_translation_vector)
            target_coordinates = np.vstack((target_coordinates,target_coordinate))
        return target_coordinates

    def image_to_camera(self,u, v, Z):
            X_c = (u - self.cx) * Z / self.fx 
            Y_c = ((self.HIGHT-v) - self.cy) * Z / self.fy
            Z_c = Z
            camera_coordinates = np.array([X_c, Y_c, Z_c])
            return camera_coordinates            
    def camera_to_world(self,camera_coordinates,rotation_matrix,translation_vector):
                world_coordinates = rotation_matrix @ camera_coordinates + translation_vector
                x,y,z = world_coordinates
                return  np.array([z,x,y])

    def world_to_arm(self,camera_coordinates,rotation_matrix,translation_vector):
                target_coordinates = rotation_matrix @ camera_coordinates + translation_vector
                return target_coordinates


class Midas():  
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
