import os
#python tools
import pyrealsense2 as rs
import numpy as np
import cv2

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

    def get_cam_param(self):
        fx, fy = self.color_intrinsics.fx, self.color_intrinsics.fy
        cx, cy = self.color_intrinsics.ppx, self.color_intrinsics.ppy
        return fx,fy,cx,cy,self.WIDTH, self.HEIGHT

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
            cv2.imshow('color_image',color_image)   
            cv2.waitKey(1)  
            return color_image,depth_image,depth_frame
        except Exception as e :
            print(e)
            color_image=None
            depth_image=None
            depth_frame=None
            return color_image,depth_image,depth_frame

    def m2mm(self,point):
        x = int(round(point[0]*1000))
        y = int(round(point[1]*1000))
        z = int(round(point[2]*1000))
        return np.array([x,y,z])

    def imgpoint_to_3dpoint(self,depth_frame,yolo_result) :
        if len(yolo_result)!=0:
            result_pos = np.empty((0,3))
            for r in yolo_result:
                u1 = round(r.u1)
                u2 = round(r.u2)
                v1 = round(r.v1)
                v2 = round(r.v2)
                u = int(round((r.u1 + r.u2) / 2))
                v = int(round((r.v1 + r.v2) / 2))
                i_d = depth_frame.get_distance(u,v) #距離推定
                point = rs.rs2_deproject_pixel_to_point(self.color_intrinsics , [u,v], i_d) #カメラ座標のx,y取得
                result_pos = np.vstack((result_pos,self.m2mm(point)))               
        else :
            result_pos = None
        return result_pos

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

