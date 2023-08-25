#python tools
import PIL  
import cv2 
import numpy as np

#ros2_tools
import rclpy  
from rclpy.node import Node 
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge, CvBridgeError

# ros2 message_filters(subscriber)
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import ReentrantCallbackGroup

import sys
sys.path.append("/home/suke/hibikino_toms_ws/src/img_package/img_package/")
from img_tools import Realsense_Module,Transform

class Harvest_Order(Node):  
    def __init__(self):
        super().__init__('harvest_order') 
        #param
        self.declare_parameter('seg_topic',"/SegInfo")
        self.declare_parameter('debug',False)
        self.declare_parameter('cam_mode',"")
        self.seg_topic = self.get_parameter('seg_topic').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.cam_mode = self.get_parameter('cam_mode').get_parameter_value().string_value

        #ros2 moduls setup
        self.bridge = CvBridge()
        #self.seg_pub_ = self.create_publisher(Image,self.seg_topic,10)
        
        #subscriber group
        self.callback_group = ReentrantCallbackGroup() 
        self.seg_sub = Subscriber(self, Image,self.seg_topic ,callback_group=self.callback_group)
        self.depth_sub = Subscriber(self, Image,'/camera/depth/image_rect_raw',callback_group=self.callback_group)
        self.yolo_sub = Subscriber(self, BoundingBoxes, "Yolov5_Result",callback_group=self.callback_group)
        self.ts = ApproximateTimeSynchronizer([self.seg_sub,self.depth_sub,self.yolo_sub], 10, 0.1) 
        self.ts.registerCallback(self.callback)
        
    def callback(self,seg_msg,depth_msg,yolo_msg):
        try:
            #ros→opencv format
            seg_img = self.bridge.imgmsg_to_cv2(seg_msg,'passthrough')  
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg,'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        bounding_boxes = yolo_msg.bounding_boxes
        tom_pos_list = [box.tom_pos for box in bounding_boxes]
        tom_pos_matrix = np.array(tom_pos_list)
        
        if not bounding_boxes:
            #self.get_logger().info("検出数0")
            pass
        else :
            #self.get_logger().info("収穫順番決定")
            #収穫順番決定
            tom_pos_matrix = self.determine_harvest_order(tom_pos_matrix,depth_img)
            self.get_logger().info(f"{tom_pos_matrix}")
            #収穫方向決定
            tom_pos_matrix = self.determine_harvest_path(tom_pos_matrix,seg_img)
            self.get_logger().info(f"{tom_pos_matrix}")
            if tom_pos_matrix.size > 0 :
                #座標変換
                if self.cam_mode == "realsense" :
                    realsense_tool = Realsense_Module()
                    target_coordinates = realsense_tool.transformation(tom_pos_matrix)
                else :
                    transform_toos = Transform()
                    target_coordinates = transform_toos.transformation(tom_pos_matrix)  
                self.get_logger().info(f"{target_coordinates.shape}")      
            self.get_logger().info(f"---------------------")



    def determine_harvest_order(self,tom_pos_matrix,depth_img):
        """
        tom_pose_matrix
        [トマトの検出数の行列*6]
        [i][0] : center_x
        [i][1] : center_y
        [i][2] : bbox_wide
        [i][3] : bbox_hight
        [i][4] : depth
        [i][5] : group
        """ 
        """
        step 1 
        画像中の左側にあるトマトから収穫に並び替える
        [x_min,y_min,x_max,y_max] ←x_minの値をみて小さい順に並び替える

        step 2 
        depth 追加
        [x_min,y_min,x_max,y_max,depth]

        step 3 距離でグループ分け
        
        """
        # step1
        sorted_column = tom_pos_matrix[:,0]
        sorted_column_indices = np.argsort(sorted_column)
        tom_pos_matrix = tom_pos_matrix[sorted_column_indices]
        #debag
        # self.get_logger().info(f"{sorted_column}")
        # self.get_logger().info(f"{sorted_column_indices}")
        # self.get_logger().info(f"{tom_pos_matrix}")
        #self.get_logger().info(f"{tom_pos_matrix.shape}")
        
        # step2
        temp_matrix = tom_pos_matrix
        tom_pos_matrix = np.empty((0,6))
        for i in range(len(temp_matrix)):
            center_y = int((temp_matrix[i][1]+temp_matrix[i][3])/2)
            center_x = int((temp_matrix[i][0]+temp_matrix[i][2])/2)
            depth = depth_img[center_y,center_x]
            new_element = np.array([temp_matrix[i][0],temp_matrix[i][1],temp_matrix[i][2],temp_matrix[i][3],depth,-1])
            tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))

        # step3
        cnt = 0
        for i in range(len(tom_pos_matrix)):
            matrix1 = tom_pos_matrix[i]
            if matrix1[5] == -1:
                tom_pos_matrix[i][5] = cnt
                for j in range(len(tom_pos_matrix)) :
                    matrix2 = tom_pos_matrix[j]
                    if i != j and matrix2[5] == -1 :
                        result = self.calculate_distance(matrix1,matrix2)
                        if result :
                            tom_pos_matrix[j][5] = cnt
                        else :               
                            pass
                    else :
                        pass
                cnt +=1
            else :
                pass
        
        # step4
        sorted_indices = np.lexsort((tom_pos_matrix[:, 4], tom_pos_matrix[:, 5]))
        tom_pos_matrix = tom_pos_matrix[sorted_indices]
        
        return tom_pos_matrix

    def calculate_distance(self,matrix1, matrix2):
        x1, y1 = matrix1[0],matrix1[1]
        x2, y2 = matrix2[0],matrix2[1]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance >10:
            flag = True
        else :
            flag = False
        return flag
        
    def determine_harvest_path(self,tom_pos_matrix,seg_img) :
        tmp_matrix = tom_pos_matrix
        tom_pos_matrix = np.empty((0,4))
        for tmp in tmp_matrix :
            x_min = int(tmp[0]-tmp[2]/2)
            x_center = tmp[0]
            x_max = int(tmp[0]+tmp[2]/2)
            y_min = int(tmp[1]-tmp[3]/2)
            y_center = tmp[1]
            y_max = int(tmp[1]+tmp[3]/2)
            if not np.any(seg_img[x_min:x_max,y_min:y_max] > 0) :
                new_element = np.array([x_center,y_center,tmp[4],0])
                tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            elif not np.any(seg_img[x_min:x_center,y_min:y_center] > 0) :
                new_element = np.array([x_center,y_center,tmp[4],1])
                tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            elif not np.any(seg_img[x_center:x_max,y_center:y_max] > 0) :
                new_element = np.array([x_center,y_center,tmp[4],2])
                tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            else : 
                pass
        
        return  tom_pos_matrix

   

def main():
    rclpy.init() 
    node=Harvest_Order() 
    try :
        rclpy.spin(node) 
    except KeyboardInterrupt :
        print("Ctrl+C has been typed")  
        print("End of Program")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()

