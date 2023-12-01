#python tools
import cv2 
import numpy as np

"""
@autor yoshida keisuke  
----------------------
機能一覧
[1] 収穫順番決定
[2] 収穫可能トマト判定 
[3] トマトへのアプローチ方向(入射角)決定 
"""


# 座標変換ツール
class Transform():
    def __init__(self) :
        #カメラ高さ
        self.HIGHT = 480
        #TODO 個々取り付け位置によって変えて #将来的にはtfにして
        self.cam2arm_rotation_matrix = np.array([[1,0,0],[0,1,0], [0,0,1]])
        self.cam2arm_translation_vector = np.array([0,170,0])

    def transformation(self,tom_pos_matrix):
        target_coordinates = np.empty((0,4))
        for tom_pos in tom_pos_matrix :
            X,Y,Z,harvest_path = tom_pos[0],tom_pos[1],tom_pos[2],tom_pos[3]
            camera_coordinate = np.array([X, Y, Z])
            # 画像座標→アーム座標
            target_coordinate = self.camera_to_arm(camera_coordinate)
            target_coordinate = np.append(target_coordinate, harvest_path)
            target_coordinates = np.vstack((target_coordinates,target_coordinate))
        return target_coordinates

    def camera_to_arm(self,camera_coordinates):
        world_coordinates = self.cam2arm_rotation_matrix @ camera_coordinates + self.cam2arm_translation_vector
        x,y,z = world_coordinates
        return  np.array([x,y,z])




class Harvest_Order():  
    def __init__(self):
        #param
        self.transform_toos = Transform()
        self.threshold_distanse = 100
                       
    def order_decision(self,seg_img,depth_img,tom_pos_matrix):
        #self.get_logger().info("収穫順番決定")
        #収穫順番決定
        tom_pos_matrix = self.determine_harvest_order(tom_pos_matrix,depth_img)
        #self.get_logger().info(f"{tom_pos_matrix}")
        #収穫方向決定
        tom_pos_matrix = self.determine_harvest_path(tom_pos_matrix,seg_img)
        #self.get_logger().info(f"{tom_pos_matrix}")
        if tom_pos_matrix is not None :
            target_coordinates = self.transform_toos.transformation(tom_pos_matrix)  
            return target_coordinates
        else :
            return None

    def determine_harvest_order(self,tom_pos_matrix,depth_img):

        # step1
        sorted_column = tom_pos_matrix[:,0]
        sorted_column_indices = np.argsort(sorted_column)
        tom_pos_matrix = tom_pos_matrix[sorted_column_indices]
        
        # step2
        temp_matrix = tom_pos_matrix
        tom_pos_matrix = np.empty((0,4))
        for tom_pos in temp_matrix:
            new_element = np.array([tom_pos[0],tom_pos[1],tom_pos[2],-1])
            tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))


        # step3
        cnt = 0
        for i in range(len(tom_pos_matrix)):
            matrix1 = tom_pos_matrix[i]
            if matrix1[3] == -1:
                tom_pos_matrix[i][3] = cnt
                for j in range(len(tom_pos_matrix)) :
                    matrix2 = tom_pos_matrix[j]
                    if i != j and matrix2[3] == -1 :
                        result = self.calculate_distance(matrix1,matrix2)
                        if result :
                            tom_pos_matrix[j][3] = cnt
                        else :               
                            pass
                    else :
                        pass
                cnt +=1
            else :
                pass
        
        # step4
        sorted_indices = np.lexsort((tom_pos_matrix[:, 2], tom_pos_matrix[:, 3]))
        tom_pos_matrix = tom_pos_matrix[sorted_indices]
        return tom_pos_matrix

    def calculate_distance(self,matrix1, matrix2):
        x1, y1 = matrix1[0],matrix1[1]
        x2, y2 = matrix2[0],matrix2[1]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if distance < self.threshold_distanse:
            return True
        else :
            return False
        
    def determine_harvest_path(self,tom_pos_matrix,seg_img) :
        tmp_matrix = tom_pos_matrix
        tom_pos_matrix = np.empty((0,4))
        for tmp in tmp_matrix :
            x = tmp[0]
            y = tmp[1]
            z = tmp[2]
            approach_direction = 90
            new_element = np.array([x,y,z,approach_direction])
            tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            #セグメンテーションはとりあえずつかわない
            # if not np.any(seg_img[x_min:x_max,y_min:y_max] > 0) :
            #     new_element = np.array([x_center,y_center,tmp[4],0])
            #     tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            # elif not np.any(seg_img[x_min:x_center,y_min:y_center] > 0) :
            #     new_element = np.array([x_center,y_center,tmp[4],1])
            #     tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            # elif not np.any(seg_img[x_center:x_max,y_center:y_max] > 0) :
            #     new_element = np.array([x_center,y_center,tmp[4],2])
            #     tom_pos_matrix = np.vstack((tom_pos_matrix,new_element))
            # else : 
            #     pass
        return  tom_pos_matrix

    def tomato_pos_msg(self, target_coordinates):
        tomato_pos = TomatoPos()
        for target in target_coordinates:  
            tomato_data = TomatoData()
            tomato_data.x = int(target[0])
            tomato_data.y = int(target[1])
            tomato_data.z = int(target[2])
            tomato_data.approach_direction = int(target[3])
            tomato_pos.tomato_data.append(tomato_data)
        return tomato_pos



