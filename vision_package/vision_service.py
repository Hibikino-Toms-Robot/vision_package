import rclpy  # ROS2のPythonモジュールをインポート
from rclpy.node import Node 
from toms_msg.msg import BoundingBoxes, BoundingBox
from toms_msg.srv import VisionService
from toms_msg.msg import TomatoPos, TomatoData

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

        #ros_msg
        self.yolo_result = BoundingBoxes()
        self.tomato_pos = TomatoPos()

        #subscriber
        self.create_subscription(BoundingBoxes, "Yolov5_Result",self.yolo_callback,10)
        self.create_subscription(TomatoPos,"/tomato_pos",self.harvest_order_callback,10)

        #service
        self.vision_host_server = self.create_service(VisionService,"vision_service", self.vision_host_server)
    
    def yolo_callback(self,msg):
        self.yolo_result = msg

    def harvest_order_callback(self,msg):
        self.tomato_pos = msg

    def vision_host_server(self,request, response):
        if request.task == "detect_check" :
            if self.yolo_result.bounding_boxes :
                response.detect_check = True
            else :
                response.detect_check = False
        else :
            response.target_pos = self.tomato_pos
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
