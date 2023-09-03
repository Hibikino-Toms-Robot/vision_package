import rclpy  # ROS2のPythonモジュールをインポート
from rclpy.node import Node 
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
from toms_msg.msg import BoundingBoxes, BoundingBox


class Vision_Service(Node):  
    def __init__(self):
        super().__init__('vision_service') 

        #
        self.harvest_order = Float32MultiArray()
        self.yolo_result = BoundingBoxes()

        #subscriber
        self.subscriber_ = self.create_subscription(BoundingBoxes, "Yolov5_Result",self.yolo_callback,10)
        self.subscriber_ = self.create_subscription(Float32MultiArray,"/harvest_order",self.harvest_order_callback,10)

        #service
        self.detect_srv = self.create_service(DetectComand,"tmt_detect", self.detect_check)
        self.analyze_srv = self.create_service(AnalyzeComand,"analyze_pos", self.get_target_pose) 
    
    def yolo_callback(self,msg):
        self.yolo_result = msg


    def harvest_order_callback(self,msg):
        self.harvest_order = msg

    # def detect_check(self,request, response):
    #     response.detect=self.tmt_detect_
    #     return response

    # def get_target_pose(self,request, response):
    #     response.target_pos = self._numpy2multiarray(numpy_array)
    #     return response


            
def main():
    rclpy.init() 
    node=Vision_Service() 
    try :
        rclpy.spin(node) 
        
    except KeyboardInterrupt :
        print("Ctrl+Cが入力されました")  
        print("プログラム終了")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
