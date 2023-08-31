#python tools
import PIL  
import cv2 
import numpy as np

#ros2_tools
import rclpy  
from rclpy.node import Node 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from torchvision import transforms

#my package
import sys
sys.path.append("/home/suke/hibikino_toms_ws/src/deeplabv3/")
from deeplab3 import Deeplabv3
        

class Seg_Module(Node):  
    def __init__(self):
        super().__init__('segmentaion') 
        #param
        self.declare_parameter('image_topic',"/camera/color/image_raw")
        self.declare_parameter('seg_topic',"/SegInfo")
        self.declare_parameter('debug',False)
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.seg_topic = self.get_parameter('seg_topic').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        #ros2 moduls setup
        self.bridge = CvBridge()
        self.seg_pub_ = self.create_publisher(Image,self.seg_topic,10)
        self.subscriber_ = self.create_subscription(Image,self.image_topic,self.image_callback,10)

        #segmentation tools 
        self.model = Deeplabv3.load_from_checkpoint("/home/suke/hibikino_toms_ws/src/deeplabv3/weights/DeepLabV3_resnet10110-0.26.ckpt",num_class=4)
        self.model.eval()
        img_height = 475
        img_width =  475
        self.transform = transforms.Compose([transforms.Resize((img_height, img_width)),transforms.ToTensor(),])
        
        #color_map
        self.color_map = {
            0: (0, 0, 0),     # 背景
            1: (255, 0, 0),   # クラス1
            2: (0, 255, 0),   # クラス2
            3: (0, 0, 255),   # クラス3
        }

    def image_callback(self,image:Image):
        try:
            #ros→opencv format
            image_raw = self.bridge.imgmsg_to_cv2(image, 'bgr8')
            h,w,c = image_raw.shape
            outputs = self.segmentation_process(image_raw)
            outputs = cv2.resize(outputs.astype(np.uint8),dsize=(w,h))
            outputs_rgb = self.convert_to_rgb(outputs,self.color_map)     
            cv2.imshow('object_detect', outputs_rgb)   
            cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return
        try:
            mat_image = self.bridge.cv2_to_imgmsg(outputs, encoding='mono8')
            self.seg_pub_.publish(mat_image)
        except CvBridgeError as e:
            print("Failure to convert") 

    def segmentation_process(self,image_raw):
        image_raw = cv2.cvtColor(image_raw,cv2.COLOR_BGR2RGB)
        image_raw = PIL.Image.fromarray(image_raw)
        image_raw = self.transform(image_raw)
        image_raw = image_raw.to("cuda")
        mask = self.model(image_raw.unsqueeze(0))
        mask = mask['out'].squeeze().cpu().detach().numpy()
        mask = mask.transpose((1, 2, 0))
        index_image = np.argmax(mask, axis=2)
        return index_image

    def convert_to_rgb(self,index_image, color_map):        
        rgb_values = np.take(np.array(list(color_map.values()), dtype=np.uint8), index_image, axis=0)
        rgb_image = np.reshape(rgb_values, (*index_image.shape, 3))
        return rgb_image

def main():
    rclpy.init() 
    node=Seg_Module() 
    try :
        rclpy.spin(node) 
    except KeyboardInterrupt :
        print("Ctrl+C has been typed")  
        print("End of Program")  
    rclpy.shutdown() 

if __name__ == '__main__':
    main()

