#python tools
import PIL  
import cv2 
import numpy as np
from torchvision import transforms

#my package
import sys
sys.path.append("/home/hibikinotoms/hibikino_toms_ws/module/deeplabv3")
from deeplab3 import Deeplabv3
        
class Seg_Module():  
    def __init__(self):
        #segmentation tools 
        self.model = Deeplabv3.load_from_checkpoint("/home/hibikinotoms/hibikino_toms_ws/module/deeplabv3/weights/DeepLabV3_resnet10110-0.26.ckpt",num_class=4)
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

    def infor(self,image_raw):
        h,w,c = image_raw.shape
        index_image = self.segmentation_process(image_raw)
        index_image = cv2.resize(index_image.astype(np.uint8),dsize=(w,h))
        outputs_rgb = self.convert_to_rgb(index_image,self.color_map)     
        cv2.imshow('segmentaion_result', outputs_rgb)   
        cv2.waitKey(1)
        return index_image

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


