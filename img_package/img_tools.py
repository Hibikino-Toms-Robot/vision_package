import os
import time
import cv2 
import numpy as np
from skimage import img_as_float64

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



class Matur_Judg():  
    def __init__(self):
        self.AverageforMatureEstimate = np.array(    
        [167.56498285, 44.98250808, 64.27371497, 126.08466818, 127.55071292, 127.79868832]
        )
        self.STDforMatureEstimate = np.array(
        [22.08992915, 23.93444563, 27.02178553, 9.09642231, 11.59547564, 12.26796652]
        )
        # 偏回帰係数と切片
        self.coef = np.array([0.0275112,-0.15477415,0.11493669,0.00621959,-0.03296917,0.0120373])
        self.intercept = 0.65064103
    def mature_judg(self,img_color,msrcr_img,bboxes):
        #収穫できるトマトだけを保存　
        box_result = np.array([])
        for i in bboxes :
            left   = bboxes[0]
            right  = bboxes[2]
            top    = bboxes[1]
            bottom = bboxes[3]
            #果実平均色の計算
            RetinexC = msrcr_img[left:right,top:bottom,:]
            RetinexCR = RetinexC[:, :, 0] 
            RetinexCG = RetinexC[:, :, 1] 
            RetinexCB = RetinexC[:, :, 2] 
            RetinexCRaverage = np.mean(RetinexCR[RetinexCR != 0])
            RetinexCGaverage = np.mean(RetinexCG[RetinexCG != 0])
            RetinexCBaverage = np.mean(RetinexCB[RetinexCB != 0])
            TargetColor=[RetinexCRaverage,RetinexCGaverage,RetinexCBaverage]

            # 周辺平均色の計算
            sBoxExpandRatio = 1.5
            sBoxCenterX = (left + right)/2
            sBoxCenterY = (top + bottom)/2
            sBoxLeft = max(1, sBoxCenterX - round(left/2*sBoxExpandRatio))
            sBoxRight = min(1080, sBoxCenterX + round(right/2*sBoxExpandRatio))
            sBoxTop = max(1, sBoxCenterY - round(top/ 2*sBoxExpandRatio))
            sBoxBottom = min(1920, sBoxCenterY + round(bottom/2*sBoxExpandRatio))
            RetinexS = msrcr_img[sBoxTop:sBoxBottom, sBoxLeft:sBoxRight, :]
            RetinexSR = RetinexS[:, :, 0] 
            RetinexSG = RetinexS[:, :, 1]
            RetinexSB = RetinexS[:, :, 2] 
            RetinexSRaverage = np.mean(RetinexSR[RetinexSR != 0])
            RetinexSGaverage = np.mean(RetinexSG[RetinexSG != 0])
            RetinexSBaverage = np.mean(RetinexSB[RetinexSB != 0])
            SurroundColor = [RetinexSRaverage, RetinexSGaverage, RetinexSBaverage]

            # 熟度推定          
            mInput = np.array(TargetColor + SurroundColor)
            mInput = (mInput - self.AverageforMatureEstimate) / self.STDforMatureEstimate
        
            #回帰モデル
            Pred = np.sum(mInput * self.coef) + self.intercept
            vEstimate = Pred * 6

            if vEstimate>3 :
                box_result = np.vstack((box_result, bboxes))
        return box_result