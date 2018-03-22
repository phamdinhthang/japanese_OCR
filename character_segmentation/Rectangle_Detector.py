# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:46:41 2018

@author: Home
"""

#Note: Adjust the detector parameters to works with various image condition.

import os
import cv2
import numpy as np
import shutil

class Rectangle_Detector(object):
    @staticmethod
    def average_image(img):
        return np.average(np.average(img,axis=0))
    
    @staticmethod
    def show_img(img,name='image'):
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    @staticmethod
    def processing_rect(img):
        blurred = cv2.blur(img,(10,10))
        _,thresholded = cv2.threshold(blurred,240,255,cv2.THRESH_BINARY)
        blurred2 = cv2.blur(thresholded,(2,2))
        bordered = Rectangle_Detector.border_image(blurred2,20)
        return bordered
    
    @staticmethod
    def border_image(img,border_size=0):
        border=cv2.copyMakeBorder(img, top=border_size, bottom=border_size, left=border_size, right=border_size, borderType= cv2.BORDER_CONSTANT, value=[255,255,255] )
        return border
    
    def __init__(self):
        self.crop_path = None
    
    def detect_rect(self,img_path,crop_path=None):
        if crop_path == None:
            src_path = os.path.abspath(os.path.dirname(__file__))
            self.crop_path = os.path.join(src_path,'cropped')
        else:
            self.crop_path = crop_path
            
        if (os.path.exists(self.crop_path)): shutil.rmtree(self.crop_path)
        os.makedirs(self.crop_path)
        
        expected_w = 200
        expected_h = 200
        size_threshold = 30
        border = 25
        rects_per_line = 8
        blank_rect_threshold = 254
        
        img = cv2.imread(img_path, 0);
        Rectangle_Detector.show_img(img,'original')
        
        h, w = img.shape[:2]
        kernel = np.ones((15,15),np.uint8)
        e = cv2.erode(img,kernel,iterations = 2)  
        Rectangle_Detector.show_img(e,'eroded')
        
        d = cv2.dilate(e,kernel,iterations = 1)
        Rectangle_Detector.show_img(d,'dilated')
        
        ret, th = cv2.threshold(d, 100, 255, cv2.THRESH_BINARY_INV)
        Rectangle_Detector.show_img(th,'thresholded')
        
        
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(th, mask, (10,10), 255); # position = (200,200)
        out = cv2.bitwise_not(th)
        Rectangle_Detector.show_img(out,'flood_filled')
        
        out= cv2.dilate(out,kernel,iterations = 3)
        Rectangle_Detector.show_img(out,'dilated')
        
        _, res, h = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #res: countour in dots
        bounds = [cv2.boundingRect(i) for i in res]
        bounds.reverse()
        
        area_expected_upper = (expected_w+size_threshold)*(expected_h+size_threshold)
        area_expected_lower = (expected_w-size_threshold)*(expected_h-size_threshold)
        cnt=0
        row_index = 0
        col_index = 0
        for rect in bounds:
            x,y,w,h = rect
            area = w*h
            if(area>area_expected_lower and area<area_expected_upper):
                
                crop= img[y+border:y+h-border,x+border:x+w-border]
                crop = Rectangle_Detector.processing_rect(crop)
                if (Rectangle_Detector.average_image(crop) < blank_rect_threshold):
                    cv2.imwrite(os.path.join(self.crop_path,str(row_index)+'_'+str(col_index)+'.jpg'), crop)
                col_index += 1
                if col_index == rects_per_line:
                    row_index += 1
                    col_index = 0
                cnt += 1
        print("Found total rects:",cnt)
        print("Done detect and crop rectangles in image. Please check cropped area at:",self.crop_path)
        return self.crop_path
