#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:46:15 2018

@author: phamdinhthang
"""

import sys
import os
from PIL import Image
import numpy as np
from RGB_CNN_Test import RGB_CNN_Test

def read_image(img_path):
    img = Image.open(img_path)
    return np.array(img)

def training_error():
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(src_path,'dataset')
    if not os.path.exists(data_path):
        print("Data folder not available. Validation failed")
        return None
    
    cnn_test = RGB_CNN_Test()
    accuracy = cnn_test.test_dataset(data_path=data_path,dataset='train')
    print("----- Train accuracy = ",accuracy)

def testing_error():
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(src_path,'dataset')
    if not os.path.exists(data_path):
        print("Data folder not available. Validation failed")
        return None
    
    cnn_test = RGB_CNN_Test()
    accuracy = cnn_test.test_dataset(data_path=data_path,dataset='test')
    print("----- Test accuracy = ",accuracy)
    
def image_test(img_path):
    if not os.path.exists(img_path):
        print("Image not available. Validation failed")
        return None
    
    cnn_test = RGB_CNN_Test()
    predic_lbl, _ = cnn_test.test_image(img_path)
    print("----- Predicted label = ",predic_lbl)
    print("----- File name: ",img_path.split(os.sep)[-1])     

def image_folder_test(folder_path):
    cnn_test = RGB_CNN_Test()
    res, accuracy = cnn_test.test_image(folder_path)
    for i in res:
        print("----- Filename =",i.get("filename"),", predicted label =",i.get("predicted_label"))
    print("Accuracy =",accuracy)

def main(run_param):
    if (run_param == 'train'):
        training_error()
        return
    if (run_param == 'test'):
        testing_error()
        return
    if (run_param.endswith('jpg') or run_param.endswith('png')):
        image_test(run_param)
        return
    if (os.path.exists(run_param) and os.path.isdir(run_param)):
        image_folder_test(run_param)
        return
    
    print("Invalid run parameter. Please try again")

if __name__ == '__main__':
    #One param of value:
    # - "train"
    # - "test"
    # - image path ends with .jpg or .png
    # - image folder contains images of type .jpg or .png
    main(sys.argv[1])