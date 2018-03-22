# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:39:29 2018

@author: phamdinhthang
"""

import os
from PIL import Image
import numpy as np

def read_image(img_path):
    img = Image.open(img_path)
    img_grey = img.convert('L')
    return np.array(img_grey)

def one_hot_encode(lbl_arr,lbl_vect):
    #lbl_arr shape = (m_sample,1), lbl_vect shape = (1,n_classes)
    one_hot_encode = []
    for i in range(lbl_arr.shape[0]):
        one_hot_vect = np.zeros((1,lbl_vect.shape[1]))
        lbl_list = lbl_vect[0].tolist()
        hot_index = lbl_list.index(lbl_arr[i,0])
        one_hot_vect[0,hot_index]=1
        one_hot_encode.append(one_hot_vect)
    one_hot_encode = np.array(one_hot_encode).reshape(lbl_arr.shape[0],lbl_vect.shape[1])
    return one_hot_encode

def load_data():
    src_path = os.getcwd()
    image_path = src_path+'/generated_image'

    folder_ls = os.listdir(image_path)
    data_map = {}
    for folder in folder_ls:
        folder_path = image_path + '/' + folder
        files_ls = os.listdir(folder_path)
        for file in files_ls:
            file_name = file.split('.')[0]
            img_arr = read_image(folder_path+'/'+file)
            if data_map.get(file_name) == None: data_map[file_name] = [img_arr]
            else: data_map.get(file_name).append(img_arr)

    char_list = [key for key in data_map.keys()]
    data_map_one_hot = []
    for key,val in data_map.items():
        one_hot_vector = np.zeros((1,len(char_list)))
        hot_index = char_list.index(key)
        one_hot_vector[0,hot_index] = 1
        data_map_one_hot.append({'char':key,'label':one_hot_vector,'data':val})

    #if augmentation, do it here


    #Create data array with shape like this
    #image.shape = (1,size*size)
    #images.shape = (m_sample,size*size)
    #labels.shape = (m_sample,1)
    #labels_one_hot.shape = (m_sample,n_classes)
    image_size = 96

    images = np.zeros((1,image_size,image_size))
    labels = np.zeros((1,1))

    for a_char in data_map_one_hot:
        label = a_char.get('char')
        img_list = a_char.get('data')
        for img in img_list:
            img = img.reshape(1,img.shape[0],img.shape[1])
            images = np.vstack((images,img))
            labels = np.vstack((labels,np.array(label).reshape(1,1)))

    char_list_arr = np.array(char_list).reshape(1,len(char_list))
    images = images[1:,]
    labels = labels[1:,]
    labels_one_hot = one_hot_encode(labels,char_list_arr)

    img_w=image_size
    img_h=image_size
    n_classes=len(char_list)

    return img_w, img_h, n_classes, images, labels, labels_one_hot, char_list_arr
