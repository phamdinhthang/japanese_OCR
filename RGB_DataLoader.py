# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:08:32 2018

@author: phamdinhthang

-----------------DATA LOADER FOR THE RGB CONVOLUTIONAL NEURAL NETWORK MODEL----------------
1. Object variable:
    - self.data_path: path to the folder contains the data for training & validation of the model. If do not specified self.data_path when init the DataLoader, it assume that data shall be stored in a folder named "dataset", under script directory.
    - self.n_classes: number of classes. Integer
    - self.lbl_list: array contain list of all labels. Array of shape (1,n_classes)
    - self.img_w: image width. Integer
    - self.img_h: image heigh. Integer
    - self.n_channels: number of channel. Integer
    - self.train_filename_list: list of tuple: [(train_img0.h5, train_lbl0.npy, train_lbl_one_hot0.npy),...]
    - self.test_filename_list: list of tuple: [(test_img0.h5, test_lbl0.npy, test_lbl_one_hot0.npy),...]

2. Data folder content:
    The data folder must contains data files for training & validation of the model, in the following format:
    - train_img0.h5: hdf5 format, dataset name = 'train_img0', contain image data for the block 0
    - train_lbl0.npy: numpy format, contain image label for the block 0
    - train_lbl_one_hot0.h5: hdf format, dataset name = 'train_lbl_one_hot0', contain one hot encoded image label for the block 0
    ... continue with index from 0 to end.

    - test_img0.h5: hdf5 format, dataset name = 'test_img0', contain image data for the block 0
    - test_lbl0.npy: numpy format, contain image label for the block 0
    - test_lbl_one_hot0.h5: numpy format, dataset name = 'test_lbl_one_hot0', contain one hot encoded image label for the block 0
    ... continue with index from 0 to end.

    - lbl_list.npy: numpy format, contain the list of label. Labels order must be identical to the one use to onehot encode the label

    NOTES: for memory efficients, each training set shall not contains more than 20000 images of size 100x100

3. Data shape and type:
    - train_img0.h5: (n_samples,image_height,image_width,n_channel), elements type: integer. n_channel usually = 3
    - train_lbl0.npy: (n_samples,1), elements type: string or integer
    - train_lbl_one_hot0.npy: (n_samples,n_classes), elements type: integer
    - lbl_list.npy: (1,n_classes), elements type: string or integer.
"""

import os
import numpy as np
import random
from matplotlib import pyplot
import h5py

class RGB_DataLoader(object):
    @staticmethod
    def show_img(img_arr):
        print('Image size =',str(img_arr.shape))
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(img_arr)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    @staticmethod
    def get_data_batch(batch_index,batch_size,img,lbl_one_hot):
        if (((batch_index+1)*batch_size) > len(img)):
            print("Batch index out of range")
            return
        start_idx = batch_index*batch_size
        end_idx = (batch_index+1)*batch_size
        return img[start_idx:end_idx,],lbl_one_hot[start_idx:end_idx,]

    def __init__(self, data_path):
        self.data_path = data_path
        self.data_is_valid = self.parse_data_info()

    def parse_data_info(self):
        try:
            self.lbl_list = np.load(os.path.join(self.data_path,'lbl_list.npy'))
            self.n_classes = self.lbl_list.shape[1]

            with h5py.File(os.path.join(self.data_path,'train_img0.h5'), 'r') as train_hf:
                train_img_h5 = train_hf.get('train_img0')
                train_img = np.array(train_img_h5)
                self.img_w = train_img.shape[2]
                self.img_h = train_img.shape[1]
                self.n_channels = train_img.shape[3]

            filenames = os.listdir(self.data_path)
            train_img_list = [f for f in filenames if f.startswith('train_img')]
            self.train_filename_list = []
            for f in train_img_list:
                idx = f[9:-3]
                train_img_filename = 'train_img'+idx+'.h5'
                train_label_filename = 'train_lbl'+idx+'.npy'
                train_label_onehot_filename = 'train_lbl_one_hot'+idx+'.h5'
                if (os.path.exists(os.path.join(self.data_path,train_img_filename)) and os.path.exists(os.path.join(self.data_path,train_label_filename)) and os.path.exists(os.path.join(self.data_path,train_label_onehot_filename))):
                    self.train_filename_list.append((train_img_filename, train_label_filename, train_label_onehot_filename))

            if (len(self.train_filename_list) <= 0):
                return False

            test_img_list = [f for f in filenames if f.startswith('test_img')]
            self.test_filename_list = []
            for f in test_img_list:
                idx = f[8:-3]
                test_img_filename = 'test_img'+idx+'.h5'
                test_label_filename = 'test_lbl'+idx+'.npy'
                test_label_onehot_filename = 'test_lbl_one_hot'+idx+'.h5'
                if (os.path.exists(os.path.join(self.data_path,test_img_filename)) and os.path.exists(os.path.join(self.data_path,test_label_filename)) and os.path.exists(os.path.join(self.data_path,test_label_onehot_filename))):
                    self.test_filename_list.append((test_img_filename, test_label_filename, test_label_onehot_filename))

            if (len(self.test_filename_list) <= 0):
                return False

            return True
        except:
            print("Error parsing data info. Please check data folder files and format")
            return False

    def get_img_lbl_onehot(self,tpl):
        img_file,lbl_file,lbl_one_hot_file = tpl[0], tpl[1], tpl[2]
        img = None
        lbl = None
        lbl_one_hot = None
        with h5py.File(os.path.join(self.data_path,img_file), 'r') as hf:
            img_h5 = hf.get(img_file.split('.')[0])
            img = np.array(img_h5)
        with h5py.File(os.path.join(self.data_path,lbl_one_hot_file), 'r') as hf:
            lbl_one_hot_h5 = hf.get(lbl_one_hot_file.split('.')[0])
            lbl_one_hot = np.array(lbl_one_hot_h5)

        lbl = np.load(os.path.join(self.data_path,lbl_file))
        return img, lbl, lbl_one_hot

    def print_data_detail(self):
        print("Number of classes = ",self.n_classes)
        print("Number of channels = ",self.n_channels)
        if self.n_classes < 20: print("List of labels = ",self.lbl_list)
        else: print("List of labels = ",self.lbl_list[0,:10],"... and",self.lbl_list.shape[1]-10," more")
        print("Image width = ",self.img_w)
        print("Image height = ",self.img_h)

    def show_example(self):
        tpl = self.train_filename_list[0]
        img, lbl, lbl_one_hot = self.get_img_lbl_onehot(tpl)
        rand_index = random.randint(0,img.shape[0]-1)
        img_arr = img[rand_index,:,:,:]
        try:
            RGB_DataLoader.show_img(img_arr)
        except:
            print("Cannot show image. Please check display device")

        print("Image label = ",lbl[rand_index,0])
        if (self.lbl_list.shape[1] < 50):
            print("Label vector = ",self.lbl_list)
            print("Image label one_hot = ",lbl_one_hot[rand_index,:])