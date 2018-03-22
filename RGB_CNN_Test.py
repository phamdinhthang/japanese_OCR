# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:10:24 2018

@author: phamdinhthang

-----------------TEST MODULE FOR THE RGB CONVOLUTIONAL NEURAL NETWORK MODEL----------------
1. Object variable:
    - self.session_path: path to the folder contains saved session
    - self.meta_path: path to the ".meta" file inside saved session folder
    - self.img_w: width of the input image of the trainded model
    - self.img_h: height of the input image of the trainded model
    - self.n_channels: number of input image channels of the trainded model
    - self.lbl_list: array of shape (1,n_classes) contains all possibile classes of the model
"""

import os
import tensorflow as tf
from datetime import datetime
from PIL import Image
import numpy as np
import pickle
from RGB_DataLoader import RGB_DataLoader

class RGB_CNN_Test(object):
    def __init__(self, session_path=None):
        if (session_path == None):
            src_path = os.path.abspath(os.path.dirname(__file__))
            self.session_path = os.path.join(src_path,'saved_session')
        else:
            self.session_path = session_path

        if not os.path.exists(self.session_path):
            print("No saved session. Please train a model then save session first")

        meta_files = [f for f in os.listdir(self.session_path) if f.endswith('.meta')]
        self.meta_path = os.path.join(self.session_path,meta_files[0])

        with open(os.path.join(self.session_path,'train_info.pickle'), 'rb') as pkl:
            train_info = pickle.load(pkl)
            self.img_w = train_info.get('img_w')
            self.img_h = train_info.get('img_h')
            self.n_channels = train_info.get('n_channels')
            self.lbl_list = train_info.get('lbl_list')
            self.lbl_list = np.array(self.lbl_list).reshape((1,len(self.lbl_list)))

    def test_dataset(self,data_path,dataset='train'):
        data = RGB_DataLoader(data_path)
        if data.data_is_valid == False:
            print("Dataset folder is invalid.Please check dataset folder")
            return

        filename_list = data.train_filename_list if dataset=='train' else data.test_filename_list

        avg_accuracy = 0
        for file_index in range(len(filename_list)):
            images, _, labels_one_hot = data.get_img_lbl_onehot(filename_list[file_index])
            accuracy = self.test(images,labels_one_hot)
            avg_accuracy += accuracy/len(filename_list)
        return avg_accuracy

    def test_image(self,img_path):
        return self.test(img_path)

    def test(self,img,lbl_one_hot=None,batch_size=64):
        def read_image(img_path,w,h):
            img = Image.open(img_path)
            img = img.resize((w, h), resample=Image.BILINEAR)
            img_arr = np.array(img)
            if len(img_arr.shape) == 2:
                img_arr = grayscale_arr_to_rgb_arr(img_arr)
            return img_arr

        def grayscale_arr_to_rgb_arr(grayscale_arr):
            img = Image.fromarray(grayscale_arr)
            img_rgb = img.convert('RGB')
            return np.array(img_rgb)

        def batch_validation(img,lbl_one_hot=None,batch_size=64):
            if (len(img) <= batch_size):
                accuracy_score = sess.run(accuracy, feed_dict={x: img, y: lbl_one_hot})
                return accuracy_score
            else:
                #validation by mini-batch
                total_batch = int(len(img)/batch_size)
                avg_accuracy = 0
                for i in range(total_batch):
                    images_mini, labels_mini = RGB_DataLoader.get_data_batch(i,batch_size,img,lbl_one_hot)
                    accuracy_score = sess.run(accuracy, feed_dict={x: images_mini, y: labels_mini})
                    if (i%10==0): print("--- "+str(datetime.now().strftime("%Y-%b-%d %H:%M:%S"))+" --- Validation batch "+str(i+1)+"/"+str(total_batch)+", accuracy = "+str(accuracy_score))
                    avg_accuracy += accuracy_score
                return avg_accuracy/total_batch

        def single_validation(img):
            img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
            lbl_one_hot = np.zeros((1,self.lbl_list.shape[1]))
            output_probs = sess.run(y_output_probs, feed_dict={x: img, y: lbl_one_hot})
            output_labels_index = sess.run(y_output_labels, feed_dict={x: img, y: lbl_one_hot})
            predict_lbl = self.lbl_list[0,output_labels_index]
            return predict_lbl, output_probs

        imported_meta = tf.train.import_meta_graph(self.meta_path)
        with tf.Session() as sess:
            imported_meta.restore(sess, tf.train.latest_checkpoint(self.session_path))

            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            y = graph.get_tensor_by_name("y:0")
            y_output_probs = graph.get_tensor_by_name("y_output_probs:0")
            y_output_labels = graph.get_tensor_by_name("y_output_labels:0")
#            y_real_labels = graph.get_tensor_by_name("y_real_labels:0")
            accuracy = graph.get_tensor_by_name("accuracy:0")

            if (isinstance(img,str)):
                if (os.path.exists(img) and (img.endswith('jpg') or img.endswith('png'))):
                    #Prediction on single image
                    img_arr = read_image(img,self.img_w,self.img_h)
                    predict_lbl, output_probs = single_validation(img_arr)
                    return predict_lbl, output_probs
                elif (os.path.exists(img) and os.path.isdir(img)):
                    #Batch prediction on folder of image
                    img_files = [file for file in os.listdir(img) if file.endswith('jpg') or file.endswith('png')]
                    res = []
                    for filename in img_files:
                        img_arr = read_image(os.path.join(img,filename),self.img_w,self.img_h)
                        if (len(img_arr.shape) != self.n_channels): return None
                        predict_lbl, _ = single_validation(img_arr)
                        res.append({'filename':filename,'predicted_label':predict_lbl})
                    correct=0
                    for i in res:
                        name = i.get('filename').split('.')[0]
                        lbl = i.get('predicted_label')[0]
                        if name==lbl: correct +=1
                    accuracy = correct/len(res)
                    return res, accuracy
            elif (len(img.shape)==self.n_channels+1):
                #Batch array validation
                accuracy_score = batch_validation(img,lbl_one_hot,batch_size)
                return accuracy_score