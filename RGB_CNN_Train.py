#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:45:13 2018

@author: phamdinhthang
References: http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/

-----------------TRAIN MODULE FOR THE RGB CONVOLUTIONAL NEURAL NETWORK MODEL----------------
1. Object variable:
    - self.data: a RGB_DataLoader object, contains valid data path, data filenames, data informations...
    - self.device_map: a dict with key = device_name, value = device_id
    - self.device_id: id of the device use to train the model, either "/cpu:0" or "/gpu:0"
    - self.network_structure: structure of the conv network to train the model
    - self.src_path: path to the source file
    - self.session_path: path to the folder contains saved session
    - self.train_log: path to the train_log.txt file to log the train process
"""

import os
import shutil
import tensorflow as tf
import time
from datetime import datetime
import pickle
from RGB_DataLoader import RGB_DataLoader
from RGB_CNN_Elements import Inception_Block, Residual_Block, Convolutinal_Layer, Pooling_Layer, Flatten_Layer, Dense_Layer

class RGB_CNN_Train(object):
    def __init__(self):
        self.data = None
        self.network_structure = None
        self.session_path = None
        self.train_log = None
        
        self.src_path = os.path.abspath(os.path.dirname(__file__))
        self.device_map = {'cpu':"/cpu:0", 'gpu':"/gpu:0"}
        self.device_id = "/cpu:0"
        
    def train_preparation(self, data_path, network_structure, device_name, session_path):
        self.data = RGB_DataLoader(data_path)
        self.data.print_data_detail()
        
        if (self.data.data_is_valid==False):
            print("Invalid data folder structure. Please check data folder structure before train")
            return False
        
        if network_structure == None:
            #Convolution layer shape = [h,w,num_filters], Pool layer shape = [h,w], Dense layer shape = [n_nodes]
            self.network_structure = [Convolutinal_Layer(3,3,16),
                                      Residual_Block(8,8),
                                      Residual_Block(8,8),
                                      Convolutinal_Layer(3,3,32),
                                      Pooling_Layer(2,2),
                                      Convolutinal_Layer(3,3,64),
                                      Pooling_Layer(2,2),
                                      Convolutinal_Layer(3,3,128),
                                      Pooling_Layer(2,2),
                                      Inception_Block(32,[(3,3,32),(5,5,16),(7,7,64)]),
                                      Flatten_Layer(),
                                      Dense_Layer(1000),
                                      Dense_Layer(1000),
                                      Dense_Layer(self.data.n_classes,use_relu=False)]
        else: self.network_structure = network_structure
        
        self.device_id = self.device_map.get(device_name,"/cpu:0")
        
        self.session_path = os.path.join(self.src_path,'saved_session') if session_path == None else session_path
        if os.path.exists(self.session_path): shutil.rmtree(self.session_path)
        os.makedirs(self.session_path)
        
        self.train_log = os.path.join(self.src_path,'train_log.txt')
        if os.path.exists(self.train_log): os.remove(self.train_log)
        
    def train(self, data_path, network_structure=None, learning_rate=0.0001, l2_beta=0.01, epochs=30, batch_size=64, device_name='cpu', session_path=None):
        if self.train_preparation(data_path, network_structure, device_name, session_path)==False: return
        
        with tf.device(self.device_id):
            tf.reset_default_graph()
            
            #Training data and labels placeholders
            x = tf.placeholder(tf.float32, [None, self.data.img_h, self.data.img_w, self.data.n_channels], name = 'x')
            y = tf.placeholder(tf.float32, [None, self.data.n_classes], name = 'y')
            
            output_layer = x
            network_weights = []
            
            #Create layers according to the network_structure
            for idx, layer in enumerate(self.network_structure):
                output_layer, weights = layer.create_layer(output_layer, name='layer'+str(idx))
                network_weights.extend(weights)
            
            #Output labels
            y_output_probs = tf.nn.softmax(output_layer, name='y_output_probs')
            y_output_labels = tf.argmax(y_output_probs, 1, name = 'y_output_labels')
            y_real_labels = tf.argmax(y, 1, name = 'y_real_labels')
            
            #Loss function with L2 regularization
            l2_weights = [tf.nn.l2_loss(weights) for weights in network_weights]
            l2_regularizer = tf.add_n([l2_weights], name = 'l2_regularizer')
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y, name = 'cross_entropy')
            loss = tf.reduce_mean(cross_entropy, name = 'loss')
            l2_loss = tf.reduce_mean(loss + l2_beta * l2_regularizer, name = 'l2_loss')
            
            #Optimization with Adam on l2_loss
            optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(l2_loss)
            
            #Accuracy assessment
            correct_prediction = tf.equal(y_real_labels, y_output_labels, name = 'correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')
        
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            def generate_log(epoch, total_epoch, train_set, total_train_set, batch, total_batch):
                return "--- "+str(datetime.now().strftime("%Y-%b-%d %H:%M:%S"))+" --- Epoch "+str(epoch+1)+"/"+str(epochs)+" --- Train set "+str(train_index+1)+"/"+str(total_train_set)+" --- Batch "+str(batch_index+1)+"/"+str(total_batch)
            
            def write_log(log_string):
                # "w" to write new file. "a" to append to current file
                log_file = open(self.train_log, "a")
                log_file.write(log_string)
                log_file.close()
        
            def print_log(log_string):
                print(log_string)
                log_string += "\n"
                write_log(log_string)
            
            sess.run(init_op)
            print_log("---------- START TRAINING. LEARNING RATE = "+str(learning_rate)+". L2 beta ="+str(l2_beta)+" EPOCHS = "+str(epochs)+". BATCH SIZE = "+str(batch_size)+" ----------")
            print_log("-------------------------Network structure-------------------------")
            for layer in self.network_structure:
                print_log(str(layer))
            
            start = time.time()
            cost_per_epoch = []
            train_accuracy_per_epoch = []
            test_accuracy_per_epoch = []
            for epoch in range(epochs):
                epoch_cost = 0
                total_train_set = len(self.data.train_filename_list)
                for train_index in range(total_train_set):
                    train_images, train_labels, train_labels_one_hot = self.data.get_img_lbl_onehot(self.data.train_filename_list[train_index])
                    total_batch = int(len(train_images) / batch_size)
                    print_log('')
                    batch_cost = 0
                    for batch_index in range(total_batch):
                        batch_x, batch_y = RGB_DataLoader.get_data_batch(batch_index,batch_size,train_images,train_labels_one_hot)
                        _, cost = sess.run([optimiser, loss], feed_dict={x: batch_x, y: batch_y})
                        batch_cost += cost / total_batch
                        
                        if (batch_index%10==0): print_log(generate_log(epoch, epochs, train_index, total_train_set, batch_index, total_batch))
                    
                    epoch_cost += batch_cost/total_train_set
                        
                #Test by mini-batch on each data set to avoid OOM with large test set
                #Validate on test set
                epoch_test_accuracy = 0
                total_test_set = len(self.data.test_filename_list)
                for test_index in range(total_test_set):
                    test_images, test_labels, test_labels_one_hot = self.data.get_img_lbl_onehot(self.data.test_filename_list[test_index])
                    total_test_batch = int(len(test_images)/batch_size)
                    batch_accuracy = 0
                    for j in range(total_test_batch):
                        images_mini, labels_mini = RGB_DataLoader.get_data_batch(j,batch_size,test_images,test_labels_one_hot)
                        accuracy_score = sess.run(accuracy, feed_dict={x: images_mini, y: labels_mini})
                        batch_accuracy += accuracy_score / total_test_batch
                    epoch_test_accuracy += batch_accuracy / total_test_set
                
                #Validate again on train set
                epoch_train_validate_accuracy = 0
                total_train_validate_set = len(self.data.train_filename_list)
                for train_validate_index in range(total_train_validate_set):
                    train_validate_images, train_validate_labels, train_validate_labels_one_hot = self.data.get_img_lbl_onehot(self.data.train_filename_list[train_validate_index])
                    total_train_validate_batch = int(len(train_validate_images)/batch_size)
                    batch_accuracy = 0
                    for j in range(total_train_validate_batch):
                        images_mini, labels_mini = RGB_DataLoader.get_data_batch(j,batch_size,train_validate_images,train_validate_labels_one_hot)
                        accuracy_score = sess.run(accuracy, feed_dict={x: images_mini, y: labels_mini})
                        batch_accuracy += accuracy_score / total_train_validate_batch
                    epoch_train_validate_accuracy += batch_accuracy / total_train_validate_set
                
                
                print_log("---------- Epoch:"+str(epoch + 1)+", Cost = {:.3f}".format(epoch_cost)+", Train accuracy: {:.3f}".format(epoch_train_validate_accuracy)+", Test accuracy: {:.3f}".format(epoch_test_accuracy)+' ----------')
                cost_per_epoch.append(epoch_cost)
                test_accuracy_per_epoch.append(epoch_test_accuracy)
                train_accuracy_per_epoch.append(epoch_train_validate_accuracy)
                
            end = time.time()
            print_log("\nTraining complete!. Total training time = " + str((end-start)/3600) + ' hours')
            print_log("Cost per epoch: "+str(cost_per_epoch))
            print_log("Train accuracy per epoch: "+str(train_accuracy_per_epoch))
            print_log("Test accuracy per epoch: "+str(test_accuracy_per_epoch))
            
            saver = tf.train.Saver(save_relative_paths=True)
            saver.save(sess, os.path.join(self.session_path,'final_model'))
            
            train_info = {'img_w':self.data.img_w,
                          'img_h':self.data.img_h,
                          'n_channels':self.data.n_channels,
                          'n_classes':self.data.n_classes,
                          'lbl_list':list(self.data.lbl_list[0])}
            
            with open(os.path.join(self.session_path,'train_info.pickle'), 'wb') as pkl:
                pickle.dump(train_info, pkl, protocol=pickle.HIGHEST_PROTOCOL)

            print_log("Successfully trained model. Check saved model at path: " + str(self.session_path))
            
        print_log("--------------------------FINISH TRAIN MODEL---------------------------------")
