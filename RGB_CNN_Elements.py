# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:48:08 2018

@author: ThangPD
"""
from abc import ABC, abstractmethod
import tensorflow as tf

class Elements(ABC):
    @abstractmethod
    def create_layer(self,input_tensor, name):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

class Inception_Block(Elements):
    """Note: for this simple implementation, only conv layer is used. One (1,1,n_filters) conv layer is used at the begining of the block to reduce depth of the block, before others stacked conv layers. For full implementation of Inception Network, please build graph from scratch"""
    
    def __init__(self, one_by_one_n_filters, conv_layer_list, strides = [1, 1, 1, 1], padding='SAME'):
        #one_by_one_n_filters: number of filters for 1x1 conv layers. Shape = (1,1,n_filters)
        #conv_layer_list: list of tuple [(filter_h,filter_w,n_filters),...]
        self.one_by_one_n_filters = one_by_one_n_filters
        self.conv_layer_list = conv_layer_list
        self.strides, self.padding = strides, padding
        
    def __str__(self):
        description = '- Inception block. 1x1 conv filter shape = (1,1,'+str(self.one_by_one_n_filters)+')\n'
        for shape in self.conv_layer_list:
            description += '\t+ Conv layer shape: ('+str(shape[0])+','+str(shape[1])+','+str(shape[2])+')\n'
        return description
    
    def create_layer(self, input_tensor, name):
        #input_tensor: tensor of shape [None, h, w, n_channels]
        conv_layer1x1, w1x1 = Convolutinal_Layer(1,1, self.one_by_one_n_filters, self.strides, self.padding).create_layer(input_tensor, name=name+'_1x1_conv')
        
        weights = [w1x1]
        layers = []
        for index, shape in enumerate(self.conv_layer_list):
            filter_h, filter_w, n_filters = shape[0], shape[1], shape[2]
            conv_layer, w = Convolutinal_Layer(filter_h, filter_w, n_filters, self.strides, self.padding).create_layer(conv_layer1x1, name=name+'_conv'+str(index))
            
            layers.append(conv_layer)
            weights.extend(w)
        
        output = tf.concat(layers, axis=3, name=name)
        return output, weights

class Residual_Block(Elements):
    def __init__(self, conv_layer_h, conv_layer_w, strides = [1, 1, 1, 1], padding='SAME'):
        self.conv_layer_h, self.conv_layer_w = conv_layer_h, conv_layer_w
        self.strides, self.padding = strides, padding
        
    def __str__(self):
        return "- Residual block. Conv filter shape = ("+str(self.conv_layer_h)+','+str(self.conv_layer_w)+')'
    
    def create_layer(self, input_tensor, name):
        #input_tensor: tensor of shape [None, h, w, n_channels]
        
        n_channels = input_tensor.get_shape().as_list()[3]
        filter_h, filter_w = self.conv_layer_h, self.conv_layer_w
        n_filters = n_channels
        
        conv1 = Convolutinal_Layer(filter_h, filter_w, n_filters, self.strides, self.padding, use_relu=True, return_bias=True)
        conv_layer1, wb1 = conv1.create_layer(input_tensor,name=name+'_conv1')
        
        conv2 = Convolutinal_Layer(filter_h, filter_w, n_filters, self.strides, self.padding, use_relu=False, return_bias=True)
        conv_layer2, wb2 = conv2.create_layer(conv_layer1,name=name+'_conv2')
        
        #Append residual input here
        conv_layer2 += input_tensor
        conv_layer2 = tf.nn.relu(conv_layer2)
        
        return conv_layer2, [wb1[0],wb1[1],wb2[0],wb2[1]]
        
class Convolutinal_Layer(Elements):
    def __init__(self, filter_h, filter_w, n_filters, strides = [1, 1, 1, 1], padding='SAME', use_relu=True, return_bias=False):
        self.filter_h, self.filter_w, self.n_filters = filter_h, filter_w, n_filters
        self.strides, self.padding = strides, padding
        self.use_relu = use_relu
        self.return_bias = return_bias
    
    def __str__(self):
        return "- Convolutional layer. Conv filter shape = ("+str(self.filter_h)+','+str(self.filter_w)+','+str(self.n_filters)+')'
    
    def create_layer(self, input_tensor, name):
        #input_tensor: tensor of shape [None, h, w, n_channels]
        n_channels = input_tensor.get_shape().as_list()[3]
        conv_filter_shape = [self.filter_h, self.filter_w, n_channels, self.n_filters]
        
        weights = tf.Variable(tf.truncated_normal(conv_filter_shape, stddev=0.03), name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([self.n_filters]), name=name+'_b')
        
        conv_layer = tf.nn.conv2d(input_tensor, weights, strides=self.strides, padding=self.padding, name=name)
        conv_layer += bias
        
        if (self.use_relu==True):
            conv_layer = tf.nn.relu(conv_layer)
        
        if (self.return_bias==False):
            return conv_layer, [weights]
        else:
            return conv_layer, [weights,bias]
        
class Pooling_Layer(Elements):
    def __init__(self,pool_h, pool_w, strides = None, pool_type='max', padding='SAME'):
        self.pool_h, self.pool_w = pool_h, pool_w
        self.strides, self.padding = strides, padding
        self.pool_type = pool_type
        
    def __str__(self):
        return "- Pooling layer. Pool type = "+self.pool_type+", pool shape = ("+str(self.pool_h)+','+str(self.pool_w)+')'
    
    def create_layer(self, input_tensor, name):
        #input tensor must of shape [None, h, w, n_channels]
        pool_size = [1, self.pool_h, self.pool_w, 1]
        if (self.strides==None): self.strides=[1, self.pool_h, self.pool_w, 1]
        pool_layer = tf.nn.max_pool(input_tensor, pool_size, strides=self.strides, padding=self.padding, name=name)
        return pool_layer, []

class Flatten_Layer(Elements):
    def __str__(self):
        return "- Flatten layer."
    
    def create_layer(self, input_tensor, name):
        #input tensor must of shape [None, h, w, n_channels]
        input_shape = input_tensor.get_shape().as_list()
        flattened_size = input_shape[1] * input_shape[2] * input_shape[3]
        flattened = tf.reshape(input_tensor, [-1, flattened_size], name = name)
        return flattened, []
    
class Dense_Layer(Elements):
    def __init__(self,n_nodes, use_relu=True):
        self.n_nodes = n_nodes
        self.use_relu = use_relu
    
    def __str__(self):
        return "- Dense layer. N_nodes = ("+str(self.n_nodes)+')'
    
    def create_layer(self, input_tensor, name, ):
        #input tensor must of shape [None,n_nodes]
        input_shape = input_tensor.get_shape().as_list()
        wd = tf.Variable(tf.truncated_normal([input_shape[1], self.n_nodes], stddev=0.03), name=name+'_W')
        bd = tf.Variable(tf.truncated_normal([self.n_nodes], stddev=0.01), name=name+'_b')
        if self.use_relu==True:
            return tf.nn.relu(tf.matmul(input_tensor, wd) + bd, name = name), [wd]
        else:
            return tf.add(tf.matmul(input_tensor, wd),bd, name=name), [wd]