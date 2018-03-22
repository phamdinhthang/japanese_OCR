# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:19:42 2018

@author: phamdinhthang
"""
from PIL import Image
import numpy as np
from matplotlib import pyplot
import matplotlib as mpl
import os
import tensorflow as tf
import random

class Image_augmentation(object):
    def __init__(self):
        print('Initialization')

    @staticmethod
    def read_image(img_path):
        return np.array(Image.open(img_path))

    @staticmethod
    def show_img(img_arr):
        print('Image size = ',str(img_arr.shape))
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(img_arr) if (len(img_arr.shape)==3) else ax.imshow(img_arr, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    @staticmethod
    def grayscale_arr_to_rgb_arr(grayscale_arr):
        img = Image.fromarray(grayscale_arr)
        img_rgb = img.convert('RGB')
        return np.array(img_rgb)

    @staticmethod
    def rgb_arr_to_grayscale_arr(rgb_arr):
        img = Image.fromarray(rgb_arr)
        img_grayscale = img.convert('L')
        return np.array(img_grayscale)

    @staticmethod
    def rotate(img_arr):
        img_dimension = len(img_arr.shape)
        if (img_dimension==2): img_arr = Image_augmentation.grayscale_arr_to_rgb_arr(img_arr)
        augmented = tf.contrib.keras.preprocessing.image.random_rotation(img_arr, 20, row_axis=0, col_axis=1, channel_axis=2)
        if (img_dimension==2): augmented = Image_augmentation.rgb_arr_to_grayscale_arr(augmented)
        return augmented

    @staticmethod
    def shear(img_arr):
        img_dimension = len(img_arr.shape)
        if (img_dimension==2): img_arr = Image_augmentation.grayscale_arr_to_rgb_arr(img_arr)
        augmented = tf.contrib.keras.preprocessing.image.random_shear(img_arr, 0.3, row_axis=0, col_axis=1, channel_axis=2)
        if (img_dimension==2): augmented = Image_augmentation.rgb_arr_to_grayscale_arr(augmented)
        return augmented

    @staticmethod
    def shift(img_arr):
        img_dimension = len(img_arr.shape)
        if (img_dimension==2): img_arr = Image_augmentation.grayscale_arr_to_rgb_arr(img_arr)
        augmented = tf.contrib.keras.preprocessing.image.random_shift(img_arr, 0.15, 0.15, row_axis=0, col_axis=1, channel_axis=2)
        if (img_dimension==2): augmented = Image_augmentation.rgb_arr_to_grayscale_arr(augmented)
        return augmented

    @staticmethod
    def zoom(img_arr):
        img_dimension = len(img_arr.shape)
        if (img_dimension==2): img_arr = Image_augmentation.grayscale_arr_to_rgb_arr(img_arr)
        augmented = tf.contrib.keras.preprocessing.image.random_zoom(img_arr, (0.8, 0.8), row_axis=0, col_axis=1, channel_axis=2)
        if (img_dimension==2): augmented = Image_augmentation.rgb_arr_to_grayscale_arr(augmented)
        return augmented

    @staticmethod
    def random_augmentation(img_arr):
        a = random.randint(1,4)
        if (a==1): return Image_augmentation.rotate(img_arr)
        if (a==2): return Image_augmentation.shear(img_arr)
        if (a==3): return Image_augmentation.shift(img_arr)
        if (a==4): return Image_augmentation.zoom(img_arr)

    @staticmethod
    def batch_augmentation(img_arr,n_sample):
        res = []
        for i in range(n_sample):
            a = random.randint(1,4)
            if (a==1):
                res.append(Image_augmentation.rotate(img_arr))
                continue
            if (a==2):
                res.append(Image_augmentation.shear(img_arr))
                continue
            if (a==3):
                res.append(Image_augmentation.shift(img_arr))
                continue
            if (a==4):
                res.append(Image_augmentation.zoom(img_arr))
                continue
        return res


def augmentation_example(img_path):
    img_arr = Image_augmentation.read_image(img_path)
    print("--------------Original-------------")
    Image_augmentation.show_img(img_arr)
    print("--------------Random rotate--------")
    Image_augmentation.show_img(Image_augmentation.rotate(img_arr))
    print("--------------Random shear---------")
    Image_augmentation.show_img(Image_augmentation.shear(img_arr))
    print("--------------Random shift---------")
    Image_augmentation.show_img(Image_augmentation.shift(img_arr))
    print("--------------Random zoom----------")
    Image_augmentation.show_img(Image_augmentation.zoom(img_arr))
    print("--------------Random augmentation--")
    Image_augmentation.show_img(Image_augmentation.random_augmentation(img_arr))

def main():
    src_path = os.path.abspath(os.path.dirname(__file__))
    img_path = os.path.join(src_path,'generated_image/SourceHanSerif-Regular/305c.jpg')
    img_path = 'C:/Users/admin/Desktop/Untitled.jpg'
    augmentation_example(img_path)

if __name__ == '__main__':
    main()