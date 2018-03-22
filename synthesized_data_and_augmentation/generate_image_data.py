# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:33:45 2018

@author: phamdinhthang
Japanese KANJI table Unicode: http://www.rikai.com/library/kanjitables/kanji_codes.unicode.shtml
"""

import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageChops
import itertools
import os
import h5py
import shutil
import random
from Image_augmentation import Image_augmentation

def padding_image(img,pad_size=0):
    new_size = (img.size[0]+pad_size*2, img.size[1]+pad_size*2)
    new_im = Image.new("L", new_size, color = (255))
    new_im.paste(img, (int((new_size[0]-img.size[0])/2), int((new_size[1]-img.size[1])/2)))
    return new_im

def get_font_name(font_path):
    file_name = os.path.split(font_path)[-1]
    return file_name.split('.')[0]

def save_image(img,unicode_hex,path):
    if not os.path.exists(path): os.mkdir(path)
    img.save(os.path.join(path, unicode_hex) + '.jpg')

def create_character_image(character, font_path, image_size, image_mode="rgb"):
    font_size = int(image_size)
    font = ImageFont.truetype(font_path,font_size)
    bg = Image.new('L',(font_size,font_size*2),'white')

    try:
        img = Image.new('L',(font_size,font_size*2),'white')
        draw = ImageDraw.Draw(img)
        draw.text((0,0), text=character, font=font)

        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            img = img.crop(bbox)
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)

            img_array = np.array(img)

            img = Image.fromarray(np.uint8(img_array))
            img = padding_image(img,int(image_size/4))
            edge_top = img_array[0, range(image_size)]
            edge_left = img_array[range(image_size), 0]
            edge_bottom = img_array[image_size - 1, range(image_size)]
            edge_right = img_array[range(image_size), image_size - 1]

            criterion = sum(itertools.chain(edge_top, edge_left, edge_bottom, edge_right))

            if criterion > 255 * image_size * 2:
                img = Image.fromarray(np.uint8(img_array))
                img = padding_image(img,int(image_size/4))
                img = img.resize((image_size, image_size), resample=Image.BILINEAR)
                if (image_mode == 'rgb' or image_mode == 'RGB'): img = img.convert('RGB')
                return img
    except:
        print("Character not supported in font")
        return None

def generate_image_file(font_path,unicode_hex_list,imgs_path,size=64,image_mode="rgb"):
    font_list =[file for file in os.listdir(font_path) if file.split('.')[-1] in ['ttf','otf','ttc']]
    print('Font list = ',font_list)
    for font in font_list:
        full_path = os.path.join(font_path,font)
        cnt=0
        for unicode_hex in unicode_hex_list:
            unicode_char = chr(int(unicode_hex, 16))
            img = create_character_image(unicode_char, full_path, size, image_mode)
            if (img != None):
                cnt += 1
                save_image(img,unicode_hex,os.path.join(imgs_path,get_font_name(full_path)))
        print('Font:',get_font_name(full_path),',',cnt,'/',len(unicode_hex_list),'character generated')
    print("Successfully generated image file. Check file at:",imgs_path)

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

def write_h5f_file(filename,file_index,dataset_path,data_arr,compress=False):
    with h5py.File(os.path.join(dataset_path,filename+str(file_index)+'.h5'), 'w') as h5f:
        if compress==True:
            h5f.create_dataset(filename+str(file_index), data=data_arr, compression='gzip',compression_opts=9)
        else:
            h5f.create_dataset(filename+str(file_index), data=data_arr)

def save_dataset(train_img, train_lbl, test_img, test_lbl, unicode_hex_arr, file_index, dataset_path, compress=False):
    train_img = np.array(train_img)
    train_lbl = np.array(train_lbl).reshape(len(train_lbl),1)
    train_lbl_one_hot = one_hot_encode(train_lbl, unicode_hex_arr)
    test_img = np.array(test_img)
    test_lbl = np.array(test_lbl).reshape(len(test_lbl),1)
    test_lbl_one_hot = one_hot_encode(test_lbl, unicode_hex_arr)

    print('Train image',file_index,'shape: ',train_img.shape)
    print('Train label',file_index,'shape: ',train_lbl.shape)
    print('Train label one hot',file_index,'shape: ',train_lbl_one_hot.shape)
    print('Test image',file_index,'shape: ',test_img.shape)
    print('Test label',file_index,'shape: ',test_lbl.shape)
    print('Test label one hot',file_index,'shape: ',test_lbl_one_hot.shape)

    write_h5f_file('train_img',file_index,dataset_path,train_img,compress)
    write_h5f_file('train_lbl_one_hot',file_index,dataset_path,train_lbl_one_hot,compress)
    write_h5f_file('test_img',file_index,dataset_path,test_img,compress)
    write_h5f_file('test_lbl_one_hot',file_index,dataset_path,test_lbl_one_hot,compress)

    np.save(os.path.join(dataset_path,'train_lbl'+str(file_index)+'.npy'),train_lbl)
    np.save(os.path.join(dataset_path,'test_lbl'+str(file_index)+'.npy'),test_lbl)
    return

def generate_dataset(font_path,unicode_hex_list,dataset_path,size=64,image_mode="rgb",augmentation_sample=0, compress=False):
    font_list =[file for file in os.listdir(font_path) if file.split('.')[-1] in ['ttf','otf','ttc']]
    print('Font list = ',font_list)

    unicode_hex_arr = np.array(unicode_hex_list).reshape(1,len(unicode_hex_list))

    if (augmentation_sample <= 0): augmentation_sample=1
    else: augmentation_sample += 1

    for i in range(augmentation_sample):
        train_img = []
        train_lbl = []
        test_img = []
        test_lbl = []
        for font in font_list:
            full_path = os.path.join(font_path,font)
            cnt=0
            for unicode_hex in unicode_hex_list:
                unicode_char = chr(int(unicode_hex, 16))
                img = create_character_image(unicode_char, full_path, size, image_mode)
                if (img != None):
                    cnt += 1
                    img_arr = np.array(img)

                    if (i != 0): img_arr = Image_augmentation.random_augmentation(img_arr)

                    if (random.random() < 0.85):
                        train_img.append(img_arr)
                        train_lbl.append(unicode_hex)
                    else:
                        test_img.append(img_arr)
                        test_lbl.append(unicode_hex)

            print('Font:',get_font_name(full_path),',',cnt,'/',len(unicode_hex_list),'character generated')
        save_dataset(train_img, train_lbl, test_img, test_lbl, unicode_hex_arr, i, dataset_path, compress)

    np.save(os.path.join(dataset_path,'lbl_list.npy'),unicode_hex_arr)
    print("Successfully generated dataset. Check dataset at:",dataset_path)

def main(char_set=['hiragana','kanji_100'], image_mode="rgb", image_size=128, augmentation=5, compress=False, font_path=None):
    src_path = os.path.abspath(os.path.dirname(__file__))
    if (font_path == None): font_path = os.path.join(src_path,'font')
    if (not os.path.exists(font_path) or not os.path.isdir(font_path) or len(os.listdir(font_path)) <= 0):
        print("Cannot find font folder. Please copy font to the specified path:",font_path)
        return

    img_path = os.path.join(src_path,'generated_image')
    if os.path.exists(img_path): shutil.rmtree(img_path)
    os.makedirs(img_path)

    dataset_path = os.path.join(src_path,'dataset')
    if os.path.exists(dataset_path): shutil.rmtree(dataset_path)
    os.makedirs(dataset_path)

    character_dict = {'hiragana':('3040','309f'),
                      'katakana':('30a0','30ff'),
                      'kanji':('4e00','9faf'),
                      'kanji_100':('4e00','5000'),
                      'kanji_1000':('4e00','5300'),
                      'kanji_5000':('4e00','61ff'),
                      'kanji_10000':('4e00','7fff')}

    tpls = [character_dict.get(i) for i in char_set]

    unicode_hex_list = []
    for tpl in tpls:
        unicode_hex_list.extend([str(hex(i))[2:] for i in range(int(tpl[0], 16),int(tpl[1], 16)+1,1)])
    print("Hex list sample: ",unicode_hex_list[:10])

    generate_image_file(font_path, unicode_hex_list, img_path, int(image_size), image_mode)
    generate_dataset(font_path, unicode_hex_list, dataset_path, int(image_size), image_mode, augmentation_sample=augmentation, compress=compress)

if __name__ == '__main__':
    main(char_set=['hiragana','katakana'], image_mode="rgb", image_size=128, augmentation=5, compress=False)
