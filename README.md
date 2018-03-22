# Japanese handwritten OCR engine
Japanese Handwritten OCR, using Convolutional Neural Network (CNN) implemented in Tensorflow

The Japanese OCR engine is designed to detect automatically handwritten Japanese Characted, such as the Hiragana table, the Katakana table, or the Kanji table. Handwritten character must be segmentized onto a squared image, in grayscale mode (RGB mode is also accepted, but images in RGB mode are automatically converted onto Grayscale while open). Others image processing techniques such as sliding windows, erode, dillute, etc shall be used to segmentized the handwritten document (paragraph) into smaller elements (lines) and unit element (characters). 


### 1. Dataset description and format
Data used to train and validate the CNN model shall be stored as **array**, in **HDF5** format. The dataset composed of two separate set: one for *training* and one for *testing* (also one can have only one training dataset and then use train_test_split to perform the split of training and testing dataset). The filenames are described as follow:

* train_img0.h5: hdf5 format, dataset name = 'train_img0', contain image data for the block 0 
* train_lbl_one_hot0.h5: hdf format, dataset name = 'train_lbl_one_hot0', contain one hot encoded image label for the block 0
* test_img0.h5: hdf5 format, dataset name = 'test_img0', contain image data for the block 0 
* test_lbl_one_hot0.h5: numpy format, dataset name = 'test_lbl_one_hot0', contain one hot encoded image label for the block 0

The block number at the end of the filename (before file extension) is used when dataset is too large to stored in a single file (i.e more than 2GB)


