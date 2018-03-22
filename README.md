# Japanese handwritten OCR engine
Japanese Handwritten OCR, using Convolutional Neural Network (CNN) implemented in Tensorflow

The Japanese OCR engine is designed to detect automatically handwritten Japanese Characted, such as the Hiragana table, the Katakana table, or the Kanji table. Handwritten character must be segmentized onto a squared image, in grayscale mode (RGB mode is also accepted, but images in RGB mode are automatically converted onto Grayscale while open). Others image processing techniques such as sliding windows, erode, dillute, etc shall be used to segmentized the handwritten document (paragraph) into smaller elements (lines) and unit element (characters). 


### 1. Dataset description and format
Data used to train and validate the CNN model shall be stored as **array**, in **HDF5** format. The dataset composed of two separate set: one for *training* and one for *testing* (also one can have only one training dataset and then use train_test_split to perform the split of training and testing dataset). The filenames are described as follow:

* train_img0.h5: hdf5 format, dataset name = 'train_img0', contain image data for the block 0 
* train_lbl_one_hot0.h5: hdf format, dataset name = 'train_lbl_one_hot0', contain one hot encoded image label for the block 0
* test_img0.h5: hdf5 format, dataset name = 'test_img0', contain image data for the block 0 
* test_lbl_one_hot0.h5: numpy format, dataset name = 'test_lbl_one_hot0', contain one hot encoded image label for the block 0
* lbl_list.npy: numpy format, contain the list of label. Labels order must be identical to the one use to onehot encode the label

The block number at the end of the filename (before file extension) is used when dataset is too large to stored in a single file (i.e more than 2GB)

Shape of the data array
* train_img0.h5: (n_samples, image_height, image_width, n_channel), elements type: integer. n_channel usually = 3
* train_lbl_one_hot0.npy: (n_samples, n_classes), elements type: integer
* lbl_list.npy: (1, n_classes), elements type: string or integer. 

### 2. Convolutional Neural Network Structure
To enhanced the predictive power of the CNN, various network structure have been implemented, in additional to some classical CNN layers. Within this CNN model to recognize Japanese character, there are five types of layers:
* Convolutional Layer: the main parameters are filter width, filter height and number of filters
* Pooling Layer: the main parameters are pooling width and pooling height
* Flatten Layer: no parameter
* Dense Layer: the main parameter are number of nodes.
* Residual Block: taken idea from this [paper](https://arxiv.org/pdf/1512.03385v1.pdf) by He. & Zhang. The main parameters are width and height of the inside Convolutional layer
* Inception Layer: taken idea from this [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) by Szegedy. The main parameters are size of the one by one layer,  a list of tuples for the inside Convolutional Layer: list of tuple [(filter_h,filter_w,n_filters),...]

An network can be defined as a list of sequential elements, just like in Keras:
```
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
```

### 3. Train model
Training module assume that training and testing data have been put into the `/dataset/` folder with structure defined in section 1. Training can be done using either `cpu` or `gpu`. With CPU, speed may be very low, hence simpler network structure, smaller dataset is prefered.
To train the model:

```
python train_model.py l2_beta epochs batch_size device
```
There are 4 arguments for the script:
* l2_beta: coefficient of the L2 regularization. Value can be set in range {0.1, 0.01, 0.001, etc}
* epochs: number of running epochs for the Adam Optimizer
* batch_size: Size of the mini-batch Gradient Descent
* device: either `cpu` or `gpu`

