#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:43:59 2018

@author: phamdinhthang
"""

import os
import sys
from RGB_CNN_Train import RGB_CNN_Train

def main(l2_beta=0.01, epochs=20, batch_size = 64, device_name = 'cpu'):
    src_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(src_path,'dataset')

    cnn_train = RGB_CNN_Train()
    cnn_train.train(data_path=data_path, l2_beta=l2_beta, epochs=epochs, batch_size=batch_size, device_name = device_name)

if __name__ == '__main__':
    l2 = float(sys.argv[1])
    epochs = int(sys.argv[2])
    batchsize = int(sys.argv[3])
    device = sys.argv[4]
    main(l2_beta=l2, epochs=epochs, batch_size = batchsize, device_name = device)

