# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:34:06 2018

@author: Home
"""

import sys

from Rectangle_Detector import Rectangle_Detector

if __name__=='__main__':
    img_path = sys.argv[1]
    detector = Rectangle_Detector()
    detector.detect_rect(img_path)