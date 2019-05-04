# -*- coding: utf-8 -*-
"""
Created on Sat May  4 11:03:08 2019

@author: felix
"""

### Imports
import cv2
from os.path import dirname, abspath, join
from random import randint, shuffle

from data_preparation import read_landmarks, gen_landmark_channels

### Constant definitions
PARENT_DIRECTORY = dirname(dirname(abspath(__file__)))
PATH_LANDMARKS = 'landmarks.txt'

debugging = True
    
if __name__ == '__main__':
    #read in landmarks
    landmark_positions = read_landmarks(PATH_LANDMARKS)
    #generate channels (Width, Height, #channels)
    channels = gen_landmark_channels(landmark_positions)
    if debugging:
        # if debugging, we plot the channel with it's respective gaussian
        for idx in range(23):
            cv2.imshow('Channel',cv2.normalize(channels[:,:,idx], None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1))
            cv2.waitKey(0)
    
    
