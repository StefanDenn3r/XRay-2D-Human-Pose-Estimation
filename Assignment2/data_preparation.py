# -*- coding: utf-8 -*-
"""
Created on Sat May  4 09:16:25 2019

@author: felix
"""

import cv2
import numpy as np
from scipy import ndimage, signal
from os import listdir
from os.path import isfile, join

HEIGHT = 480
WIDTH  = 616

def read_img(path):
    # we load the image in grayscale
    return cv2.imread(path, 0)


def normalize_img(img):
    # might have to check if image is plane
    return (img - img.min()) / (img.max() - img.min())


def read_landmarks(path):
    # read in landmarks as an array of size 23x2 (x,y)
    file = open(path, 'r')
    positions = [[int(point.split(';')[0]), int(point.split(';')[1])] for point in file.readlines()]
    return positions

def gen_landmark_channels(positions):
    # get landmarks and creates channels(one channel per landmark) with a gaussian around the landmark
    channels= np.zeros((WIDTH,HEIGHT,23))
    for idx,point in enumerate(positions):
        #check if point is inside boundaries
        if (point[0] >=0 and point[0] < WIDTH and point[1]>=0 and point[1]< HEIGHT):
            channels[point[0], point[1],idx] = 1
            channels[:,:,idx]= ndimage.filters.gaussian_filter(channels[:,:,idx],10)
            channels[:,:,idx] = normalize_img(channels[:,:,idx])            
    return channels
    

def get_folders(path):
    onlyfolders = [f for f in listdir(path) if not isfile(join(path, f))]
    return onlyfolders

def get_files(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


def gkern_scipy(size, sigma):
    unit_impulse = signal.unit_impulse(size,'mid')
    return ndimage.filters.gaussian_filter(unit_impulse,sigma).reshape(30,30)



    
    
    

    
