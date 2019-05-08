# -*- coding: utf-8 -*-
"""

@author: anum
"""
import os, sys
import torch
#import pandas as pd
from torch.utils.data import Dataset, DataLoader

PATH_TESTING = '/media/narvis/902310aa-b57d-49a2-841d-8199ff125fca/ATLAS/Registration/Data/Everything/Testing/ABD_LYMPH_004'
PATH_TRAINING = '/media/narvis/902310aa-b57d-49a2-841d-8199ff125fca/ATLAS/Registration/Data/Everything/Training/'
PATH_VALIDATION = '/media/narvis/902310aa-b57d-49a2-841d-8199ff125fca/ATLAS/Registration/Data/Everything/Validation/ABD_LYMPH_080/'


def read_data():

    # The directory for testing data is empty so we are not gonna do anything with it
    #dirs_testing = os.listdir(PATH_TESTING)

    dirs_training = []
    dirs_temp = os.listdir(PATH_TRAINING)
    for d in dirs_temp:
        dirs_training += list(os.listdir(d))

    print (dirs_training.shape)

    dirs_validation = os.listdir(PATH_VALIDATION)
    print(dirs_validation.shape)

    return dirs_training, dirs_validation


'''
class XrayLandmarksDataset(Dataset):

    def __init__(self, label_file_path, image_file_path):
        """
        Args:
            label_file_path (string): Path to the csv file with landmark annotations.
            image_file_path (string): Path to the image file..

        """

        self.landmarks_frame = pd.read_csv(label_file_path)
        self.image_file_path = image_file_path
        self.transform()



    def transform(self):
        pass

'''
dirs_training, dirs_validation = read_data()

#test_dataloader = XrayLandmarksDataset(label_file_path='data/label.txt', image_file_path = 'data/xray.png' )