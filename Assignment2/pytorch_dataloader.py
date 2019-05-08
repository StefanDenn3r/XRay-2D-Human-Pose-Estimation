import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

PATH = '/media/narvis/902310aa-b57d-49a2-841d-8199ff125fca/ATLAS/Registration/Data/Everything'


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



test_dataloader = XrayLandmarksDataset(label_file_path='data/label.txt', image_file_path = 'data/xray.png' )