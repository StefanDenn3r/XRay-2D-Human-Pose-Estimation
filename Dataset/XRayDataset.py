import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from config import XRAY_CONFIG
import utils


class XRayDataset(Dataset):
    """X-Ray Landmarks dataset."""

    def __init__(self, root_dir, training, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied  on a sample.
        """
        self.root_dir = root_dir
        self.data_dir_paths = []

        if training:
            self.data_dir_paths += utils.retrieve_sub_folder_paths(os.path.join(self.root_dir, "Training"))
            self.data_dir_paths += utils.retrieve_sub_folder_paths(os.path.join(self.root_dir, "Validation"))
        else:
            self.data_dir_paths += utils.retrieve_sub_folder_paths(os.path.join(self.root_dir, "Test"))

        self.transform = transform

        # only use percentage_to_use of all available data. For training/validation and test
        dataset_size = len(self.data_dir_paths)
        if dataset_size <= 10:
            return
        indices = list(range(dataset_size))
        split = int(np.floor(XRAY_CONFIG['fraction_of_dataset'] * dataset_size))

        np.random.seed(42)
        np.random.shuffle(indices)

        self.data_dir_paths = np.array(self.data_dir_paths)[indices[:split]].tolist()

    def __len__(self):
        return len(self.data_dir_paths)

    def __getitem__(self, idx):
        item_dir = self.data_dir_paths[idx]

        image = cv2.imread(glob.glob(os.path.join(item_dir, "*.png"))[0], 0)

        (height, width) = image.shape

        item_landmarks = np.array(
            [np.array([int(i) for i in line.rstrip('\n').split(";")])
             for line in open(glob.glob(os.path.join(item_dir, "*.txt"))[0])]
        )

        target = np.zeros([item_landmarks.shape[0], height, width])

        for (i, (x, y)) in enumerate(item_landmarks):
            if 0 <= x < width and 0 <= y < height:
                target[i, y, x] = 1

        sample = (image, target)

        if self.transform:
            sample = self.transform(sample)

        return sample
