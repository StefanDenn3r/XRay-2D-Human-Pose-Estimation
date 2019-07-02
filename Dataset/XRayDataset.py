import glob
import os

import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import utils
from config import CONFIG
from custom_transforms.Gaussfilter import Gaussfilter
from custom_transforms.Normalize import Normalize
from custom_transforms.Resize import Resize
from custom_transforms.ToTensor import ToTensor


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
        self.items_called = 0
        self.sigma = CONFIG['sigma']
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
        split = int(np.floor(CONFIG['fraction_of_dataset'] * dataset_size))

        np.random.seed(42)
        np.random.shuffle(indices)

        self.data_dir_paths = np.array(self.data_dir_paths)[indices[:split]].tolist()

    def __len__(self):
        return len(self.data_dir_paths)

    def __getitem__(self, idx):

        self.items_called += 1
        item_dir = self.data_dir_paths[idx]

        im = Image.open(glob.glob(os.path.join(item_dir, "*.png"))[0])
        im = np.asarray(im)
        im = np.float32(im)
        maxI = np.max(im)
        minI = np.min(im)
        im = (im - minI) / (maxI - minI)
        image = np.asarray(im)
        # image = Image.open(glob.glob(os.path.join(item_dir, "*.png"))[0], 0)

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
            sample = self.get_transform()(sample)

        return sample

    def set_sigma(self):
        self.sigma = self.sigma * 0.995

    def get_transform(self):
        input_rescale = (CONFIG['rescale_X_input'], CONFIG['rescale_Y_input'])
        if CONFIG['arch']['args']['dilation'] == 1:
            target_rescale = (CONFIG['rescale_X_target'], CONFIG['rescale_Y_target'])
        else:
            target_rescale = input_rescale
        transform = transforms.Compose([
            Gaussfilter(self.sigma),
            Normalize(),
            Resize(
                rescale_input=input_rescale,
                rescale_target=target_rescale
            ),
            ToTensor()
        ])
        return transform
