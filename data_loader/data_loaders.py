import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from Dataset.XRayDataset import XRayDataset
from base import BaseDataLoader
from custom_transforms.Gaussfilter import Gaussfilter
from custom_transforms.Normalize import Normalize
from custom_transforms.ToTensor import ToTensor


class XRayDataLoader(BaseDataLoader):
    """
    XRay data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, is_training=True,
                 sigma=0.6):
        trsfm = transforms.Compose([
            Normalize(),
            Gaussfilter(sigma),
            ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = XRayDataset(self.data_dir, is_training, transform=trsfm)
        super(XRayDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def main():
    batch_size = 30
    for i_batch, sample_batched in enumerate(
            XRayDataLoader(os.path.join(Path(__file__).parent.parent, "data/XRay/Everything"), batch_size)):
        batch_images, batch_landmarks = sample_batched

        if i_batch == 0:
            print(f"batch_images shape: {batch_images.shape}; batch_landmark shape: {batch_landmarks.shape}")

        for i in range(batch_size):
            (image, landmarks) = batch_images[i].numpy(), batch_landmarks[i].numpy()
            # Display normalized Image
            # Image.fromarray(image*255).show()

            # Display Landmarks
            # Image.fromarray(np.sum(landmarks, axis=0)*255).show()

            # Display Landsmarks in normalized Image
            # stacked_image = (np.squeeze(image, axis=0) * 255 + np.sum(landmarks, axis=0) * 255)
            # Image.fromarray(cv2.normalize(stacked_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F) * 255).show()


#main()
