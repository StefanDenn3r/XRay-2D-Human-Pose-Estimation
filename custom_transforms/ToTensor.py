import numpy as np
import torch


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}
