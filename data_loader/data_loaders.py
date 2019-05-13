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
            Gaussfilter(sigma),
            Normalize(),
            ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = XRayDataset(self.data_dir, is_training, transform=trsfm)
        super(XRayDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)