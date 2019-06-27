from torchvision import transforms

from Dataset.XRayDataset import XRayDataset
from base import BaseDataLoader
from config import CONFIG
from custom_transforms.Gaussfilter import Gaussfilter
from custom_transforms.Normalize import Normalize
from custom_transforms.Resize import Resize
from custom_transforms.ToTensor import ToTensor


class XRayDataLoader(BaseDataLoader):
    """
    XRay data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        input_rescale = (CONFIG['rescale_X_input'], CONFIG['rescale_Y_input'])
        if CONFIG['arch']['args']['dilation'] == 1:
            target_rescale = (CONFIG['rescale_X_target'], CONFIG['rescale_Y_target'])
        else:
            target_rescale = input_rescale

        trsfm = transforms.Compose([
            Gaussfilter(CONFIG['sigma']),
            Normalize(),
            Resize(
                rescale_input=input_rescale,
                rescale_target=target_rescale
            ),
            ToTensor()
        ])

        self.data_dir = data_dir
        self.dataset = XRayDataset(self.data_dir, training, transform=None)
        super(XRayDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
