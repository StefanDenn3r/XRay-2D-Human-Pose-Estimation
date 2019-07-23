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

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, custom_args=None):
        # Isn't used anymore due to requirement of adjusting sigma based on epoch
        # trsfm = transforms.Compose([
        #     Gaussfilter(custom_args['sigma']),
        #     Normalize(),
        #     # Resize(
        #     #     rescale_input=(custom_args['rescale_X_input'], custom_args['rescale_Y_input'])
        #     # ),
        #     ToTensor()
        # ])

        self.data_dir = data_dir
        self.dataset = XRayDataset(self.data_dir, transform=True, custom_args=custom_args)
        super(XRayDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
