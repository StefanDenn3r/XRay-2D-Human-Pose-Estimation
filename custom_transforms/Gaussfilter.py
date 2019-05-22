import numpy as np
from scipy.ndimage import gaussian_filter


class Gaussfilter(object):
    """Applies Gaussian filter on landmark matrices"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image, landmarks = sample

        landmarks = np.array([gaussian_filter(landmark, sigma=self.sigma) for landmark in landmarks])

        return image, landmarks
