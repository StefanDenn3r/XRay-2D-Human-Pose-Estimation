from scipy.ndimage import gaussian_filter


class Gaussfilter(object):
    """Applies Gaussian filter on landmark matrices"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        landmarks = gaussian_filter(landmarks, sigma=self.sigma)

        return {'image': image, 'landmarks': landmarks}
