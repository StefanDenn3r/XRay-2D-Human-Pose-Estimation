import cv2
import numpy as np


class Resize(object):
    """Resizes labels and images to given shape"""

    def __init__(self, rescale_input, rescale_target, interpolation=cv2.INTER_CUBIC):
        self.rescale_input = rescale_input
        self.rescale_target = rescale_target
        self.interpolation = interpolation

    def __call__(self, sample):
        image, landmarks = sample

        image = cv2.resize(image, self.rescale_input, self.interpolation)

        landmarks = np.array(
            [cv2.resize(landmark, self.rescale_target, self.interpolation) for landmark in landmarks]
        )

        return image, landmarks
