import cv2
import numpy as np


class ResizeLabels(object):
    """Resizes labels to fit output shape of hourglass. Factor of resizing set in constructor, by default 2"""

    def __init__(self, factor=2, interpolation=cv2.INTER_CUBIC):
        self.factor = factor
        self.interpolation = interpolation

    def __call__(self, sample):
        image, landmarks = sample

        channels, height, width = landmarks.shape

        landmarks = np.array(
            [cv2.resize(landmark, (width // self.factor, height // self.factor), self.interpolation)
             for landmark in landmarks]
        )

        return image, landmarks
