import cv2
import numpy as np


class ResizeLabels(object):
    """Resizes labels to fit output shape of hourglass. Factor of resizing set in constructor, by default 2"""

    def __init__(self, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation

    def __call__(self, sample):
        image, landmarks = sample

        landmarks = np.array(
            [cv2.resize(landmark, (256, 256), self.interpolation)
             for landmark in landmarks]
        )

        return image, landmarks
