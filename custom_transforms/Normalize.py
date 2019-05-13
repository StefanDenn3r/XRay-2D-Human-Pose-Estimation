import cv2
import numpy as np


class Normalize(object):
    """Normalizes image with MinMaxNorm and values [0,1]"""

    def __call__(self, sample):
        image, landmarks = sample

        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        landmarks = np.array(
            [cv2.normalize(landmark, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F) for landmark in landmarks]
        )
        return image, landmarks
