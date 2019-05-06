import cv2


class Normalize(object):
    """Normalizes image with MinMaxNorm and values [0,1]"""

    def __call__(self, sample):
        image, landmarks = sample

        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        return image, landmarks
