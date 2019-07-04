import cv2


class Resize(object):
    """Resizes labels and images to given shape"""

    def __init__(self, rescale_input, interpolation=cv2.INTER_CUBIC):
        self.rescale_input = rescale_input
        self.interpolation = interpolation

    def __call__(self, sample):
        image, landmarks = sample

        image = cv2.resize(image, self.rescale_input, self.interpolation)

        return image, landmarks
