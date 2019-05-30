import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_loader.data_loaders import XRayDataLoader


def main():
    batch_size = 2
    for i_batch, sample_batched in enumerate(
            XRayDataLoader(os.path.join(Path(__file__).parent.parent, "data/XRay/Ex3"), batch_size)):
        batch_images, batch_landmarks = sample_batched

        if i_batch == 0:
            print(f"batch_images shape: {batch_images.shape}; batch_landmark shape: {batch_landmarks.shape}")

        for i in range(batch_size):
            (image, landmarks) = batch_images[i].numpy(), batch_landmarks[i].numpy()
            channel, height, width = image.shape
            image = cv2.resize(image[0], (256, 256), cv2.INTER_CUBIC)

            # Display normalized Image
            # Image.fromarray(image[0] * 255).show()

            # Display Landmarks
            # Image.fromarray(np.sum(landmarks, axis=0) * 255).show()

            # Display Landsmarks in normalized Image
            stacked_image = np.maximum(image * 255, (np.sum(landmarks, axis=0) * 255))
            Image.fromarray(stacked_image).show()

        break


main()
