import numpy as np
import cv2
import random
from typing import Union
from PIL import Image
from albumentations import Superpixels


class SuperPixels:
    """
    Initialize the Superpixels class.

    Args:
        p_replace (tuple| list | float): Probability range for replacing pixels with superpixels.
        p (float): Probability of applying the superpixels augmentation.
        n_segments (tuple | list| float): Range of the number of segments for superpixel generation.
        max_size (int, optional): Maximum size of the image. Defaults to None.
        interpolation (int): Interpolation method for resizing the image. Defaults to cv2.INTER_LINEAR.
    """
    def __init__(self, p_replace=(0, 0.1), p=0.5, n_segments=[50, 120], max_size: Union[None, int] = None, interpolation: int = cv2.INTER_LINEAR):    
        self.augmenter = Superpixels(p_replace=p_replace, p=p, n_segments=n_segments, max_size=max_size, interpolation=interpolation)

    def __call__(self, image):
        """
        Apply superpixels to the input image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with superpixels applied.
        """
        return_pil = False
        if isinstance(image, Image.Image):
            return_pil = True
            image = np.asarray(image)
        image = self.augmenter(image=image)["image"]
        return Image.fromarray(image) if return_pil else image
