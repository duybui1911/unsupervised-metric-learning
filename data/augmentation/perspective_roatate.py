import cv2
import random
import numpy as np
from PIL import Image
from typing import Tuple


class RandomPerspectiveAndRotation:
    """
    A class that applies random perspective and rotation transformations to an image.

    Args:
        perspective (float, optional): Probability to apply perspective transform. Defaults to 0.5.
        rotation (float, optional): Probability to apply rotation transform. Defaults to 0.5.
        angle (int, optional): Angle range for rotation. Defaults to 45.
        small_range (Tuple, optional): Range for small perspective transform. Defaults to (0.1, 0.15).
        large_range (Tuple, optional): Range for large perspective transform. Defaults to (0.2, 0.5).
    """

    def __init__(
        self,
        perspective=0.5,
        rotation=0.5,
        angle=45,
        small_range: Tuple = (0.1, 0.15),
        large_range: Tuple = (0.2, 0.5)
    ):
        """Initialize the class with the desired perspective and rotation ranges.
        """
        self.do_perspective = perspective
        self.do_rotation = rotation
        self.angle = angle
        self.small_range = small_range
        self.large_range = large_range
        self.dict_points = {}

    def __call__(self, image):
        """
        Apply random transformations to the input image.

        Args:
            image (numpy.ndarray or PIL.Image.Image): The input image.

        Returns:
            numpy.ndarray or PIL.Image.Image: The transformed image.
        """
        return_pil = False
        # Convert the image to a numpy array if it is a PIL image
        if isinstance(image, Image.Image):
            return_pil = True
            image = np.asarray(image)

        # Randomly select if the rotation should be applied
        if random.uniform(0, 1) < self.do_rotation:
            image = self.rotate_image(
                image, random.uniform(-self.angle, self.angle))

        # Randomly select if the perspective should be applied
        if random.uniform(0, 1) < self.do_perspective:
            self.set_points()
            key = random.choice(list(self.dict_points.keys()))
            points = self.dict_points[key]
            points = [[int(x * image.shape[1]), int(y * image.shape[0])]
                      for x, y in points]
            image = self.transform_image(image, points)

        return Image.fromarray(image) if return_pil else image

    @staticmethod
    def transform_image(image, points):
        """Apply a perspective transformation to an image.

        Args:
            image (numpy.ndarray): The input image.
            points (numpy.ndarray): The four corners of the desired output.

        Returns:
            numpy.ndarray: The transformed image.
        """
        width = image.shape[1]
        height = image.shape[0]

        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32(points)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        value = int(random.uniform(0, 1) * 255)
        result = cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(value, value, value)
        )

        return result

    # The business requirements in my project require this augment step.
    @staticmethod
    def rotate_image(image, angle):
        """
        Rotates an image by the specified angle.

        Args:
            image (numpy.ndarray): The input image to be rotated.
            angle (float): The angle (in degrees) by which the image should be rotated.

        Returns:
            numpy.ndarray: The rotated image.
        """
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        value = int(random.uniform(0, 1) * 255)
        rotated_image = cv2.warpAffine(
            image,
            matrix,
            (image.shape[1], image.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(value, value, value)
        )

        return rotated_image

    def set_points(self):
        """
        Sets the points for warping.

        This method generates random values for the warp points and updates the `dict_points` dictionary
        with the new values.
        """
        v_large1 = random.uniform(self.large_range[0], self.large_range[1])
        v_large2 = random.uniform(self.large_range[0], self.large_range[1])
        v_small1 = random.uniform(self.small_range[0], self.small_range[1])
        v_small2 = random.uniform(self.small_range[0], self.small_range[1])

        dict_points = self.dict_points
        dict_points["left_warp"] = [
            [v_large1, v_small1], [1, 0], [v_large2, 1 - v_small2], [1, 1]
        ]
        dict_points["right_warp"] = [
            [0, 0], [1 - v_large1, v_small1], [0, 1], [1 - v_large2, 1 - v_small2]
        ]
        dict_points["top_warp"] = [
            [v_small1, v_large1], [1 - v_small2, v_large2], [0, 1], [1, 1]
        ]
        dict_points["bottom_warp"] = [
            [0, 0], [1, 0], [v_small1, 1-v_large1], [1 - v_small2, 1-v_large2]
        ]

        # Adjust the points for the corners
        v_small1 += 0.05
        v_large1 -= 0.1
        dict_points["top_left_warp"] = [
            random.choice([[v_large1, v_small1], [v_small1, v_large1]]), [
                1, 0], [0, 1], [1, 1]
        ]
        dict_points["top_right_warp"] = [
            [0, 0], random.choice(
                [[1 - v_large1, v_small1], [1 - v_small1, v_large1]]), [0, 1], [1, 1]
        ]
        dict_points["bottom_left_warp"] = [
            [0, 0], [1, 0], random.choice(
                [[v_small1, 1-v_large1], [v_large1, 1-v_small1]]), [1, 1]
        ]
        dict_points["bottom_right_warp"] = [
            [0, 0], [1, 0], [0, 1], random.choice(
                [[1 - v_small1, 1-v_large1], [1-v_large1, 1 - v_small1]])
        ]

        self.dict_points = dict_points
