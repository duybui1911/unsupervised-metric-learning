import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class RandomText:
    """
    A class that applies random text to an input image.

    Args:
        range_text_lenght (tuple): A tuple specifying the range of text lengths to generate. Default is (5, 20).
        font (str): The font file to use for the text. Default is "arial.ttf".
        range_font_size (tuple): A tuple specifying the range of font sizes to use. Default is (10, 30).
    """

    def __init__(
        self,
        range_text_lenght=(5, 30),
        font="arial.ttf",
        range_font_size=(10, 30),
        char_set='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ><!@#$%^&*()_+'
    ):
        """Initialize the class with the desired text, font, size, color, and position.
        """
        self.font = font
        self.range_font_size = range_font_size
        self.range_text_lenght = range_text_lenght
        self.char_set = char_set

    def __call__(self, image):
        """
        Apply random text to the input image.

        Args:
            image (numpy.ndarray or PIL.Image.Image): The input image.

        Returns:
            numpy.ndarray or PIL.Image.Image: The image with the text added.
        """
        return_pil = False
        # Convert the image to a PIL image if it is a numpy array
        if isinstance(image, Image.Image):
            return_pil = True
            image = np.asarray(image)
        
        height, width = image.shape[:2]
        font_size = random.randint(*self.range_font_size) / (max(self.range_font_size) + 5)
        text = ''.join(random.choices(
            self.char_set,
            k=random.randint(*self.range_text_lenght)
        ))
        direction = random.choice(['rtl', 'ltr', 'ttb'])
        position = (random.randint(0, 2*width//3 - 1), random.randint(0, 2*height//3 - 1))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # Load the font and create a drawing object
        image = cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            random.randint(1, 2),
            bottomLeftOrigin=direction=="ttb"
        )

        # Draw the text on the image
        return image if not return_pil else Image.fromarray(image)

if __name__ == "__main__":
    image = cv2.imread("/media/anlab/data-1tb/ruybk/mericar_crawl/product_images_0113/m32519809493/2.jpg")
    image = cv2.resize(image, (224, 224))
    image = RandomText()(image)
    cv2.imwrite("image.jpg", image)