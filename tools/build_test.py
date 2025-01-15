import os
import random
import numpy as np
from glob import glob
from PIL import Image
from torchvision import transforms

from data.augmentation.augmentations import DataAugmentation

def denormalize_image(image,):
    image = image.permute(1, 2, 0).cpu().numpy()
    
    image = image * std + mean
    
    image = np.clip(image, 0, 1)
    image = image * 255
    return image.astype(np.uint8)

transform = DataAugmentation(
    (0.7, 1.),
    (0.6, 0.8),
    1,
    224,
    112
)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


db_path = "/media/anlab/data-1tb/ruybk/kaitori-robo-dbs/Resma/train"
dst_path = "/media/anlab/data-1tb/ruybk/kaitori-robo-dbs/Resma/build_test"
os.makedirs(dst_path, exist_ok=True)
image_paths = glob(db_path + "/*/image_1*") + glob(db_path + "/*/image_2*") + glob(db_path + "/*/image_3*")

random_paths = random.sample(image_paths, 100)

for idx, path in enumerate(random_paths):
    image = Image.open(path)
    image = transform(image)
    transformed = denormalize_image(image["global_crops"][0])
    Image.fromarray(transformed).save(os.path.join(dst_path, f"{idx}.jpg"))
