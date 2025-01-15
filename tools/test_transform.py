import numpy as np
from PIL import Image
from torchvision import transforms

from data.augmentation.augmentations import DataAugmentation

# Hàm ngược chuẩn hóa
def denormalize_image(image,):
    """
    Ngược chuẩn hóa ảnh.
    Args:
        image: Tensor ảnh chuẩn hóa (C, H, W)
        mean: Danh sách các giá trị mean (R, G, B)
        std: Danh sách các giá trị std (R, G, B)
    Returns:
        Ảnh đã ngược chuẩn hóa (numpy array)
    """
    # Chuyển từ Tensor sang numpy và chuyển trục từ (C, H, W) -> (H, W, C)
    image = image.permute(1, 2, 0).cpu().numpy()
    
    # Ngược chuẩn hóa
    image = image * std + mean
    
    # Clip giá trị trong khoảng [0, 1]
    image = np.clip(image, 0, 1)
    image = image * 255
    return image.astype(np.uint8)

transform = DataAugmentation(
    (0.7, 1.),
    (0.6, 0.8),
    2,
    224,
    112

)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = Image.open("/media/anlab/data-1tb/ruybk/mericar_crawl/product_images_0113/m32519809493/3.jpg")

image_transformed = transform(image)
all_transformed = tuple(image_transformed["global_crops"] + image_transformed["local_crops"])


for idx, im_item in enumerate(all_transformed):
    Image.fromarray(denormalize_image(im_item)).save(f"{idx}.jpg")

assert False
local_0 = image_transformed["local_crops"][0]
local_1 = image_transformed["local_crops"][1]

global_crop0 = image_transformed["global_crops"][0]
global_crop1 = image_transformed["global_crops"][1]

Image.fromarray(denormalize_image(local_1)).save("local1.jpg")

Image.fromarray(denormalize_image(global_crop0)).save("glocal0.jpg")
Image.fromarray(denormalize_image(global_crop1)).save("glocal1.jpg")

# image_transformed.save("data/augmentation/1_transformed.jpg")