# Modified from:
#   https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py

from torchvision import transforms

from .gauss_blur import GaussianBlur, make_normalize_transform
from .perspective_roatate import RandomPerspectiveAndRotation
from .superpixels import SuperPixels

class DataAugmentation(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=224,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        print("###################################")
        print("Using data augmentation parameters:")
        print(f"global_crops_scale: {global_crops_scale}")
        print(f"local_crops_scale: {local_crops_scale}")
        print(f"global_crops_size: {global_crops_size}")
        print(f"local_crops_size: {local_crops_size}")
        print("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                RandomPerspectiveAndRotation(
                    perspective=1., rotation=0.5, large_range=(0.2, 0.4)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                RandomPerspectiveAndRotation(
                    perspective=0.3, rotation=0.3, small_range=(0.05, 0.1), large_range=(0.15, 0.3)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.2)],
                    p=0.7,
                ),
            ]
        )

        global_transfo1_extra = transforms.Compose(
            [
                GaussianBlur(p=0.4),
            ]
        )
        global_transfo2_extra = transforms.Compose(
            [
                SuperPixels(p=0.3),
            ]
        )
        local_transfo_extra = transforms.Compose(
            [
                GaussianBlur(p=0.3),
            ]
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform()
            ]
        )

        self.global_transform1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transform2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transform = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transform1(im1_base)

        # im2_base = self.geometric_augmentation_global(image)
        # global_crop_2 = self.global_transform2(im2_base)

        output["global_crops"] = [global_crop_1]

        # local crops:
        local_crops = [
            self.local_transform(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops

        return output