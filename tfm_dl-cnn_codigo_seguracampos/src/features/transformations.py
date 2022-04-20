"""
In this file, the transformations that will be applied to the data are being defined.
"""
from torchvision import transforms

MEAN_VECTOR = [0.5032, 0.4517, 0.4679]
STD_VECTOR = [0.1513, 0.1399, 0.1533]

def no_augmentation_creator(image_resize: int
    ) -> transforms.Compose:
    """Creates the No Augmentation transform with the new size for the images"""
    no_augmentation_transform = transforms.Compose([
        transforms.Resize((image_resize,image_resize)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_VECTOR,
                            STD_VECTOR)
    ])
    return no_augmentation_transform

def flip_creator(image_resize: int
    ) -> transforms.Compose:
    """Creates the Flip Augmentation transform with the new size for the images"""
    flip_transform = transforms.Compose([
        transforms.Resize((image_resize,image_resize)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_VECTOR,
                            STD_VECTOR)
    ])
    return flip_transform
