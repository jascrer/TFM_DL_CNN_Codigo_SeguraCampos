"""
In this file, the transformations that will be applied to the data are being defined.
"""
from torchvision import transforms

MEAN_VECTOR = [0.5032, 0.4517, 0.4679]
STD_VECTOR = [0.1513, 0.1399, 0.1533]

no_augmentation_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_VECTOR,
                        STD_VECTOR)
])

flip_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_VECTOR,
                        STD_VECTOR)
])
