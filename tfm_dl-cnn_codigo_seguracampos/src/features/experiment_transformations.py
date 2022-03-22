"""
In this file, the transformations that will be applied to the data are being defined.
"""
from torchvision import transforms

no_augmentation_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

flip_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])