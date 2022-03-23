"""
In this file, the builders for the DataLoaders will be created.
"""
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_train_dataloader(
        transform: transforms.Compose,
        root: str,
        batch_size: int,
        seed: int
    ) -> DataLoader:
    """Creates a dataloader for training, with an specific transform"""
    return _create_dataloader(transform, root, "train", batch_size, seed)


def create_test_dataloader(
        transform: transforms.Compose,
        root: str,
        batch_size: int,
        seed: int
    ) -> DataLoader:
    """Creates a dataloader for testing, with an specific transform"""
    return _create_dataloader(transform, root, "test", batch_size, seed)


def _create_dataloader(
        transform: transforms.Compose,
        root: str,
        folder:str,
        batch_size: int,
        seed: int
    ) -> DataLoader:
    """Creates a dataloader, with an specific transform"""
    torch.manual_seed(seed)
    data = datasets.ImageFolder(join(root, folder), transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=True)
