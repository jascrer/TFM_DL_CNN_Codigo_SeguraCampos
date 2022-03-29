"""
In this file, the builders for the DataLoaders will be created.
"""
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_FOLDER = 'data/interim/experiment'

def create_train_dataloader(
        transform: transforms.Compose,
        batch_size: int,
        seed: int
    ) -> DataLoader:
    """Creates a dataloader for training, with an specific transform"""
    folder = join(DATA_FOLDER, "train")
    return _create_dataloader(transform, folder, batch_size, seed, True)


def create_test_dataloader(
        transform: transforms.Compose,
        batch_size: int,
        seed: int
    ) -> DataLoader:
    """Creates a dataloader for testing, with an specific transform"""
    folder = join(DATA_FOLDER, "test")
    return _create_dataloader(transform, folder, batch_size, seed, False)


def _create_dataloader(
        transform: transforms.Compose,
        folder:str,
        batch_size: int,
        seed: int,
        shuffle: bool
    ) -> DataLoader:
    """Creates a dataloader, with an specific transform"""
    torch.manual_seed(seed)
    data = datasets.ImageFolder(folder, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)
