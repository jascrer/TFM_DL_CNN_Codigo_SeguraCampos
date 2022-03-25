"""
In this file, there will be defined the experiments that are going to be executed
"""

from src.features.transformations import no_augmentation_transform, flip_transform
from src.data.load_dataset import create_test_dataloader, create_train_dataloader
from src.experiment_runner import run_experiment
from src.models.custom_letnet5 import PhiLetnet


def phi_experiment_no_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        lr: float,
        data_folder: str
    ) -> None:
    """Function to run the original experiment - No augmentation in the data"""
    train_dataloader = create_train_dataloader(
        transform=no_augmentation_transform,
        root=data_folder,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=no_augmentation_transform,
        root=data_folder,
        batch_size=batch_size,
        seed=seed)

    phi_model=PhiLetnet()

    run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        lr,
        phi_model
    )

def phi_experiment_flip_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        lr: float,
        data_folder: str
    ) -> None:
    """Function to run the original experiment - No augmentation in the data"""
    train_dataloader = create_train_dataloader(
        transform=flip_transform,
        root=data_folder,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=flip_transform,
        root=data_folder,
        batch_size=batch_size,
        seed=seed)

    phi_model=PhiLetnet()

    run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        lr,
        phi_model
    )
