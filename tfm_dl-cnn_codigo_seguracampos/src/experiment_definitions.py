"""
In this file, there will be defined the experiments that are going to be executed
"""
from time import time
from enum import Enum, auto

from src.data.save_data import save_experiment_data
from src.features.transformations import no_augmentation_creator, flip_creator
from src.data.load_dataset import create_test_dataloader, create_train_dataloader
from src.experiment_runner import run_experiment
from src.models.alexnet import AlexNet
from src.models.custom_letnet5 import PhiLetnet

PHIMODEL_IMAGESIZE = 32
ALEXMODEL_IMAGESIZE = 224

class Experiments(Enum):
    """Enum for defining each experiment"""
    PHI_NO_AUG = auto()
    PHI_FLIP = auto()
    ALEX_NO_AUG = auto()
    ALEX_FLIP = auto()

def log_experiments(experiment_name: str,
    execution_time: int):
    """Logs in console when an experiment has finished"""
    print(f'Experiment - {experiment_name} - has finished in {execution_time/60} minutes')

def phi_model_no_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the original model - No augmentation in the data"""
    no_augmentation_transform = no_augmentation_creator(PHIMODEL_IMAGESIZE)

    train_dataloader = create_train_dataloader(
        transform=no_augmentation_transform,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=no_augmentation_transform,
        batch_size=batch_size,
        seed=seed)

    phi_model=PhiLetnet()

    start_time = time()
    model, metrics = run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        learning_rate,
        phi_model
    )
    end_time = time()
    log_experiments(Experiments.PHI_NO_AUG.name.lower(), end_time-start_time)
    save_experiment_data(Experiments.PHI_NO_AUG.name, model, metrics)

def phi_model_flip_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the original model - Flip augmentation in the data"""
    flip_transform = flip_creator(PHIMODEL_IMAGESIZE)

    train_dataloader = create_train_dataloader(
        transform=flip_transform,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=flip_transform,
        batch_size=batch_size,
        seed=seed)

    phi_model=PhiLetnet()

    start_time = time()
    model, metrics = run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        learning_rate,
        phi_model
    )
    end_time = time()
    log_experiments(Experiments.PHI_FLIP.name.lower(), end_time-start_time)
    save_experiment_data(Experiments.PHI_FLIP.name, model, metrics)

def alexnet_model_no_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the AlexNet model - No augmentation in the data"""
    no_augmentation_transform = no_augmentation_creator(ALEXMODEL_IMAGESIZE)

    train_dataloader = create_train_dataloader(
        transform=no_augmentation_transform,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=no_augmentation_transform,
        batch_size=batch_size,
        seed=seed)

    alex_model = AlexNet()

    start_time = time()
    model, metrics = run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        learning_rate,
        alex_model
    )
    end_time = time()
    log_experiments(Experiments.ALEX_NO_AUG.name.lower(), end_time-start_time)
    save_experiment_data(Experiments.ALEX_NO_AUG.name.lower(), model, metrics)

def alexnet_model_flip_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the AlexNet model - No augmentation in the data"""
    flip_transform = flip_creator(ALEXMODEL_IMAGESIZE)

    train_dataloader = create_train_dataloader(
        transform=flip_transform,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=flip_transform,
        batch_size=batch_size,
        seed=seed)

    alex_model = AlexNet()

    start_time = time()
    model, metrics = run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        learning_rate,
        alex_model
    )
    end_time = time()
    log_experiments(Experiments.ALEX_FLIP.name.lower(), end_time-start_time)
    save_experiment_data(Experiments.ALEX_FLIP.name.lower(), model, metrics)
