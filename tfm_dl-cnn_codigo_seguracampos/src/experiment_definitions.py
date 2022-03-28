"""
In this file, there will be defined the experiments that are going to be executed
"""
from os.path import join
from time import time
from enum import Enum, auto

import torch
from src.features.transformations import no_augmentation_transform, flip_transform
from src.data.load_dataset import create_test_dataloader, create_train_dataloader
from src.experiment_runner import run_experiment
from src.models.custom_letnet5 import PhiLetnet

DATA_FOLDER = 'data/interim/experiment'
DATA_PROCESSED_FOLDER = 'data/processed'

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

def phi_experiment_no_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the original experiment - No augmentation in the data"""
    train_dataloader = create_train_dataloader(
        transform=no_augmentation_transform,
        root=DATA_FOLDER,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=no_augmentation_transform,
        root=DATA_FOLDER,
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
    experiment_folder = join(DATA_PROCESSED_FOLDER, Experiments.PHI_NO_AUG.name.lower())
    torch.save(model.state_dict(),
        join(experiment_folder, f'{Experiments.PHI_NO_AUG.name.lower()}.pt'))
    metrics.save_metrics(experiment_folder)
    log_experiments(Experiments.PHI_NO_AUG.name.lower(), end_time-start_time)

def phi_experiment_flip_transform(
        epoch_count: int,
        batch_size: int,
        seed: int,
        learning_rate: float
    ) -> None:
    """Function to run the original experiment - No augmentation in the data"""
    train_dataloader = create_train_dataloader(
        transform=flip_transform,
        root=DATA_FOLDER,
        batch_size=batch_size,
        seed=seed)

    test_dataloader = create_test_dataloader(
        transform=flip_transform,
        root=DATA_FOLDER,
        batch_size=batch_size,
        seed=seed)

    phi_model=PhiLetnet()

    start_time = time()
    model, metrics= run_experiment(epoch_count,
        train_dataloader,
        test_dataloader,
        learning_rate,
        phi_model
    )
    end_time = time()
    experiment_folder = join(DATA_PROCESSED_FOLDER, Experiments.PHI_FLIP.name.lower())
    torch.save(model.state_dict(),
        join(experiment_folder, f'{Experiments.PHI_FLIP.name.lower()}.pt'))
    metrics.save_metrics(experiment_folder)
    log_experiments(Experiments.PHI_NO_AUG.name.lower(), end_time-start_time)
