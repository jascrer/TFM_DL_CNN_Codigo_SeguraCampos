"""
Functions for saving experiment data
"""

from os.path import join
import torch
from torch.nn import Module
from src.metrics.metric_recollector import MetricRecollector

DATA_PROCESSED_FOLDER = 'data/processed'
MODELS_FOLDER = 'models'

def save_experiment_data(
    experiment: str,
    model: Module,
    metrics: MetricRecollector) -> None:
    """Function to save experiment data"""
    experiment_folder = join(DATA_PROCESSED_FOLDER, experiment.lower())
    metrics.save_metrics(experiment_folder)
    torch.save(model.state_dict(),
        join(MODELS_FOLDER, f'{experiment.lower()}.pt'))
