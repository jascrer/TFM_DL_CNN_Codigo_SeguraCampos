"""
Functions for saving experiment data
"""

from os.path import join
import torch
from torch.nn import Module
from src.metrics.metric_recollector import MetricRecollector

DATA_PROCESSED_FOLDER = 'data/processed'

def save_experiment_data(
    experiment: str,
    model: Module,
    metrics: MetricRecollector) -> None:
    """Function to save experiment data"""
    experiment_folder = join(DATA_PROCESSED_FOLDER, experiment.lower())
    torch.save(model.state_dict(),
        join(experiment_folder, f'{experiment.lower()}.pt'))
    metrics.save_metrics(experiment_folder)
