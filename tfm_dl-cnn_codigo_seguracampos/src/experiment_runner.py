# pylint: disable=too-many-arguments
"""
In this file, it will be defined the functions and classes that will be needed
to train and test the models.
"""
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.metric_recollector import MetricRecollector


class ExperimentRunner:
    """
    This class defines the methods to train and test the model
    """
    def __init__(self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Any
    ) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.metrics = MetricRecollector() # TODO: Use this property

    def train_model(self) -> None:
        """Defines the standard process to train a model"""
        for _, (X_train, y_train) in enumerate(self.train_loader):# TODO: Use the bath number
            # Model Application
            y_pred = self.model(X_train)
            # Loss calculation
            loss = self.criterion(y_pred,y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def test_model(self) -> None:
        """Defines the standard process to test a model"""
# TODO: Implement this method
