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

        self.metrics = MetricRecollector()

    def train_model(self) -> None:
        """Defines the standard process to train a model"""
        train_corr = 0
        for batch_number, (X_train, y_train) in enumerate(self.train_loader):
            # Model Application
            y_pred = self.model(X_train)

            # Loss calculation
            loss = self.criterion(y_pred,y_train)
            #Accuracy calculation - TODO: Increase precision, improve code
            predicted = torch.max(y_pred.data, 1)[1]
            train_corr += (predicted == y_train).sum()
            accuracy: float = train_corr.item()/((batch_number+1)*2500)
            #Recollect metrics
            self.metrics.add_train_metrics(batch_number, 0, loss.item())
            print(f'batch number: {batch_number}, accuracy: {accuracy},loss: {loss.item()}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def test_model(self) -> None:
        """Defines the standard process to test a model"""
        with torch.no_grad():
            for batch_number, (X_test, y_test) in enumerate(self.test_loader):
                y_validation = self.model(X_test)

                # Loss calculation
                loss = self.criterion(y_validation, y_test)
                #Accuracy calculation
                #accuracy: float = accuracy_score(y_test.detach().numpy(),
                #                                y_validation.detach().numpy())
                self.metrics.add_test_metrics(batch_number, 0, loss.item())


def run_experiment(
        epoch_count: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        lr: float,
        model: nn.Module) -> None:
    """Runs the experiment"""
    runner = ExperimentRunner(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr),
        criterion=nn.NLLLoss())

    for epoch in range(epoch_count):
        #if epoch%50 ==0:
        print(f'Epoch: {epoch}/{epoch_count}')
        runner.train_model()

        runner.test_model()
