# pylint: disable=too-many-arguments
"""
In this file, it will be defined the functions and classes that will be needed
to train and test the models.
"""
from typing import Any
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics.metric_recollector import MetricRecollector, accuracy_calculation


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
        self.model = model.cuda()
        self.optimizer = optimizer
        self.criterion = criterion

        self.metrics = MetricRecollector()

    def train_model(self, epoch: int) -> None:
        """Defines the standard process to train a model"""
        correct_predicted: int = 0
        data_size: int = 0
        for batch_number, (X_train, y_train) in enumerate(self.train_loader):
            # Collect size for accuracy measure
            data_size += X_train.size(0)
            #Convert to CUDA
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            # Model Application
            y_pred = self.model(X_train)

            # Loss calculation
            loss: torch.Tensor = self.criterion(y_pred,y_train)
            #Accuracy calculation
            accuracy, correct_predicted = accuracy_calculation(
                y_true=y_train,
                y_pred=y_pred,
                data_size=data_size,
                correct_predicted=correct_predicted
            )
            #Recollect metrics
            self.metrics.add_train_metrics(epoch,batch_number, accuracy, loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def test_model(self, epoch: int) -> None:
        """Defines the standard process to test a model"""
        with torch.no_grad():
            correct_predicted: int = 0
            data_size: int = 0
            for batch_number, (X_test, y_test) in enumerate(self.test_loader):
                # Collect size for accuracy measure
                data_size += X_test.size(0)

                #Convert to CUDA
                X_test = X_test.cuda()
                y_test = y_test.cuda()
                # Model evaluation
                y_validation = self.model(X_test)

                # Loss calculation
                loss = self.criterion(y_validation, y_test)
                #Accuracy calculation
                accuracy, correct_predicted = accuracy_calculation(
                    y_true=y_test,
                    y_pred=y_validation,
                    data_size=data_size,
                    correct_predicted=correct_predicted
                )
                self.metrics.add_test_metrics(epoch,batch_number, accuracy, loss.item())


def run_experiment(
        epoch_count: int,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        learning_rate: float,
        model: nn.Module) -> MetricRecollector:
    """Runs the experiment"""
    runner = ExperimentRunner(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        criterion=nn.NLLLoss())

    for epoch in tqdm(range(epoch_count), desc='Epoch', ncols=epoch_count):
        runner.train_model(epoch)

        runner.test_model(epoch)

    return runner.model, runner.metrics
