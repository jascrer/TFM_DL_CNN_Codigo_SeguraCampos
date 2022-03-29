# pylint: disable=no-member
"""
In this file, it will be defined the necessary classes and functions to
recollect the accuracy for the training and testing steps.
Accuracy is the key metric of the project.
"""
from os.path import join
import pandas as pd

import torch
from torch import Tensor

class MetricRecollector:
    """This class will recollect the loss and accuracy for the train and the test steps"""
    def __init__(self) -> None:
        self.train_metrics = pd.DataFrame(columns=['epoch','batch_number', 'loss', 'accuracy'])
        self.test_metrics = pd.DataFrame(columns=['epoch','batch_number', 'loss', 'accuracy'])

    def add_train_metrics(self,
        epoch:int,
        batch:int,
        accuracy: float,
        loss: Tensor
    ) -> None:
        """Adds the train metrics to the DataFrame"""
        self.train_metrics = self.train_metrics.append({
            'epoch': epoch+1,
            'batch_number': batch+1,
            'loss': loss,
            'accuracy': accuracy
        }, ignore_index=True)

    def add_test_metrics(self,
        epoch:int,
        batch:int,
        accuracy: float,
        loss: float
    ) -> None:
        """Adds the train metrics to the DataFrame"""
        self.test_metrics = self.test_metrics.append({
            'epoch': epoch+1,
            'batch_number': batch+1,
            'loss': loss,
            'accuracy': accuracy
        }, ignore_index=True)

    def save_metrics(self,
        path_to_save: str):
        """Save a dataframe in csv for later analysis"""
        self.train_metrics.to_csv(join(path_to_save, "train.csv"))
        self.test_metrics.to_csv(join(path_to_save, "test.csv"))


def accuracy_calculation(y_true: Tensor,
    y_pred: Tensor,
    data_size: int,
    correct_predicted: int
) -> float:
    """Calculates the accuracy measure based on the true and predicted values"""
    predicted: Tensor = torch.max(y_pred, dim=1)[1]
    correct_predicted += (predicted == y_true).sum()
    accuracy_score: float = correct_predicted.item()/data_size
    return accuracy_score, correct_predicted
