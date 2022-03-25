"""
In this file, it will be defined the necessary classes and functions to
recollect the accuracy for the training and testing steps.
Accuracy is the key metric of the project.
"""

import pandas as pd

from torch import Tensor

class MetricRecollector:
    """This class will recollect the loss and accuracy for the train and the test steps"""
    def __init__(self) -> None:
        self.train_metrics = pd.DataFrame(columns=['batch_number', 'loss', 'accuracy'])
        self.test_metrics = pd.DataFrame(columns=['batch_number', 'loss', 'accuracy'])

    def add_train_metrics(self,
        batch:int,
        accuracy: float,
        loss: Tensor
    ) -> None:
        """Adds the train metrics to the DataFrame"""
        self.train_metrics = self.train_metrics.append({
            'batch_number': batch,
            'loss': loss,
            'accuracy': accuracy
        }, ignore_index=True)

    def add_test_metrics(self,
        batch:int,
        accuracy: float,
        loss: float
    ) -> None:
        """Adds the train metrics to the DataFrame"""
        self.test_metrics = self.test_metrics.append({
            'batch_number': batch,
            'loss': loss,
            'accuracy': accuracy
        }, ignore_index=True)
