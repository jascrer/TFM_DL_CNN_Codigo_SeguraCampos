"""
In this file, it will be defined the necessary classes and functions to
recollect the accuracy for the training and testing steps.
Accuracy is the key metric of the project.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

class MetricRecollector:
    """This class will recollect the loss for the train and the test steps"""
    def __init__(self) -> None:
        self.metrics = pd.DataFrame(columns=['batch_number', 'loss', 'accuracy'])

    def add_metrics(self, 
        batch:int,
        y_true: np.array,
        y_predicted: np.array,
        loss: float
    ) -> None:
        """Adds the metrics to the DataFrame"""
        ac_score = accuracy_score(y_true, y_predicted)
        metrics = metrics.append({'batch_number': batch, 'loss': loss, 'accuracy': ac_score})
# TODO: Add more methods over the metrics
