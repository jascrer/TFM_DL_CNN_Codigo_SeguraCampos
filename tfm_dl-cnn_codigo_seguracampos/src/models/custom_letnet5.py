"""
In this file, the custom LetNet-5 use in the experiment will be recreated.
Version: 1.0
"""
from torch import nn


class PhiLetnet(nn.Module):
    """
    The custom LetNet-5 CNN described in the article, defined with the greek letter PHI
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride= 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride= 1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x_parameter):
        """
        Data processing method
        """
        x_parameter = self.features(x_parameter)
        x_parameter = self.flatten(x_parameter)
        x_parameter = self.classifier(x_parameter)
        return x_parameter
