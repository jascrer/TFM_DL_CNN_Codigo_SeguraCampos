"""
In this file, it will be defined the AlexNet definition for later comparations with the PhiLetNet.
Version: 1.0
"""
from torch import nn


class AlexNet(nn.Module):
    """
    This is the definition of the AlexNet which will be compared to the Phi-LetNet-5.
    This AlexNet will be modified in order to classify just two classes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=9216, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024,out_features=2),
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
