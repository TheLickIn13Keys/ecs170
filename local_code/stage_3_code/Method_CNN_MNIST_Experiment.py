
import torch
from torch import nn

from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST


class Method_CNN_MNIST_Experiment(Method_CNN_MNIST):

    def __init__(
        self,
        mName,
        mDescription,
        kernel_size=3,
        num_conv_layers=2
    ):
        super().__init__(mName, mDescription)

        padding = kernel_size // 2

        layers = []
        in_channels = 1

        channels = [16, 32, 64][:num_conv_layers]

        for i, out_channels in enumerate(channels):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            layers.append(nn.ReLU())

            # Pool after the first two convolutional layers
            if i < 2:
                layers.append(nn.MaxPool2d(2))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the flattened feature size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            feature_size = self.conv_layers(dummy).numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
