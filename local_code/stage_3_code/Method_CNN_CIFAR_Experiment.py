
import torch
from torch import nn

from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR


class Method_CNN_CIFAR_Experiment(Method_CNN_CIFAR):

    def __init__(
        self,
        mName,
        mDescription,
        kernel_size=3,
        pooling='max',
        hidden_dim=256,
        num_conv_layers=3,
        loss='cross_entropy'
    ):
        super().__init__(mName, mDescription)

        self.loss_type = loss

        padding = kernel_size // 2

        layers = []
        in_channels = 3

        channels = [32, 64, 128][:num_conv_layers]

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

            # Match the original CIFAR architecture:
            # pooling after the first two convolutions
            if i < 2:
                if pooling == 'max':
                    layers.append(nn.MaxPool2d(2))
                elif pooling == 'avg':
                    layers.append(nn.AvgPool2d(2))
                else:
                    raise ValueError("Unsupported pooling type")

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Automatically determine flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            feature_size = self.conv_layers(dummy).numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )


    def train(self, X, y):
        # Select the loss function for this experiment
        if self.loss_type == 'cross_entropy':
            self.loss_function = nn.CrossEntropyLoss()
        elif self.loss_type == 'multi_margin':
            self.loss_function = nn.MultiMarginLoss()
        else:
            raise ValueError("Unsupported loss function")

        return super().train(X, y)