
import torch
from torch import nn

from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL


class Method_CNN_ORL_Experiment(Method_CNN_ORL):

    def __init__(
        self,
        mName,
        mDescription,
        hidden_dim=128,
        num_conv_layers=2
    ):
        super().__init__(mName, mDescription)

        layers = []
        in_channels = 1
        channels = [16, 32][:num_conv_layers]

        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Automatically calculate flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 112, 92)
            feature_size = self.conv_layers(dummy).numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 40)
        )
