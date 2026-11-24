
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Removes right-side padding so the convolution
    does not use future time steps.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A residual TCN block containing two
    causal dilated convolution layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        output = self.network(x)
        residual = self.residual(x)

        return self.activation(output + residual)


class FedGuardTCN(nn.Module):
    """
    FedGuard-ICS Temporal Convolutional Network.

    Input:
        (batch_size, sequence_length, input_features)

    Output:
        (batch_size, number_of_classes)
    """

    def __init__(
        self,
        input_features=51,
        hidden_channels=64,
        num_classes=2,
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        channels = [
            hidden_channels,
            hidden_channels,
            hidden_channels,
            hidden_channels
        ]

        dilations = [1, 2, 4, 8]

        layers = []
        in_channels = input_features

        for out_channels, dilation in zip(
            channels, dilations
        ):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        self.classifier = nn.Linear(
            hidden_channels,
            num_classes
        )

    def forward(self, x):
        # Input: (batch, time, features)
        # Conv1D expects: (batch, channels, time)
        x = x.transpose(1, 2)

        x = self.tcn(x)

        # Use the final time step for classification.
        x = x[:, :, -1]

        return self.classifier(x)
