import torch
import torch.nn as nn


class MultiScaleFusion(nn.Module):

    def __init__(
        self,
        channels=768,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 3,
                channels,
                kernel_size=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        coarse,
        medium,
        fine,
    ):

        x = torch.cat(
            [
                coarse,
                medium,
                fine,
            ],
            dim=1,
        )

        return self.fusion(x)