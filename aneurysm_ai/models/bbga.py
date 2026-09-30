import torch
import torch.nn as nn


class BBGA(nn.Module):
    """
    Bounding-Box / point-guided spatial attention.

    The input guide is a soft spatial map [B, 1, H, W].
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            32,
        )

        self.attention = nn.Sequential(
            nn.Conv2d(
                1,
                hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
            ),

            nn.Sigmoid(),
        )

        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        features,
        spatial_guide,
    ):

        attention = self.attention(
            spatial_guide
        )

        projected = self.projection(
            features
        )

        output = (
            features
            + projected * attention
        )

        return output, attention