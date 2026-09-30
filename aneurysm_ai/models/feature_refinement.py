import torch
import torch.nn as nn

from .deformable import (
    DeformableAttention2D,
)
from .attention import (
    VesselSpatialAttention,
)
class FeatureRefinement(nn.Module):

    def __init__(
        self,
        channels=768,
    ):
        super().__init__()

        self.deformable = (
            DeformableAttention2D(
                channels=channels,
                heads=8,
                points=4,
            )
        )

        self.vessel = (
            VesselSpatialAttention(
                channels=channels,
                reduction=8,
            )
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
            ),
        )

        self.norm = nn.BatchNorm2d(
            channels
        )

    def forward(
        self,
        features,
        vessel_mask=None,
    ):

        features = self.deformable(
            features
        )

        vessel_attention = None

        if vessel_mask is not None:

            features, vessel_attention = (
                self.vessel(
                    features,
                    vessel_mask,
                )
            )

        residual = features

        features = self.refine(
            features
        )

        features = self.norm(
            features + residual
        )

        return (
            features,
            vessel_attention,
        )