import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselSpatialAttention(nn.Module):

    def __init__(
        self,
        channels=768,
        reduction=8
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            32
        )

        self.mask_projection = nn.Sequential(
            nn.Conv2d(
                1,
                hidden,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                1,
                kernel_size=1
            )
        )

        self.feature_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.norm = nn.BatchNorm2d(
            channels
        )

    def forward(
        self,
        features,
        vessel_mask
    ):

        vessel_mask = F.interpolate(
            vessel_mask.float(),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        attention = self.mask_projection(
            vessel_mask
        )

        attention = torch.sigmoid(
            attention
        )

        projected = self.feature_projection(
            features
        )

        output = (
            features
            + projected * attention
        )

        output = self.norm(
            output
        )

        return output, attention

class VesselAwareFeatures(nn.Module):

    def __init__(
        self,
        channels=768
    ):
        super().__init__()

        self.attention = (
            VesselSpatialAttention(
                channels=channels
            )
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.GELU()
        )

    def forward(
        self,
        features,
        vessel_mask
    ):

        features, attention = (
            self.attention(
                features,
                vessel_mask
            )
        )

        features = self.refine(
            features
        )

        return features, attention