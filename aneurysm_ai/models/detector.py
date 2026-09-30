import torch
import torch.nn as nn

from .transfer import AneurysmFeatureEncoder


class DetectionHead(nn.Module):
    """
    Predicts:
      - objectness: [B, 1]
      - normalized 3D center: [B, 3]
      - anatomy class: [B, 13]
    """

    def __init__(
        self,
        in_channels=768,
        num_classes=13,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.objectness = nn.Linear(hidden_dim, 1)

        self.center = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        self.anatomy = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        x = self.pool(features)
        x = torch.flatten(x, 1)

        x = self.shared(x)

        return {
            "objectness": self.objectness(x),
            "center": self.center(x),
            "anatomy": self.anatomy(x),
        }


class AneurysmDetector(nn.Module):
    """
    Swin + vessel-aware feature encoder + detection head.
    """

    def __init__(
        self,
        model_name,
        checkpoint_path,
        num_classes=13,
    ):
        super().__init__()

        self.encoder = AneurysmFeatureEncoder(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
        )

        self.head = DetectionHead(
            in_channels=768,
            num_classes=num_classes,
        )

    def forward(self, images, vessel_mask=None):
        features, vessel_attention = self.encoder(
            images,
            vessel_mask=vessel_mask,
        )

        predictions = self.head(features)

        predictions["features"] = features
        predictions["vessel_attention"] = vessel_attention

        return predictions

    def freeze_encoder(self):
        self.encoder.freeze_encoder()

    def unfreeze_last_stage(self):
        self.encoder.unfreeze_last_stage()

    def unfreeze_encoder(self):
        self.encoder.unfreeze_encoder()