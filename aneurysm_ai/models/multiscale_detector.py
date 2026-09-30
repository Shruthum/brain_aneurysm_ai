import torch
import torch.nn as nn

from .transfer import AneurysmFeatureEncoder
from .multiscale import MultiScaleFusion
from .bbga import BBGA
from .detector import DetectionHead
from .feature_refinement import FeatureRefinement

class MultiScaleAneurysmDetector(nn.Module):

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

        self.fusion = MultiScaleFusion(
            channels=768
        )

        self.bbga = BBGA(
            channels=768
        )

        self.refinement = FeatureRefinement(
            channels=768
        )

        self.head = DetectionHead(
            in_channels=768,
            num_classes=num_classes,
        )

    def encode(self, image):

        features, _ = self.encoder(
            image
        )

        return features

    def forward(
        self,
        coarse,
        medium,
        fine,
        center=None,
        vessel_mask=None,
    ):

        coarse_features = self.encode(
            coarse
        )

        medium_features = self.encode(
            medium
        )

        fine_features = self.encode(
            fine
        )

        features = self.fusion(
            coarse_features,
            medium_features,
            fine_features,
        )
        features,vessel_attention = self.refinement(features,vessel_mask)
        bbga_attention = None

        if center is not None:

            from ..data.bbga import (
                gaussian_guide,
            )

            guide = gaussian_guide(
                center[:, :2],
                features.shape[-2],
                features.shape[-1],
            )

            features, bbga_attention = (
                self.bbga(
                    features,
                    guide,
                )
            )

        predictions = self.head(
            features
        )

        predictions[
            "features"
        ] = features

        predictions[
            "bbga_attention"
        ] = bbga_attention

        predictions[
            "vessel_attention"
        ] = vessel_attention

        return predictions

    def freeze_encoder(self):
        self.encoder.freeze_encoder()

    def unfreeze_last_stage(self):
        self.encoder.unfreeze_last_stage()

    def unfreeze_encoder(self):
        self.encoder.unfreeze_encoder()