from pathlib import Path

import torch
import torch.nn as nn

from .vessel import SwinVesselEncoder
from .attention import VesselAwareFeatures


class AneurysmFeatureEncoder(nn.Module):

    def __init__(
        self,
        model_name,
        checkpoint_path,
        num_classes=13
    ):
        super().__init__()

        self.encoder = SwinVesselEncoder(
            model_name=model_name,
            pretrained=True
        )

        self.vessel_attention = (
            VesselAwareFeatures(
                channels=self.encoder.out_channels
            )
        )

        self._load_vessel_checkpoint(
            checkpoint_path
        )

    def _load_vessel_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        state = checkpoint["model"]

        encoder_state = {}

        for key, value in state.items():

            if key.startswith("encoder."):

                new_key = key[
                    len("encoder."):]

                encoder_state[
                    new_key
                ] = value

        if not encoder_state:
            raise RuntimeError(
                "No encoder weights found "
                "in vessel checkpoint."
            )

        result = self.encoder.load_state_dict(
            encoder_state,
            strict=True
        )

        if result.missing_keys:
            raise RuntimeError(
                f"Missing keys: "
                f"{result.missing_keys}"
            )

        if result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys: "
                f"{result.unexpected_keys}"
            )

    def forward(
        self,
        images,
        vessel_mask=None
    ):

        features = self.encoder(
            images
        )

        if vessel_mask is not None:

            features, attention = (
                self.vessel_attention(
                    features,
                    vessel_mask
                )
            )

        else:

            attention = None

        return {
            "features": features,
            "vessel_attention": attention
        }

    def freeze_encoder(self):

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def unfreeze_encoder(self):

        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

    def unfreeze_last_stage(self):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        layers = (
            self.encoder.encoder.encoder.layers
        )
        for parameter in layers[-1].parameters():
            parameter.requires_grad = True

    def trainable_parameters(self):
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]