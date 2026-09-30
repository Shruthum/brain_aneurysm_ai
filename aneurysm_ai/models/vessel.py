import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SwinModel,SwinConfig

class SwinVesselEncoder(nn.Module):

    def __init__(
        self,
        model_name="microsoft/swin-tiny-patch4-window7-224",
        pretrained=True
    ):
        super().__init__()

        if pretrained:
            self.encoder = SwinModel.from_pretrained(
                model_name
            )
        else:

            config = SwinConfig.from_pretrained(
                model_name
            )

            self.encoder = SwinModel(
                config
            )

        self.out_channels = (
            self.encoder.config.hidden_size
        )

    def forward(self, x):

        outputs = self.encoder(
            pixel_values=x
        )

        features = outputs.last_hidden_state

        b, n, c = features.shape

        h = w = int(n ** 0.5)

        if h * w != n:
            raise RuntimeError(
                f"Cannot reshape {n} tokens "
                "into a square feature map."
            )

        features = features.transpose(
            1, 2
        )

        features = features.reshape(
            b,
            c,
            h,
            w
        )

        return features


class VesselDecoder(nn.Module):

    def __init__(
        self,
        in_channels=768,
        num_classes=14
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                384,
                3,
                padding=1
            ),
            nn.BatchNorm2d(384),
            nn.GELU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                384,
                192,
                3,
                padding=1
            ),
            nn.BatchNorm2d(192),
            nn.GELU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                192,
                96,
                3,
                padding=1
            ),
            nn.BatchNorm2d(96),
            nn.GELU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(
                96,
                48,
                3,
                padding=1
            ),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        self.head = nn.Conv2d(
            48,
            num_classes,
            1
        )

    def forward(self, x):

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.block1(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.block2(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.block3(x)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.block4(x)

        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        return self.head(x)


class VesselModel(nn.Module):

    def __init__(
        self,
        model_name="microsoft/swin-tiny-patch4-window7-224",
        num_classes=14
    ):
        super().__init__()

        self.encoder = SwinVesselEncoder(
            model_name=model_name,
            pretrained=True
        )

        self.decoder = VesselDecoder(
            in_channels=self.encoder.out_channels,
            num_classes=num_classes
        )

    def forward(self, x):

        features = self.encoder(x)

        logits = self.decoder(
            features
        )

        return logits

    def encode(self, x):

        return self.encoder(x)