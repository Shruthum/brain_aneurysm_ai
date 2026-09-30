import torch
import torch.nn as nn
class VesselPrior(nn.Module):

    def __init__(
        self,
        vessel_model,
    ):
        super().__init__()

        self.vessel_model = vessel_model

    def forward(self, image):

        logits = self.vessel_model(
            image
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )

        # Background = class 0
        vessel_probability = (
            1.0 - probability[:, 0:1]
        )

        return vessel_probability