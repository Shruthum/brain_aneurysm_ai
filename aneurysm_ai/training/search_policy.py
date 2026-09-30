import torch
import numpy as np

from ..data.roi import (
    extract_roi,
    roi_to_25d,
    resize_25d,
)


class DetectorSearchAdapter:

    def __init__(
        self,
        detector,
        volume,
        roi_sizes,
        image_size,
        device,
    ):
        self.detector = detector
        self.volume = volume
        self.roi_sizes = roi_sizes
        self.image_size = image_size
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        position,
    ):

        images = {}

        for name, roi_size in (
            self.roi_sizes.items()
        ):

            roi = extract_roi(
                self.volume,
                position,
                roi_size,
            )

            image = roi_to_25d(
                roi
            )

            image = resize_25d(
                image,
                self.image_size,
            )

            images[name] = torch.from_numpy(
                image
            ).float().unsqueeze(0).to(
                self.device
            )

        output = self.detector(
            images["coarse"],
            images["medium"],
            images["fine"],
            center=None,
            vessel_mask=None,
        )

        confidence = torch.sigmoid(
            output["objectness"]
        )

        return float(
            confidence.item()
        )
    def _extract_multiscale(
        self,
        position,
        scale=1.0,
    ):
        """
        Extract coarse, medium and fine 2.5D ROIs
        around the current PPO search position.

        position:
            [x, y, z] in global voxel coordinates.

        Returns:
            Tensor [3, 3, 224, 224]
            = [scales, channels, H, W]
        """

        position = np.asarray(
            position,
            dtype=np.float32,
        )

        images = []

        for base_size in self.roi_sizes:

            # Apply PPO zoom action.
            size = tuple(
                max(4, int(round(v * scale)))
                for v in base_size
            )

            roi = extract_roi(
                self.volume,
                center=position,
                roi_size=size,
            )

            image = roi_to_25d(roi)

            image = resize_25d(
                image,
                size=self.image_size,
            )

            image = torch.from_numpy(
                image
            ).float()

            images.append(image)

        return torch.stack(images, dim=0)

    @torch.no_grad()
    def confidence(
        self,
        position,
        scale=1.0,
    ):
        """
        Run the detector at the current
        PPO search position.
        """

        images = self._extract_multiscale(
            position,
            scale,
        )

        images = images.to(self.device)

        self.detector.eval()

        predictions = self.detector(
            images
        )

        confidence = torch.sigmoid(
            predictions["objectness"]
        )

        return float(
            confidence.mean().item()
        )