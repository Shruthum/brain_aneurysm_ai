import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom import DICOMSeries
from .roi import (
    extract_roi,
    roi_to_25d,
    resize_25d,
)


class MultiScaleAneurysmDataset(Dataset):

    def __init__(
        self,
        samples,
        series_dir,
        roi_sizes,
        image_size=224,
    ):
        self.samples = samples
        self.series_dir = series_dir
        self.roi_sizes = roi_sizes
        self.image_size = image_size

        self._volumes = {}

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, uid):

        if uid not in self._volumes:

            series = DICOMSeries(
                self.series_dir / str(uid)
            )

            self._volumes[uid] = (
                series.load_volume(),
                series.shape,
            )

        return self._volumes[uid]

    def __getitem__(self, index):

        sample = self.samples[index]

        uid = sample["SeriesInstanceUID"]

        volume, _ = self._load_volume(uid)

        center = np.asarray(
            sample["center"],
            dtype=np.float32,
        )

        outputs = {}

        for name, size in self.roi_sizes.items():

            roi = extract_roi(
                volume,
                center,
                size,
            )

            image = roi_to_25d(roi)

            image = resize_25d(
                image,
                self.image_size,
            )

            outputs[name] = torch.from_numpy(
                image
            ).float()

        outputs["center"] = torch.from_numpy(
            center
        ).float()

        outputs["anatomy"] = torch.tensor(
            sample["anatomy_id"],
            dtype=torch.long,
        )

        outputs["objectness"] = torch.tensor(
            1.0,
            dtype=torch.float32,
        )

        outputs["series_uid"] = uid

        return outputs