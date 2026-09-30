from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom import DICOMSeries
from .mapping import NiftiDicomMapper


def normalize_slice(image):
    image = image.astype(np.float32)
    low, high = np.percentile(image,[1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip(
        image,
        low,
        high
    )
    image = ( image - low )/(high - low)
    return image.astype(np.float32)
def resize_image(image, size):
    return cv2.resize(image,(size, size),interpolation=cv2.INTER_LINEAR)

def resize_mask(mask, size):
    return cv2.resize(mask.astype(np.uint8),(size, size),interpolation=cv2.INTER_NEAREST)


class VesselDataset(Dataset):
    def __init__(self,samples,image_size=224,cache_dir=None):
        self.samples = samples
        self.image_size = image_size

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else None
        )

        if self.cache_dir:
            self.cache_dir.mkdir(
                parents=True,
                exist_ok=True
            )

    def __len__(self):
        return len(self.samples)

    def _cache_path(self, series_uid):
        if self.cache_dir is None:
            return None

        return (
            self.cache_dir /
            f"{series_uid}.npz"
        )

    def _load_series(self, series_uid, nifti_path):

        cache_path = self._cache_path(
            series_uid
        )

        if cache_path is not None:
            if cache_path.exists():

                data = np.load(
                    cache_path
                )

                return (
                    data["volume"],
                    data["segmentation"]
                )

        dicom = DICOMSeries(
            self._series_path(series_uid)
        )

        volume = dicom.load_volume()

        mapper = NiftiDicomMapper(
            dicom
        )

        segmentation = mapper.resample(
            nifti_path
        )

        if cache_path is not None:

            np.savez_compressed(
                cache_path,
                volume=volume.astype(
                    np.float32
                ),
                segmentation=segmentation.astype(
                    np.uint8
                )
            )

        return volume, segmentation

    def _series_path(self, series_uid):
        raise NotImplementedError(
            "Set series path through "
            "VesselDatasetFactory."
        )

    def __getitem__(self, index):

        sample = self.samples[index]

        volume, segmentation = self._load_series(
            sample["series_uid"],
            sample["nifti_path"]
        )

        z = int(sample["z"])

        depth = volume.shape[0]

        z0 = max(0, z - 1)
        z1 = z
        z2 = min(depth - 1, z + 1)

        image = np.stack(
            [
                volume[z0],
                volume[z1],
                volume[z2]
            ],
            axis=0
        )

        channels = []

        for channel in image:

            channel = normalize_slice(
                channel
            )

            channel = resize_image(
                channel,
                self.image_size
            )

            channels.append(channel)

        image = np.stack(
            channels,
            axis=0
        )

        mask = resize_mask(
            segmentation[z],
            self.image_size
        )

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask.astype(np.int64)
        ).long()

        return {
            "image": image,
            "mask": mask,
            "series_uid": sample["series_uid"],
            "z": z,
        }

class VesselDatasetFactory:

    def __init__(
        self,
        series_dir,
        image_size=224,
        cache_dir=None
    ):
        self.series_dir = Path(
            series_dir
        )

        self.image_size = image_size
        self.cache_dir = cache_dir

    def create(self, samples):

        dataset = VesselDataset(
            samples=samples,
            image_size=self.image_size,
            cache_dir=self.cache_dir
        )

        dataset._series_path = (
            lambda uid:
            self.series_dir / str(uid)
        )

        return dataset

def build_vessel_samples(
    mapping,
    series_dir
):
    """
    mapping:
        list of dictionaries:

        {
            "series_uid": "...",
            "nifti_path": "..."
        }
    """

    samples = []

    series_dir = Path(
        series_dir
    )

    for item in mapping:

        series_uid = str(
            item["series_uid"]
        )

        nifti_path = Path(
            item["nifti_path"]
        )

        series_path = (
            series_dir / series_uid
        )

        if not series_path.exists():
            continue

        dicom = DICOMSeries(
            series_path
        )

        depth = dicom.num_slices

        for z in range(depth):

            samples.append(
                {
                    "series_uid": series_uid,
                    "nifti_path": str(
                        nifti_path
                    ),
                    "z": z,
                }
            )

    return samples
def split_series(
    mapping,
    val_fraction=0.15,
    seed=42
):
    rng = np.random.default_rng(
        seed
    )

    series = sorted(
        {
            str(x["series_uid"])
            for x in mapping
        }
    )

    rng.shuffle(series)

    n_val = max(
        1,
        int(len(series) * val_fraction)
    )

    val_series = set(
        series[:n_val]
    )

    train_mapping = []
    val_mapping = []

    for item in mapping:

        uid = str(
            item["series_uid"]
        )

        if uid in val_series:
            val_mapping.append(item)
        else:
            train_mapping.append(item)

    return (
        train_mapping,
        val_mapping
    )

def filter_vessel_samples(
    samples,
    factory,
    negative_ratio=1.0,
):
    grouped = {}

    for sample in samples:
        uid = sample["series_uid"]

        grouped.setdefault(
            uid,
            []
        ).append(sample)

    positive = []
    negative = []

    for uid, series_samples in grouped.items():

        first = series_samples[0]

        _, segmentation = (
            factory.create(
                [first]
            )._load_series(
                first["series_uid"],
                first["nifti_path"]
            )
        )

        for sample in series_samples:

            z = sample["z"]

            if np.any(
                segmentation[z] > 0
            ):
                positive.append(sample)
            else:
                negative.append(sample)

    rng = np.random.default_rng(42)

    max_negative = int(
        len(positive) *
        negative_ratio
    )

    if len(negative) > max_negative:

        indices = rng.choice(
            len(negative),
            size=max_negative,
            replace=False
        )

        negative = [
            negative[i]
            for i in indices
        ]

    return positive + negative