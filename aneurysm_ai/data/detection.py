import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .dicom import DICOMSeries
from .roi import (
    extract_roi,
    roi_to_25d,
    resize_25d,
)
class AneurysmROIDataset(Dataset):

    def __init__(
        self,
        samples,
        series_dir,
        roi_size=(24, 96, 96),
        image_size=224,
        positive_shift=12,
        negative_samples=1,
        seed=42,
    ):
        self.samples = samples
        self.series_dir = series_dir
        self.roi_size = tuple(roi_size)
        self.image_size = image_size
        self.positive_shift = positive_shift
        self.negative_samples = negative_samples
        self.rng = random.Random(seed)
        self.items = self._build_items()
        self._volumes = {}

    def _build_items(self):

        items = []

        for sample in self.samples:

            # Original positive ROI
            items.append({
                "sample": sample,
                "type": "positive",
                "shift": (0, 0, 0),
            })

            # Shifted positive ROIs
            for _ in range(2):

                dz = self.rng.randint(
                    -self.positive_shift,
                    self.positive_shift,
                )

                dy = self.rng.randint(
                    -self.positive_shift,
                    self.positive_shift,
                )

                dx = self.rng.randint(
                    -self.positive_shift,
                    self.positive_shift,
                )

                items.append({
                    "sample": sample,
                    "type": "shifted_positive",
                    "shift": (dx, dy, dz),
                })

            # Negative candidates
            for _ in range(self.negative_samples):

                items.append({
                    "sample": sample,
                    "type": "negative",
                    "shift": None,
                })

        return items

    def __len__(self):
        return len(self.items)

    def _load_volume(self, series_uid):

        if series_uid not in self._volumes:

            series_path = (
                self.series_dir /
                str(series_uid)
            )

            series = DICOMSeries(series_path)

            self._volumes[series_uid] = (
                series.load_volume(),
                series.shape,
            )

        return self._volumes[series_uid]

    def _random_negative_center(self, shape):
        z, y, x = shape
        return np.array([
            self.rng.uniform(0, x - 1),
            self.rng.uniform(0, y - 1),
            self.rng.uniform(0, z - 1),
        ], dtype=np.float32)
    def _valid_negative(
        self,
        candidate,
        centers,
        min_distance,
    ):
        for center in centers:
            distance = np.linalg.norm(
                candidate - center
            )

            if distance < min_distance:
                return False

        return True
    
    def _center_inside_roi(
        self,
        aneurysm_center,
        roi_center,
    ):
        roi_x = self.roi_size[2]
        roi_y = self.roi_size[1]
        roi_z = self.roi_size[0]

        roi_start = roi_center - np.array(
            [
                roi_x / 2.0,
                roi_y / 2.0,
                roi_z / 2.0,
            ],
            dtype=np.float32,
        )

        relative = (
            aneurysm_center - roi_start
        )

        return np.array(
            [
                relative[0] / roi_x,
                relative[1] / roi_y,
                relative[2] / roi_z,
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index):

        item = self.items[index]
        sample = item["sample"]

        series_uid = sample["SeriesInstanceUID"]

        # --------------------------------------------------
        # Load volume
        # --------------------------------------------------

        volume, shape = self._load_volume(series_uid)

        aneurysm_center = np.asarray(
            sample["center"],
            dtype=np.float32,
        )

        all_centers = sample.get(
            "all_centers",
            [aneurysm_center],
        )

        all_centers = np.asarray(
            all_centers,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Select ROI center
        # --------------------------------------------------

        if item["type"] == "positive":

            # ROI centered directly on aneurysm
            roi_center = aneurysm_center.copy()

        elif item["type"] == "shifted_positive":

            # Shift ROI while keeping aneurysm inside
            dx, dy, dz = item["shift"]

            roi_center = (
                aneurysm_center
                + np.array(
                    [dx, dy, dz],
                    dtype=np.float32,
                )
            )

        else:

            # --------------------------------------------------
            # Generate a true negative ROI
            # --------------------------------------------------

            min_distance = max(self.roi_size)

            roi_center = None

            for _ in range(50):

                candidate = self._random_negative_center(
                    shape
                )

                if self._valid_negative(
                    candidate,
                    all_centers,
                    min_distance,
                ):
                    roi_center = candidate
                    break

            # Fallback if a valid random location
            # could not be found.
            if roi_center is None:

                roi_center = self._random_negative_center(
                    shape
                )

        # --------------------------------------------------
        # Calculate position of the selected aneurysm
        # inside the ROI
        # --------------------------------------------------

        target_center = self._center_inside_roi(
            aneurysm_center,
            roi_center,
        )

        # --------------------------------------------------
        # Determine whether ANY aneurysm is inside ROI
        # --------------------------------------------------

        positive_center = None

        for center in all_centers:

            relative = self._center_inside_roi(
                center,
                roi_center,
            )

            if (
                np.all(relative >= 0.0)
                and np.all(relative <= 1.0)
            ):
                positive_center = relative
                break

        # --------------------------------------------------
        # Objectness
        # --------------------------------------------------

        if positive_center is not None:

            objectness = 1

            # If another aneurysm was found inside the ROI,
            # use that aneurysm's relative center.
            #
            # For the current sample, the selected target
            # remains the annotated aneurysm whenever it is
            # inside the ROI.

            selected_relative = target_center

            if not (
                np.all(selected_relative >= 0.0)
                and np.all(selected_relative <= 1.0)
            ):
                selected_relative = positive_center

            target_center = selected_relative

        else:

            objectness = 0

            # Center target is ignored by the loss for
            # negative samples.
            target_center = np.zeros(
                3,
                dtype=np.float32,
            )

        # --------------------------------------------------
        # Extract 3D ROI
        # --------------------------------------------------

        roi = extract_roi(
            volume,
            roi_center,
            self.roi_size,
        )

        # --------------------------------------------------
        # Convert 3D ROI -> 2.5D
        # --------------------------------------------------

        image = roi_to_25d(
            roi
        )

        # --------------------------------------------------
        # Resize to Swin input
        # --------------------------------------------------

        image = resize_25d(
            image,
            self.image_size,
        )

        # --------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        objectness = torch.tensor(
            objectness,
            dtype=torch.float32,
        )

        target_center = torch.from_numpy(
            target_center
        ).float()

        anatomy = torch.tensor(
            sample["anatomy_id"],
            dtype=torch.long,
        )

        # --------------------------------------------------
        # Return complete training sample
        # --------------------------------------------------

        return {
            "image": image,

            "objectness": objectness,

            "center": target_center,

            "anatomy": anatomy,

            "series_uid": series_uid,

            "center_global": torch.from_numpy(
                aneurysm_center
            ).float(),

            "roi_center": torch.from_numpy(
                roi_center
            ).float(),

            "roi_type": item["type"],
        }