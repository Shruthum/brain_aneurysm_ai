from pathlib import Path

import numpy as np
import pandas as pd

from .dicom import DICOMSeries
from aneurysm_ai.constants import ANATOMY_TO_INDEX

class AneurysmAnnotationBuilder:

    def __init__(
        self,
        train_csv,
        localizer_csv,
        series_dir
    ):

        self.train_df = pd.read_csv(
            train_csv
        )

        self.localizer_df = pd.read_csv(
            localizer_csv
        )

        self.series_dir = Path(
            series_dir
        )

    def build(self):

        records = []

        for _, row in self.localizer_df.iterrows():

            series_uid = str(
                row["SeriesInstanceUID"]
            )

            sop_uid = str(
                row["SOPInstanceUID"]
            )

            location = str(
                row["location"]
            )

            coordinates = self._parse_coordinates(
                row["coordinates"]
            )

            series_path = (
                self.series_dir /
                series_uid
            )

            if not series_path.exists():
                continue

            dicom = DICOMSeries(
                series_path
            )

            z = dicom.get_z_index(
                sop_uid
            )

            if z is None:
                continue

            world = dicom.voxel_to_world(
                [
                    coordinates[0],
                    coordinates[1],
                    z
                ]
            )

            voxel = dicom.world_to_voxel(
                world
            )

            records.append(
                {
                    "series_uid": series_uid,
                    "sop_uid": sop_uid,
                    "location": location,
                    "x": float(voxel[0]),
                    "y": float(voxel[1]),
                    "z": float(voxel[2]),
                }
            )

        return pd.DataFrame(
            records
        )

    @staticmethod
    def _parse_coordinates(value):

        if isinstance(
            value,
            (list, tuple)
        ):
            return np.asarray(
                value,
                dtype=np.float32
            )

        text = str(value).strip()

        text = text.replace(
            "[",
            ""
        ).replace(
            "]",
            ""
        )

        values = [
            float(x.strip())
            for x in text.split(",")
        ]

        if len(values) != 2:
            raise ValueError(
                f"Expected x,y coordinates: {value}"
            )

        return np.asarray(
            values,
            dtype=np.float32
        )
def validate_annotations(
    annotations,
    series_dir
):

    errors = []

    for _, row in annotations.iterrows():

        uid = str(
            row["series_uid"]
        )

        series_path = (
            Path(series_dir) / uid
        )

        try:

            dicom = DICOMSeries(
                series_path
            )

            point = np.array(
                [
                    row["x"],
                    row["y"],
                    row["z"]
                ]
            )

            if not (
                0 <= point[0] < dicom.columns
                and
                0 <= point[1] < dicom.rows
                and
                0 <= point[2] < dicom.num_slices
            ):
                errors.append(
                    {
                        "series_uid": uid,
                        "point": point.tolist()
                    }
                )

        except Exception as exc:

            errors.append(
                {
                    "series_uid": uid,
                    "error": str(exc)
                }
            )

    return errors

def build_positive_samples(
    annotations,
    series_dir,
    roi_size
):

    samples = []

    cache = {}

    for _, row in annotations.iterrows():

        uid = str(
            row["series_uid"]
        )

        if uid not in cache:

            dicom = DICOMSeries(
                Path(series_dir) / uid
            )

            cache[uid] = dicom

        dicom = cache[uid]

        center = np.array(
            [
                row["x"],
                row["y"],
                row["z"]
            ],
            dtype=np.float32
        )
        anatomy_id = ANATOMY_TO_INDEX[row["location"]]

        samples.append(
            {
                "series_uid": uid,
                "location": row["location"],
                "anatomy_id": anatomy_id,
                "center": center,
                "roi_size": roi_size,
            }
        )

    return samples

def build_detection_target(
    sample,
    roi_shape,
):
    """
    Convert an aneurysm ROI sample into detector targets.

    roi_shape = (D, H, W)
    center is represented as normalized (x, y, z).
    """

    center = np.asarray(
        sample["center"],
        dtype=np.float32,
    )

    roi_center = np.asarray(
        roi_shape[::-1],
        dtype=np.float32,
    ) / 2.0

    # center is currently expressed in global volume coordinates.
    # Convert to ROI-relative coordinates.
    roi_start = center.copy() - roi_center

    relative_center = center - roi_start

    normalized_center = np.array(
        [
            relative_center[0] / roi_shape[2],
            relative_center[1] / roi_shape[1],
            relative_center[2] / roi_shape[0],
        ],
        dtype=np.float32,
    )

    anatomy_name = sample["location"]

    anatomy_id = ANATOMY_TO_INDEX[anatomy_name]

    return {
        "objectness": 1.0,
        "center": normalized_center,
        "anatomy": anatomy_id,
    }