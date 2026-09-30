from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from .nifti import NiftiSegmentation
from aneurysm_ai.data import DICOMSeries
import json

class NiftiDicomMapper:

    def __init__(self, dicom_series):
        self.dicom = dicom_series

    def _dicom_world_grid(self, z_index):
        """
        Generate world coordinates for one DICOM slice.

        Output:
            [rows, columns, 3]
        """

        rows = self.dicom.rows
        columns = self.dicom.columns

        y, x = np.meshgrid(
            np.arange(rows, dtype=np.float64),
            np.arange(columns, dtype=np.float64),
            indexing="ij"
        )

        origin = self.dicom.positions[z_index]

        world = (
            origin
            + x[..., None]
            * self.dicom.col_spacing
            * self.dicom.col_direction
            + y[..., None]
            * self.dicom.row_spacing
            * self.dicom.row_direction
        )

        return world

    def resample(
        self,
        nifti_path,
        chunk_size=8
    ):
        """
        Resample categorical NIfTI segmentation
        onto the DICOM voxel grid.

        Returns:
            numpy array [Z, Y, X]
        """

        nifti = NiftiSegmentation(nifti_path)

        nifti.validate_labels()

        depth = self.dicom.num_slices
        rows = self.dicom.rows
        columns = self.dicom.columns

        output = np.zeros(
            (depth, rows, columns),
            dtype=np.uint8
        )

        for start in range(0, depth, chunk_size):

            end = min(start + chunk_size, depth)

            world_slices = []

            for z in range(start, end):
                world_slices.append(
                    self._dicom_world_grid(z)
                )

            world = np.stack(world_slices, axis=0)

            nifti_voxel = nifti.world_to_voxel(world)

            coords = np.stack(
                [
                    nifti_voxel[..., 0],
                    nifti_voxel[..., 1],
                    nifti_voxel[..., 2]
                ],
                axis=0
            )

            sampled = map_coordinates(
                nifti.data,
                coords,
                order=0,
                mode="constant",
                cval=0,
                prefilter=False
            )

            output[start:end] = sampled.astype(np.uint8)

        return output

    def inspect(self, nifti_path):

        nifti = NiftiSegmentation(nifti_path)

        labels = nifti.validate_labels()

        center = (
            np.asarray(nifti.shape, dtype=np.float64) - 1
        ) / 2.0

        world = nifti.voxel_to_world(center)

        dicom_voxel = self.dicom.world_to_voxel(world)

        return {
            "nifti_shape": nifti.shape,
            "nifti_spacing": nifti.spacing,
            "nifti_labels": labels.tolist(),
            "dicom_shape": self.dicom.shape,
            "dicom_spacing": self.dicom.spacing_xyz.tolist(),
            "center_world": world.tolist(),
            "center_dicom_voxel": dicom_voxel.tolist(),
        }

def save_resampled_segmentation(
    segmentation,
    output_path
):
    """
    Save the DICOM-aligned segmentation as NumPy.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(output_path, segmentation)

def verify_nifti_dicom_pair(
    series_uid,
    series_dir,
    nifti_path,
):
    series_uid = str(series_uid)
    nifti_path = Path(nifti_path)

    dicom_path = Path(series_dir) / series_uid

    if not dicom_path.exists():
        return False, "DICOM series does not exist"

    if not nifti_path.exists():
        return False, "NIfTI file does not exist"

    try:
        dicom = DICOMSeries(dicom_path)
        nifti = NiftiSegmentation(nifti_path)

        labels = nifti.validate_labels()

        if len(labels) <= 1:
            return False, "Segmentation contains no foreground labels"

        mapper = NiftiDicomMapper(dicom)

        # Check physical center transformation.
        center = (
            np.asarray(nifti.shape, dtype=np.float64) - 1
        ) / 2.0

        world = nifti.voxel_to_world(center)

        dicom_voxel = dicom.world_to_voxel(world)

        finite = np.all(
            np.isfinite(dicom_voxel)
        )

        if not finite:
            return False, "Invalid coordinate transformation"

        # Perform actual resampling.
        segmentation = mapper.resample(
            nifti_path
        )

        if segmentation.shape != dicom.shape:
            return False, (
                f"Shape mismatch: "
                f"{segmentation.shape} != {dicom.shape}"
            )

        aligned_labels = np.unique(
            segmentation
        )

        if not set(aligned_labels).issubset(
            set(range(14))
        ):
            return False, "Invalid aligned labels"

        if not np.any(segmentation > 0):
            return False, (
                "No foreground after DICOM alignment"
            )

        return True, {
            "series_uid": series_uid,
            "nifti_path": str(nifti_path),
            "dicom_shape": list(dicom.shape),
            "nifti_shape": list(nifti.shape),
            "labels": labels.tolist(),
            "aligned_labels": aligned_labels.tolist(),
        }

    except Exception as exc:
        return False, str(exc)


def save_verified_mapping(
    mapping,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            mapping,
            f,
            indent=2
        )


def load_verified_mapping(
    path
):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)

def build_verified_mapping(
    series_dir,
    segmentation_dir,
    max_pairs=None,
):
    """
    Discover NIfTI segmentations and verify possible
    SeriesInstanceUID matches.

    Returns:
        verified_mapping:
        [
            {
                "series_uid": "...",
                "nifti_path": "...",
            }
        ]
    """

    series_dir = Path(series_dir)
    segmentation_dir = Path(segmentation_dir)

    nifti_files = sorted(
        list(segmentation_dir.glob("*.nii")) +
        list(segmentation_dir.glob("*.nii.gz"))
    )

    verified_mapping = []

    for nifti_path in nifti_files:

        filename = nifti_path.name

        candidates = []

        for series_path in series_dir.iterdir():

            if not series_path.is_dir():
                continue

            uid = series_path.name

            if uid in filename:
                candidates.append(uid)

        if len(candidates) != 1:
            continue

        series_uid = candidates[0]

        # Geometry-level verification
        try:
            dicom = DICOMSeries(
                series_dir / series_uid
            )

            mapper = NiftiDicomMapper(
                dicom
            )

            info = mapper.inspect(
                nifti_path
            )

            # Force actual resampling validation.
            segmentation = mapper.resample(
                nifti_path,
                chunk_size=8
            )

            if segmentation.shape != dicom.shape:
                continue

            labels = np.unique(segmentation)

            if not set(labels).issubset(
                set(range(14))
            ):
                continue

            verified_mapping.append(
                {
                    "series_uid": series_uid,
                    "nifti_path": str(nifti_path),
                    "nifti_shape": info["nifti_shape"],
                    "dicom_shape": info["dicom_shape"],
                    "labels": labels.tolist(),
                }
            )

        except Exception:
            continue

        if (
            max_pairs is not None
            and len(verified_mapping) >= max_pairs
        ):
            break

    return verified_mapping