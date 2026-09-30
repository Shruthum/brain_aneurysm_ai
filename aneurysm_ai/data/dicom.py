from pathlib import Path
import numpy as np
import pydicom


class DICOMSeries:

    def __init__(self, series_path):

        self.series_path = Path(series_path)

        if not self.series_path.exists():
            raise FileNotFoundError(
                f"Series not found: {self.series_path}"
            )

        self.datasets = []
        self.sop_to_index = {}

        self._load_headers()
        self._sort_slices()
        self._build_geometry()

    # --------------------------------------------------
    # Load DICOM headers
    # --------------------------------------------------

    def _load_headers(self):

        files = [
            p for p in self.series_path.rglob("*")
            if p.is_file()
        ]

        for path in files:

            try:
                ds = pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True
                )

                if not hasattr(ds, "ImagePositionPatient"):
                    continue

                if not hasattr(ds, "ImageOrientationPatient"):
                    continue

                self.datasets.append({
                    "path": path,
                    "ds": ds,
                })

            except Exception:
                continue

        if not self.datasets:
            raise RuntimeError(
                f"No valid DICOM slices found in "
                f"{self.series_path}"
            )

    # --------------------------------------------------
    # Spatial sorting
    # --------------------------------------------------

    def _sort_slices(self):

        first_ds = self.datasets[0]["ds"]

        orientation = np.asarray(
            first_ds.ImageOrientationPatient,
            dtype=np.float64
        )

        row_direction = orientation[:3]
        col_direction = orientation[3:]

        self.row_direction = (
            row_direction /
            np.linalg.norm(row_direction)
        )

        self.col_direction = (
            col_direction /
            np.linalg.norm(col_direction)
        )

        self.slice_direction = np.cross(
            self.row_direction,
            self.col_direction
        )

        self.slice_direction /= np.linalg.norm(
            self.slice_direction
        )

        for item in self.datasets:

            position = np.asarray(
                item["ds"].ImagePositionPatient,
                dtype=np.float64
            )

            item["position"] = position

            item["slice_coordinate"] = float(
                np.dot(
                    position,
                    self.slice_direction
                )
            )

        self.datasets.sort(
            key=lambda x: x["slice_coordinate"]
        )

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    def _build_geometry(self):

        ds = self.datasets[0]["ds"]

        self.rows = int(ds.Rows)
        self.columns = int(ds.Columns)

        pixel_spacing = np.asarray(
            ds.PixelSpacing,
            dtype=np.float64
        )

        self.row_spacing = float(
            pixel_spacing[0]
        )

        self.col_spacing = float(
            pixel_spacing[1]
        )

        self.positions = np.stack([
            item["position"]
            for item in self.datasets
        ])

        coordinates = np.asarray([
            item["slice_coordinate"]
            for item in self.datasets
        ])

        if len(coordinates) > 1:

            differences = np.abs(
                np.diff(coordinates)
            )

            valid = differences[
                differences > 1e-5
            ]

            if len(valid):
                self.slice_spacing = float(
                    np.median(valid)
                )
            else:
                self.slice_spacing = float(
                    getattr(
                        ds,
                        "SliceThickness",
                        1.0
                    )
                )

        else:

            self.slice_spacing = float(
                getattr(
                    ds,
                    "SliceThickness",
                    1.0
                )
            )

        self.spacing_xyz = np.array([
            self.col_spacing,
            self.row_spacing,
            self.slice_spacing
        ])

        self.origin = self.positions[0]

        for index, item in enumerate(
            self.datasets
        ):

            sop_uid = str(
                item["ds"].SOPInstanceUID
            )

            self.sop_to_index[sop_uid] = index

    # --------------------------------------------------
    # Properties
    # --------------------------------------------------

    @property
    def num_slices(self):
        return len(self.datasets)

    @property
    def shape(self):
        return (
            self.num_slices,
            self.rows,
            self.columns
        )

    # --------------------------------------------------
    # Pixel data
    # --------------------------------------------------

    def _read_pixels(self, item):

        ds = pydicom.dcmread(
            str(item["path"])
        )

        image = ds.pixel_array.astype(
            np.float32
        )

        slope = float(
            getattr(ds, "RescaleSlope", 1.0)
        )

        intercept = float(
            getattr(ds, "RescaleIntercept", 0.0)
        )

        return image * slope + intercept

    def load_volume(self):

        volume = np.stack([
            self._read_pixels(item)
            for item in self.datasets
        ])

        return volume

    # --------------------------------------------------
    # SOP → slice index
    # --------------------------------------------------

    def get_z_index(self, sop_instance_uid):

        return self.sop_to_index.get(
            str(sop_instance_uid)
        )

    # --------------------------------------------------
    # DICOM voxel → physical coordinates
    #
    # voxel = [x, y, z]
    # x = column
    # y = row
    # z = slice
    # --------------------------------------------------

    def voxel_to_world(self, voxel):

        x, y, z = np.asarray(
            voxel,
            dtype=np.float64
        )

        point = (
            self.origin
            + x * self.col_spacing * self.col_direction
            + y * self.row_spacing * self.row_direction
            + z * self.slice_spacing * self.slice_direction
        )

        return point

    # --------------------------------------------------
    # Physical coordinates → voxel
    # --------------------------------------------------

    def world_to_voxel(self, world):

        world = np.asarray(
            world,
            dtype=np.float64
        )

        delta = world - self.origin

        x = np.dot(
            delta,
            self.col_direction
        ) / self.col_spacing

        y = np.dot(
            delta,
            self.row_direction
        ) / self.row_spacing

        z = np.dot(
            delta,
            self.slice_direction
        ) / self.slice_spacing

        return np.array([
            x,
            y,
            z
        ])