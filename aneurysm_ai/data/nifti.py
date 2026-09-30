from pathlib import Path

import nibabel as nib
import numpy as np

VALID_LABELS = set(range(14))
class NiftiSegmentation:
    def __init__(self, path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.image = nib.load(str(self.path))
        self.data = np.asarray(self.image.dataobj)

        if self.data.ndim != 3:
            raise ValueError(
                f"Expected 3D NIfTI, got shape {self.data.shape}"
            )

        self.affine = np.asarray(
            self.image.affine,
            dtype=np.float64
        )

        self.inverse_affine = np.linalg.inv(self.affine)

    @property
    def shape(self):
        return self.data.shape

    @property
    def spacing(self):
        return np.asarray(
            self.image.header.get_zooms()[:3],
            dtype=np.float64
        )

    def voxel_to_world(self, voxel):
        voxel = np.asarray(voxel, dtype=np.float64)

        if voxel.shape[-1] != 3:
            raise ValueError("voxel must have 3 coordinates")

        original_shape = voxel.shape

        flat = voxel.reshape(-1, 3)

        homogeneous = np.concatenate(
            [
                flat,
                np.ones((len(flat), 1))
            ],
            axis=1
        )

        world = homogeneous @ self.affine.T

        return world[:, :3].reshape(original_shape)

    def world_to_voxel(self, world):
        world = np.asarray(world, dtype=np.float64)

        if world.shape[-1] != 3:
            raise ValueError("world must have 3 coordinates")

        original_shape = world.shape

        flat = world.reshape(-1, 3)

        homogeneous = np.concatenate(
            [
                flat,
                np.ones((len(flat), 1))
            ],
            axis=1
        )

        voxel = homogeneous @ self.inverse_affine.T

        return voxel[:, :3].reshape(original_shape)

    def validate_labels(self):
        labels = np.unique(self.data)

        invalid = [
            int(x)
            for x in labels
            if int(x) not in VALID_LABELS
        ]

        if invalid:
            raise ValueError(
                f"Invalid segmentation labels in {self.path}: "
                f"{invalid}"
            )

        return labels