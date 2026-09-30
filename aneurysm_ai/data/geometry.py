import numpy as np


def voxel_distance_mm(
    point_a,
    point_b,
    spacing_xyz
):

    point_a = np.asarray(
        point_a,
        dtype=np.float64
    )

    point_b = np.asarray(
        point_b,
        dtype=np.float64
    )

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64
    )

    delta = (
        point_a - point_b
    ) * spacing_xyz

    return float(
        np.linalg.norm(delta)
    )


def clip_voxel(
    voxel,
    shape
):

    x, y, z = voxel

    depth, height, width = shape

    return np.array([
        np.clip(x, 0, width - 1),
        np.clip(y, 0, height - 1),
        np.clip(z, 0, depth - 1)
    ])


def inside_volume(
    voxel,
    shape
):

    x, y, z = voxel

    depth, height, width = shape

    return (
        0 <= x < width
        and
        0 <= y < height
        and
        0 <= z < depth
    )