import numpy as np
import cv2

def roi_to_25d(
    roi,
    center_z=None
):
    """
    Convert 3D ROI to 2.5D representation.

    Output:
        [3,H,W]
    """

    depth = roi.shape[0]

    if center_z is None:
        center_z = depth // 2

    center_z = int(
        np.clip(
            center_z,
            0,
            depth - 1
        )
    )

    z0 = max(
        0,
        center_z - 1
    )

    z1 = center_z

    z2 = min(
        depth - 1,
        center_z + 1
    )

    return np.stack(
        [
            roi[z0],
            roi[z1],
            roi[z2]
        ],
        axis=0
    )
def clip_center(
    center,
    shape
):
    """
    center: [x, y, z]
    shape:  [z, y, x]
    """

    x, y, z = center

    depth, height, width = shape

    return np.array(
        [
            np.clip(x, 0, width - 1),
            np.clip(y, 0, height - 1),
            np.clip(z, 0, depth - 1),
        ],
        dtype=np.float32
    )


def extract_roi(
    volume,
    center,
    roi_size
):
    """
    Extract a 3D ROI centered at [x,y,z].

    roi_size:
        (depth, height, width)

    Returns:
        roi
        actual_center
        slices
    """

    center = clip_center(
        center,
        volume.shape
    )

    x, y, z = np.round(
        center
    ).astype(int)

    rd, rh, rw = roi_size

    z0 = z - rd // 2
    y0 = y - rh // 2
    x0 = x - rw // 2

    z1 = z0 + rd
    y1 = y0 + rh
    x1 = x0 + rw

    pad_before = (
        max(0, -z0),
        max(0, -y0),
        max(0, -x0)
    )

    pad_after = (
        max(0, z1 - volume.shape[0]),
        max(0, y1 - volume.shape[1]),
        max(0, x1 - volume.shape[2])
    )

    z0_clip = max(0, z0)
    y0_clip = max(0, y0)
    x0_clip = max(0, x0)

    z1_clip = min(
        volume.shape[0],
        z1
    )

    y1_clip = min(
        volume.shape[1],
        y1
    )

    x1_clip = min(
        volume.shape[2],
        x1
    )

    roi = volume[
        z0_clip:z1_clip,
        y0_clip:y1_clip,
        x0_clip:x1_clip
    ]

    if any(
        p > 0
        for p in pad_before + pad_after
    ):

        roi = np.pad(
            roi,
            (
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="constant"
        )

    return roi


def global_to_roi_center(
    center,
    roi_origin
):
    """
    Convert global [x,y,z] voxel coordinate
    to ROI-relative coordinate.
    """

    center = np.asarray(
        center,
        dtype=np.float32
    )

    origin = np.asarray(
        roi_origin,
        dtype=np.float32
    )

    return center - origin


def normalize_center(
    center,
    roi_size
):
    """
    Convert ROI-relative [x,y,z]
    to normalized [0,1].
    """

    x, y, z = center

    depth, height, width = roi_size

    return np.array(
        [
            x / max(width - 1, 1),
            y / max(height - 1, 1),
            z / max(depth - 1, 1),
        ],
        dtype=np.float32
    )

def resize_25d(
    image,
    size=224
):

    channels = []

    for channel in image:

        low, high = np.percentile(
            channel,
            [1, 99]
        )

        if high <= low:
            channel = np.zeros_like(
                channel,
                dtype=np.float32
            )
        else:
            channel = np.clip(
                channel,
                low,
                high
            )

            channel = (
                channel - low
            ) / (
                high - low
            )

        channel = cv2.resize(
            channel.astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_LINEAR
        )

        channels.append(
            channel
        )

    return np.stack(
        channels,
        axis=0
    ).astype(np.float32)