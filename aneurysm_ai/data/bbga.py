import torch


def gaussian_guide(
    center_xy,
    height,
    width,
    sigma=0.12,
):
    """
    Create a normalized Gaussian spatial guide.

    center_xy:
        normalized [x, y]
    """

    device = center_xy.device

    y = torch.linspace(
        0.0,
        1.0,
        height,
        device=device,
    )

    x = torch.linspace(
        0.0,
        1.0,
        width,
        device=device,
    )

    yy, xx = torch.meshgrid(
        y,
        x,
        indexing="ij",
    )

    cx = center_xy[:, 0].view(-1, 1, 1)
    cy = center_xy[:, 1].view(-1, 1, 1)

    distance = (
        (xx.unsqueeze(0) - cx).pow(2)
        +
        (yy.unsqueeze(0) - cy).pow(2)
    )

    guide = torch.exp(
        -distance / (2.0 * sigma * sigma)
    )

    return guide.unsqueeze(1)