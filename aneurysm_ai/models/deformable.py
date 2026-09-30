import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention2D(nn.Module):
    """
    Lightweight deformable attention.

    This is a compact research implementation and is
    not intended to reproduce the complete Deformable DETR
    multi-scale attention operator.
    """

    def __init__(
        self,
        channels=768,
        heads=8,
        points=4,
        offset_scale=0.15,
    ):
        super().__init__()

        if channels % heads != 0:
            raise ValueError(
                "channels must be divisible by heads"
            )

        self.channels = channels
        self.heads = heads
        self.points = points
        self.head_dim = channels // heads
        self.offset_scale = offset_scale

        self.offset = nn.Conv2d(
            channels,
            heads * points * 2,
            kernel_size=3,
            padding=1,
        )

        self.weight = nn.Conv2d(
            channels,
            heads * points,
            kernel_size=3,
            padding=1,
        )

        self.value = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.output = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.norm = nn.BatchNorm2d(channels)

    def _base_grid(
        self,
        batch,
        height,
        width,
        device,
        dtype,
    ):
        y = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )

        x = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        grid = torch.stack(
            [xx, yy],
            dim=-1,
        )

        return grid.unsqueeze(0).expand(
            batch,
            -1,
            -1,
            -1,
        )

    def forward(self, x):

        B, C, H, W = x.shape

        value = self.value(x)

        offsets = self.offset(x)

        weights = self.weight(x)

        offsets = offsets.view(
            B,
            self.heads,
            self.points,
            2,
            H,
            W,
        )

        weights = weights.view(
            B,
            self.heads,
            self.points,
            H,
            W,
        )

        weights = torch.softmax(
            weights,
            dim=2,
        )

        base_grid = self._base_grid(
            B,
            H,
            W,
            x.device,
            x.dtype,
        )

        output = torch.zeros_like(value)

        value_heads = value.view(
            B,
            self.heads,
            self.head_dim,
            H,
            W,
        )

        for point in range(self.points):

            offset = offsets[:, :, point]

            # [B, heads, 2, H, W]
            offset = offset.permute(
                0,
                1,
                3,
                4,
                2,
            )

            offset = offset * self.offset_scale

            # [B, heads, H, W, 2]
            grid = (
                base_grid.unsqueeze(1)
                + offset
            )

            sampled_heads = []

            for head in range(self.heads):

                sampled = F.grid_sample(
                    value_heads[
                        :, head
                    ],
                    grid[
                        :, head
                    ],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )

                sampled_heads.append(
                    sampled
                )

            sampled = torch.stack(
                sampled_heads,
                dim=1,
            )

            attention = weights[
                :, :, point
            ].unsqueeze(2)

            sampled = (
                sampled * attention
            )

            sampled = sampled.reshape(
                B,
                C,
                H,
                W,
            )

            output = output + sampled

        output = self.output(output)

        output = self.norm(
            output + x
        )

        return output