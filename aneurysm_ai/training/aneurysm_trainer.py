import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probabilities = torch.sigmoid(logits)

        pt = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce

        return loss.mean()


class AneurysmDetectionLoss(nn.Module):

    def __init__(
        self,
        center_weight=2.0,
        anatomy_weight=1.0,
    ):
        super().__init__()

        self.objectness_loss = FocalLoss(
            alpha=0.25,
            gamma=2.0,
        )

        self.center_weight = center_weight
        self.anatomy_weight = anatomy_weight

        self.center_loss = nn.SmoothL1Loss(
            reduction="none"
        )

        self.anatomy_loss = nn.CrossEntropyLoss(
            reduction="none"
        )

    def forward(
        self,
        predictions,
        objectness,
        center,
        anatomy,
    ):
        pred_objectness = predictions["objectness"]

        objectness = objectness.float().view_as(
            pred_objectness
        )

        loss_object = self.objectness_loss(
            pred_objectness,
            objectness,
        )

        # --------------------------------------------------
        # Center loss
        # Only positive samples should contribute.
        # --------------------------------------------------

        pred_center = predictions["center"]

        center = center.float()

        center_loss = self.center_loss(
            pred_center,
            center,
        ).mean(dim=-1)

        positive = objectness.squeeze(-1) > 0.5

        if positive.any():
            loss_center = center_loss[positive].mean()
        else:
            loss_center = pred_center.sum() * 0.0

        # --------------------------------------------------
        # Anatomy loss
        # Only positive samples should contribute.
        # --------------------------------------------------

        pred_anatomy = predictions["anatomy"]

        anatomy = anatomy.long().view(-1)

        anatomy_loss = self.anatomy_loss(
            pred_anatomy,
            anatomy,
        )

        if positive.any():
            loss_anatomy = anatomy_loss[positive].mean()
        else:
            loss_anatomy = pred_anatomy.sum() * 0.0

        total = (
            loss_object
            + self.center_weight * loss_center
            + self.anatomy_weight * loss_anatomy
        )

        return {
            "total": total,
            "objectness": loss_object,
            "center": loss_center,
            "anatomy": loss_anatomy,
        }