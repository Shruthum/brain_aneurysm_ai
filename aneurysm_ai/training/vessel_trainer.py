from pathlib import Path

import torch
from torch.amp import autocast, GradScaler

from .losses import (
    vessel_loss,
    multiclass_dice_score
)


class VesselTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs,
        checkpoint_dir,
        scheduler=None
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=device.type == "cuda"
        )

        self.best_dice = -1.0

    def _run_epoch(
        self,
        loader,
        training=True
    ):

        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_dice_loss = 0.0
        total_dice = 0.0

        count = 0

        for batch in loader:

            image = batch["image"].to(
                self.device,
                non_blocking=True
            )

            target = batch["mask"].to(
                self.device,
                non_blocking=True
            )

            if training:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

            with autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda"
            ):

                logits = self.model(
                    image
                )

                losses = vessel_loss(
                    logits,
                    target
                )

            if training:

                self.scaler.scale(
                    losses["total"]
                ).backward()

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            batch_size = image.size(0)

            total_loss += (
                losses["total"].item()
                * batch_size
            )

            total_ce += (
                losses["ce"].item()
                * batch_size
            )

            total_dice_loss += (
                losses["dice_loss"].item()
                * batch_size
            )

            with torch.no_grad():

                dice = multiclass_dice_score(
                    logits,
                    target
                )

            total_dice += (
                dice.item()
                * batch_size
            )

            count += batch_size

        return {
            "loss": total_loss / count,
            "ce": total_ce / count,
            "dice_loss": total_dice_loss / count,
            "dice": total_dice / count,
        }

    def fit(self):

        for epoch in range(
            1,
            self.epochs + 1
        ):

            train_metrics = self._run_epoch(
                self.train_loader,
                training=True
            )

            with torch.no_grad():

                val_metrics = self._run_epoch(
                    self.val_loader,
                    training=False
                )

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch}/{self.epochs}"
            )

            print(
                f"Train "
                f"loss={train_metrics['loss']:.4f} "
                f"CE={train_metrics['ce']:.4f} "
                f"DiceLoss={train_metrics['dice_loss']:.4f} "
                f"Dice={train_metrics['dice']:.4f}"
            )

            print(
                f"Val   "
                f"loss={val_metrics['loss']:.4f} "
                f"CE={val_metrics['ce']:.4f} "
                f"DiceLoss={val_metrics['dice_loss']:.4f} "
                f"Dice={val_metrics['dice']:.4f}"
            )

            if val_metrics["dice"] > self.best_dice:

                self.best_dice = (
                    val_metrics["dice"]
                )

                self.save(
                    epoch,
                    val_metrics
                )

                print(
                    "Saved best vessel model."
                )

    def save(
        self,
        epoch,
        metrics
    ):

        path = (
            self.checkpoint_dir /
            "best_vessel.pt"
        )

        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": (
                    self.optimizer.state_dict()
                ),
                "metrics": metrics,
            },
            path
        )