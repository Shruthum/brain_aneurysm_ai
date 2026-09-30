import torch
import numpy as np

from .metrics import (
    binary_metrics,
    localization_error_mm,
    anatomy_accuracy,
    top_k_accuracy,
)


class AneurysmEvaluator:

    def __init__(
        self,
        detector,
        device="cuda",
    ):
        self.detector = detector
        self.device = torch.device(device)

    @torch.no_grad()
    def evaluate_detector(
        self,
        dataloader,
    ):
        self.detector.eval()

        object_targets = []
        object_scores = []

        center_errors = []

        anatomy_predictions = []
        anatomy_targets = []

        for batch in dataloader:

            images = batch["image"].to(
                self.device
            )

            predictions = self.detector(
                images
            )

            object_score = torch.sigmoid(
                predictions["objectness"]
            ).flatten()

            object_targets.extend(
                batch["objectness"]
                .cpu()
                .numpy()
                .tolist()
            )

            object_scores.extend(
                object_score
                .cpu()
                .numpy()
                .tolist()
            )

            positive = (
                batch["objectness"] > 0.5
            )

            if positive.any():

                pred_center = (
                    predictions["center"]
                    [positive]
                    .cpu()
                    .numpy()
                )

                true_center = (
                    batch["center"]
                    [positive]
                    .cpu()
                    .numpy()
                )

                errors = np.linalg.norm(
                    pred_center - true_center,
                    axis=1,
                )

                center_errors.extend(
                    errors.tolist()
                )

                pred_anatomy = torch.argmax(
                    predictions["anatomy"]
                    [positive],
                    dim=1,
                )

                true_anatomy = (
                    batch["anatomy"]
                    [positive]
                )

                anatomy_predictions.extend(
                    pred_anatomy
                    .cpu()
                    .numpy()
                    .tolist()
                )

                anatomy_targets.extend(
                    true_anatomy
                    .cpu()
                    .numpy()
                    .tolist()
                )

        results = binary_metrics(
            object_targets,
            object_scores,
        )

        results["mean_center_error"] = (
            float(np.mean(center_errors))
            if center_errors
            else 0.0
        )

        if anatomy_targets:

            results["anatomy_accuracy"] = (
                anatomy_accuracy(
                    anatomy_predictions,
                    anatomy_targets,
                )
            )

        return results