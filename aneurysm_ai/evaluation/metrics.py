import math
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(
    targets,
    scores,
    threshold=0.5,
):
    targets = np.asarray(targets).astype(np.int64)
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype(np.int64)

    tp = np.sum((predictions == 1) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-8)
    )

    accuracy = (
        (tp + tn)
        / max(tp + tn + fp + fn, 1)
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def localization_error_voxel(
    predicted,
    target,
):
    predicted = np.asarray(predicted)
    target = np.asarray(target)

    return float(
        np.linalg.norm(predicted - target)
    )


def localization_error_mm(
    predicted_voxel,
    target_voxel,
    geometry,
):
    predicted_world = geometry.voxel_to_world(
        predicted_voxel
    )

    target_world = geometry.voxel_to_world(
        target_voxel
    )

    return float(
        np.linalg.norm(
            predicted_world - target_world
        )
    )


def anatomy_accuracy(
    predictions,
    targets,
):
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    return float(
        np.mean(predictions == targets)
    )


def top_k_accuracy(
    logits,
    targets,
    k=3,
):
    logits = np.asarray(logits)
    targets = np.asarray(targets)

    top_k = np.argsort(
        logits,
        axis=1,
    )[:, -k:]

    return float(
        np.mean(
            [
                target in row
                for target, row in zip(
                    targets,
                    top_k,
                )
            ]
        )
    )


def search_efficiency(
    successful,
    evaluated_rois,
):
    return float(
        successful
        / max(evaluated_rois, 1)
    )


def mean_search_distance(distances):
    distances = np.asarray(distances)

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))