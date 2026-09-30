import numpy as np


def evaluate_search_episode(
    distances,
    confidences,
    success_threshold_mm=3.0,
):
    distances = np.asarray(distances)
    confidences = np.asarray(confidences)

    if len(distances) == 0:
        return {
            "success": False,
            "steps": 0,
            "final_distance_mm": float("inf"),
            "best_distance_mm": float("inf"),
            "max_confidence": 0.0,
        }

    return {
        "success": bool(
            np.min(distances)
            <= success_threshold_mm
        ),
        "steps": int(len(distances)),
        "final_distance_mm": float(
            distances[-1]
        ),
        "best_distance_mm": float(
            np.min(distances)
        ),
        "max_confidence": float(
            np.max(confidences)
        ),
    }


def aggregate_search_metrics(episodes):

    if not episodes:
        return {}

    return {
        "success_rate": float(
            np.mean(
                [e["success"] for e in episodes]
            )
        ),
        "mean_steps": float(
            np.mean(
                [e["steps"] for e in episodes]
            )
        ),
        "mean_final_distance_mm": float(
            np.mean(
                [
                    e["final_distance_mm"]
                    for e in episodes
                ]
            )
        ),
        "mean_best_distance_mm": float(
            np.mean(
                [
                    e["best_distance_mm"]
                    for e in episodes
                ]
            )
        ),
        "mean_max_confidence": float(
            np.mean(
                [
                    e["max_confidence"]
                    for e in episodes
                ]
            )
        ),
    }