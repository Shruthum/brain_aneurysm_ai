import torch
import torch.nn.functional as F


def multiclass_dice_score(
    logits,
    target,
    num_classes=14,
    include_background=False
):
    prediction = torch.argmax(
        logits,
        dim=1
    )

    scores = []

    start = (
        0
        if include_background
        else 1
    )

    for cls in range(
        start,
        num_classes
    ):

        pred = (
            prediction == cls
        ).float()

        true = (
            target == cls
        ).float()

        intersection = (
            pred * true
        ).sum()

        denominator = (
            pred.sum()
            + true.sum()
        )

        if denominator == 0:
            continue

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        scores.append(dice)

    if not scores:
        return logits.new_tensor(1.0)

    return torch.stack(
        scores
    ).mean()


def multiclass_dice_loss(
    logits,
    target,
    num_classes=14
):
    probability = F.softmax(
        logits,
        dim=1
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=num_classes
    )

    target_one_hot = (
        target_one_hot
        .permute(0, 3, 1, 2)
        .float()
    )

    scores = []

    for cls in range(
        1,
        num_classes
    ):

        pred = probability[:, cls]

        true = target_one_hot[:, cls]

        intersection = (
            pred * true
        ).sum(dim=(1, 2))

        denominator = (
            pred.sum(dim=(1, 2))
            + true.sum(dim=(1, 2))
        )

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        scores.append(
            dice.mean()
        )

    return 1.0 - torch.stack(
        scores
    ).mean()


def vessel_loss(
    logits,
    target,
    dice_weight=1.0
):
    ce = F.cross_entropy(
        logits,
        target
    )

    dice = multiclass_dice_loss(
        logits,
        target
    )

    total = ce + dice_weight * dice

    return {
        "total": total,
        "ce": ce,
        "dice_loss": dice
    }