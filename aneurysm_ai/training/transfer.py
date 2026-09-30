import torch


def freeze_encoder(model):
    model.freeze_encoder()


def unfreeze_last_stage(model):
    model.unfreeze_last_stage()


def unfreeze_encoder(model):
    model.unfreeze_encoder()


def parameter_groups(
    model,
    encoder_lr=1e-5,
    new_layer_lr=1e-4,
    weight_decay=1e-4
):

    encoder_parameters = []
    new_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith("encoder."):

            encoder_parameters.append(
                parameter
            )

        else:

            new_parameters.append(
                parameter
            )

    groups = []

    if new_parameters:

        groups.append(
            {
                "params": new_parameters,
                "lr": new_layer_lr,
                "weight_decay": weight_decay,
            }
        )

    if encoder_parameters:

        groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            }
        )

    return groups


def create_optimizer(
    model,
    encoder_lr=1e-5,
    new_layer_lr=1e-4,
    weight_decay=1e-4
):

    groups = parameter_groups(
        model,
        encoder_lr=encoder_lr,
        new_layer_lr=new_layer_lr,
        weight_decay=weight_decay
    )

    return torch.optim.AdamW(
        groups
    )