import torch
@torch.no_grad()
def extract_vessel_features(
    encoder,
    attention_module,
    images,
    vessel_mask
):
    features = encoder(
        images
    )
    vessel_mask = (
        vessel_mask > 0
    ).float().unsqueeze(1)
    features, attention = (
        attention_module(
            features,
            vessel_mask
        )
    )
    return {
        "features": features,
        "attention": attention,
    }