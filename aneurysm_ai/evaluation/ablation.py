from dataclasses import dataclass


@dataclass
class AblationConfig:

    name: str

    vessel_pretraining: bool = False
    vessel_attention: bool = False
    multi_scale: bool = False
    bbga: bool = False
    deformable: bool = False
    ppo: bool = False


ABLATIONS = [

    AblationConfig(
        name="baseline_swin",
    ),

    AblationConfig(
        name="vessel_pretrained",
        vessel_pretraining=True,
    ),

    AblationConfig(
        name="vessel_attention",
        vessel_pretraining=True,
        vessel_attention=True,
    ),

    AblationConfig(
        name="multi_scale",
        vessel_pretraining=True,
        vessel_attention=True,
        multi_scale=True,
    ),

    AblationConfig(
        name="bbga",
        vessel_pretraining=True,
        vessel_attention=True,
        multi_scale=True,
        bbga=True,
    ),

    AblationConfig(
        name="deformable",
        vessel_pretraining=True,
        vessel_attention=True,
        multi_scale=True,
        bbga=True,
        deformable=True,
    ),

    AblationConfig(
        name="full_model",
        vessel_pretraining=True,
        vessel_attention=True,
        multi_scale=True,
        bbga=True,
        deformable=True,
        ppo=True,
    ),
]