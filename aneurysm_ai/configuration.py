from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Configuration:
    data_root: Path
    @property
    def train_csv(self):
        return self.data_root/"train.csv"
    @property
    def localizer_csv(self):
        return self.data_root/"train_localizers.csv"
    @property
    def series_dir(self):
        return self.data_root/"series"
    @property
    def segmentation_dir(self):
        return self.data_root/"segmentations"
    image_size: int = 224
    roi_coarse = (32,128,128)
    roi_medium = (24,96,96)
    roi_fine = (16,64,64)
    swin_model = "microsoft/swin-twin-patch4-window7-224"
    num_actions = 9
    max_steps = 64
    move_step = 8
    batch_size = 2
    num_workers = 2
    encoder_lr: 1e-5
    decoder_lr: 1e-4
    weight_decay = 1e-4
    epochs = 10
    num_anatomy_classes = 13
    num_segmentation_classes = 14
    rl_state_dim: int = 5
    rl_action_dim: int = 9
    rl_max_steps: int = 64
    rl_move_step: float = 0.08
    rl_scale_step: float = 0.15
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_clip_eps: float = 0.2
    rl_lr: float = 3e-4
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @classmethod
    def from_dataset(cls,data_root):
        return cls(data_root = Path(data_root))