from .configuration import Configuration
from .seed import set_seed
import pandas as pd

def create_config(data_root,seed = 444):
    cfg = Configuration.from_dataset(data_root)
    set_seed(seed)
    return cfg

def check_dataset(cfg: Configuration):
    required = [cfg.train_csv,cfg.localizer_csv,cfg.series_dir,cfg.segmentation_dir]
    status = {}
    for path in required:
        status[str(path)] = path.exists()
    return status

def load_train_dataframe(cfg: Configuration):
    return pd.read_csv(cfg.train_csv)