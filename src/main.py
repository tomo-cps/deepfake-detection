import torch
import hydra
from omegaconf import DictConfig

from models.train import run_training
from optimization.optuna_tuner import run_optuna
from models.inference import run_inference
from visualization.visualize import run_visualize
from utils.logger import setup_logger

logger = setup_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.training_config.is_training:
        if cfg.training_config.is_optimizing:
            run_optuna(cfg)
        else:
            run_training(cfg)
    elif cfg.training_config.is_inference:
        run_inference(cfg)
    elif cfg.training_config.is_visualizing:
        run_visualize(cfg)

if __name__ == "__main__":
    main()
