import torch
import hydra
from omegaconf import DictConfig

from dataloaders.dataset import create_dataloaders
from models.model_arch import create_model
from models.inference import run_inference
from optimization.optuna_tuner import run_optuna
from models.train import train_model
from models.eval import evaluate_model, save_evaluation_results

from utils.logger import setup_logger
logger = setup_logger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(cfg: DictConfig):    
    logger.info(f"Running training using \033[1;36m Fake News Detection using \"{cfg.model.type}\" Model\033[0m")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    logger.info("Running creating model...")
    model = create_model(cfg)
    
    logger.info("Running training model...")
    train_model(
        cfg,
        model, 
        train_loader, 
        val_loader
    )
    logger.info("Running evaluating model...")
    metrics, predictions = evaluate_model(model, test_loader, loader_name="Test Loader")
    
    logger.info("Running saving results...")
    save_evaluation_results(cfg, model, metrics, predictions)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run_inference(cfg)
    # if cfg.train.mode:
    #     if cfg.train.optimize:
    #         run_optuna(cfg)
    #     else:
    #         run_training(cfg)
    # else:
    #     run_inference(cfg)

if __name__ == "__main__":
    main()
