import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from dataloaders.dataset import create_dataloaders
from models.train import train_model_for_visualize
from models.eval import evaluate_model_for_visualize, save_evaluation_results
from models.model_arch import create_model
from utils.logger import setup_logger

logger = setup_logger(__name__)

def run_visualize(cfg: DictConfig):
    logger.info(f"THIS IS THE\033[1;36m VISUALIZATION\033[0m MODEL")
    logger.info(f"Running training using \033[1;36m Fake News Detection using \"{cfg.model.type}\" Model\033[0m")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    logger.info("Running creating model...")
    hidden_size = cfg.model.hidden_size
    dropout_rate = cfg.model.dropout_rate
    
    model = create_model(cfg, hidden_size, dropout_rate)
    
    logger.info("Running training model...")
    train_model_for_visualize(
        cfg,
        model, 
        train_loader, 
        val_loader
    )
    logger.info("Running evaluating model...")
    metrics, predictions = evaluate_model_for_visualize(model, test_loader, loader_name="Test Loader")
    
    logger.info("Running saving results...")
    save_evaluation_results(cfg, model, metrics, predictions)