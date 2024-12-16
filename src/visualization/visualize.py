import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from models.eval import evaluate_model, save_evaluation_results
from dataloaders.dataset import MultiModalDataset
from models.model_arch import MultiModalClassifier, MultiModalClassifierWithCaption, MultiModalClassifierWithCaptionUsingAttention
from utils.logger import setup_logger

logger = setup_logger(__name__)

@hydra.main(config_path="../configs", config_name="config.yaml", version_base="1.1")
def run_inference(cfg: DictConfig):
    run_dir = Path(cfg.inference.run_output_dir)
    prev_config_path = run_dir / ".hydra" / "config.yaml"
    prev_cfg = OmegaConf.load(str(prev_config_path))

    checkpoint_files = list((run_dir / "output" / "checkpoints").glob("**/*.pth"))
    checkpoint_path = checkpoint_files[0]
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    prev_cfg = OmegaConf.load(str(prev_config_path))

    cfg = OmegaConf.merge(cfg, prev_cfg)
    
    test_dataset = MultiModalDataset(cfg=cfg, data_name=cfg.data.test_data)
    test_loader = DataLoader(test_dataset, batch_size=cfg.inference.batch_size, shuffle=False)
    
    model_type = cfg.model.type
    hidden_size = cfg.model.hidden_size
    dropout_rate = cfg.model.dropout_rate
    num_heads = cfg.model.num_heads
    
    if model_type == "multi_modal":
        model = MultiModalClassifier(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
        logger.info("Using MultiModalClassifier model")
    elif model_type == "multi_modal_with_caption":
        model = MultiModalClassifierWithCaption(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
        logger.info("Using MultiModalClassifierWithCaption model")
    elif model_type == "multi_modal_with_caption_using_cross_attention":
        model = MultiModalClassifierWithCaptionUsingAttention(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate, num_heads=num_heads)
        logger.info(f"Using MultiModalClassifierWithCaptionUsingAttention model with head size {num_heads}")
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info("Running loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    logger.info("Running evaluating model...")
    metrics, predictions = evaluate_model(model, test_loader, loader_name="Test Loader")
    
    logger.info("Running saving results...")
    save_evaluation_results(cfg, model, metrics, predictions)