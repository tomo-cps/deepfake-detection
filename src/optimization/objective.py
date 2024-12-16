import hydra
from omegaconf import DictConfig
import optuna
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataloaders.dataset import MultiModalDataset, create_dataloaders  # 修正：build_features.pyではなくdataset.pyから読み込む
from models.model_arch import MultiModalClassifier, MultiModalClassifierWithCaption, MultiModalClassifierWithCaptionUsingAttention
from models.train import train_model_for_optuna
from models.eval import evaluate_model
import os
import re
import json
import pandas as pd
from datetime import datetime

from utils.logger import setup_logger
logger = setup_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(cfg: DictConfig, trial):
    model_type = cfg.model.type
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

    logger.info(f"Starting trial {trial.number}:")
    logger.info(f"Proposed hyperparameters - Learning Rate: {learning_rate}, Batch Size: {batch_size}, Hidden Size: {hidden_size}")

    train_dataset = MultiModalDataset(cfg=cfg, data_name=cfg.data.train_data)
    val_dataset = MultiModalDataset(cfg=cfg, data_name=cfg.data.val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "multi_modal":
        model = MultiModalClassifier(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
        logger.info("Using MultiModalClassifier model")
    elif model_type == "multi_modal_with_caption":
        model = MultiModalClassifierWithCaption(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
        logger.info("Using MultiModalClassifierWithCaption model")
    elif model_type == "multi_modal_with_caption_using_cross_attention":
        num_heads = trial.suggest_int('num_heads', 1, 4)
        model = MultiModalClassifierWithCaptionUsingAttention(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate, num_heads=num_heads)
        logger.info(f"Using MultiModalClassifierWithCaptionUsingAttention model with head size {num_heads}")
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)
    logger.info(f"Model loaded to {device}")

    patience = 3
    logger.info(f"Training started with patience={patience}, num_epochs={cfg.optuna.trial_epochs}, learning_rate={learning_rate}")
    f1_val = train_model_for_optuna(model, train_loader, val_loader, num_epochs=cfg.optuna.trial_epochs, patience=patience, learning_rate=learning_rate)
    logger.info(f"Training finished with validation F1 score: {f1_val:.4f}")

    metrics, _ = evaluate_model(model, val_loader, loader_name="Val Loader")
    logger.info(f"Evaluation metrics: {metrics}")

    trial.set_user_attr("model_state_dict", model.state_dict())
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("model_params", {"hidden_size": hidden_size})

    return f1_val
