import hydra
from omegaconf import DictConfig
import optuna
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataloaders.dataset import MultiModalDataset, create_dataloaders  # 修正：build_features.pyではなくdataset.pyから読み込む
from models.model_arch import MultiModalClassifier, MultiModalClassifierWithCaption, MultiModalClassifierWithCaptionUsingAttention
from models.train import train_model
from models.eval import evaluate_model
import logging
import os
import re
import json
import pandas as pd
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def objective(trial, cfg: DictConfig):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)

    base_path = Path(cfg.base_path)
    train_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.train_data, cfg=cfg)
    val_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.val_data, cfg=cfg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model_type = cfg.model.type
    if model_type == "multi_modal":
        model = MultiModalClassifier(cfg, hidden_size=hidden_size)
    elif model_type == "multi_modal_with_caption":
        model = MultiModalClassifierWithCaption(cfg, hidden_size=hidden_size)
    elif model_type == "multi_modal_with_caption_using_cross_attention":
        num_heads = cfg.model.get("num_heads", 3)
        model = MultiModalClassifierWithCaptionUsingAttention(cfg, hidden_size=hidden_size, num_heads=num_heads)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model = model.to(device)

    f1_val = train_model(model, train_loader, val_loader, num_epochs=cfg.optuna.trial_epochs, learning_rate=learning_rate)
    metrics, _ = evaluate_model(model, val_loader)

    trial.set_user_attr("model_state_dict", model.state_dict())
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("model_params", {"hidden_size": hidden_size})

    return f1_val