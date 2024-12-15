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
from optimization import objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

def run_optuna(cfg: DictConfig):
    logger.info("Starting Optuna hyperparameter optimization...")

    checkpoints_dir = f"output/checkpoints/"
    results_dir = f"output/results/"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="hyperparameter_tuning")
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.optuna.n_trials)

    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")

    best_trial = study.best_trial
    best_params = best_trial.params
    best_hidden_size = best_params['hidden_size']
    best_learning_rate = best_params['learning_rate']
    best_batch_size = best_params['batch_size']

    # 最良パラメータをcfgに反映
    cfg.model.hidden_size = best_hidden_size
    cfg.data.batch_size = best_batch_size
    cfg.training.learning_rate = best_learning_rate
    cfg.training.num_epochs = cfg.optuna.eval_epochs

    logger.info("Retraining with best parameters from Optuna...")
    # 最良パラメータが反映されたcfgを用いて通常トレーニングフローを再実行
    run_training(cfg)

    # Optuna結果保存
    study.trials_dataframe().to_csv(f"{results_dir}/optuna_results.csv", index=False)