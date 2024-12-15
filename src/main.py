# main.py
import hydra
from omegaconf import DictConfig
import optuna
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from dataloaders.dataset import MultiModalDataset  # 修正：build_features.pyではなくdataset.pyから読み込む
from models.model_arch import MultiModalClassifier
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

def create_dataloaders(cfg: DictConfig, base_path: Path):
    train_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.train_data, cfg=cfg)
    val_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.val_data, cfg=cfg)
    test_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.test_data, cfg=cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_model(cfg: DictConfig):
    input_size = 0
    if cfg.data.text_feature.use:
        input_size += cfg.model.input_size.text
    if cfg.data.image_feature.use:
        input_size += cfg.model.input_size.image
    if cfg.data.caption_feature.use:
        input_size += cfg.model.input_size.caption

    hidden_size = cfg.model.hidden_size
    output_size = cfg.model.output_size

    model = MultiModalClassifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    return model.to(device)

def objective(trial, cfg: DictConfig):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)

    base_path = Path(cfg.base_path)
    train_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.train_data, cfg=cfg)
    val_dataset = MultiModalDataset(base_path=base_path, data_name=cfg.data.val_data, cfg=cfg)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_size = cfg.model.input_size.text + cfg.model.input_size.image + cfg.model.input_size.caption
    output_size = cfg.model.output_size

    model = MultiModalClassifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = model.to(device)

    f1_val = train_model(model, train_loader, val_loader, learning_rate=learning_rate)
    metrics, _ = evaluate_model(model, val_loader)

    # モデルの状態とパラメータを保存
    trial.set_user_attr("model_state_dict", model.state_dict())
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("model_params", {"hidden_size": hidden_size})

    return f1_val

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

    # 最良のトライアルを取得
    best_trial = study.best_trial
    best_model_state_dict = best_trial.user_attrs["model_state_dict"]
    best_metrics = best_trial.user_attrs["metrics"]

    # モデルの再作成：最良トライアルのパラメータでモデルを生成
    best_hidden_size = best_trial.params['hidden_size']
    model = MultiModalClassifier(
        input_size=cfg.model.input_size.text + cfg.model.input_size.image + cfg.model.input_size.caption,
        hidden_size=best_hidden_size,
        output_size=cfg.model.output_size
    )
    model.load_state_dict(best_model_state_dict)

    # 最良モデルの保存
    dummy_predictions = pd.DataFrame({"true_label": [], "prediction": []})
    save_evaluation_results(model, best_metrics, dummy_predictions)

    # Optuna結果保存
    study.trials_dataframe().to_csv(f"{results_dir}/optuna_results.csv", index=False)


def run_training(cfg: DictConfig):
    logger.info("Running training without Optuna...")

    base_path = Path(cfg.base_path)
    train_loader, val_loader, test_loader = create_dataloaders(cfg, base_path)
    model = create_model(cfg)

    train_model(model, train_loader, val_loader)
    metrics, predictions = evaluate_model(model, test_loader)
    save_evaluation_results(model, metrics, predictions)


def save_evaluation_results(model, metrics, predictions):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    checkpoints_dir = f"output/checkpoints/{date_str}"
    results_dir = f"output/results/{date_str}"
    predictions_dir = f"output/results/{date_str}"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    model_name = get_model_name(model)

    model_path = f"{checkpoints_dir}/{model_name}_model_{time_str}.pth"
    results_path = f"{results_dir}/metrics_{model_name}_{time_str}.json"
    predictions_path = f"{predictions_dir}/predictions_{model_name}_{time_str}.csv"

    torch.save(model.state_dict(), model_path)
    print(f"Model state_dict saved to {model_path}")

    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Results saved to {results_path}")
    
    predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

def get_model_name(model) -> str:
    model_name = model.__class__.__name__
    model_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
    return model_name

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.optimize:
        run_optuna(cfg)
    else:
        run_training(cfg)

if __name__ == "__main__":
    main()
