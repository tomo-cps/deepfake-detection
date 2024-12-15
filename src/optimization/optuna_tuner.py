import hydra
from omegaconf import DictConfig
import optuna
import torch
import os
from optimization.objective import objective
import torch
import hydra
from omegaconf import DictConfig

from dataloaders.dataset import create_dataloaders
from models.model_arch import create_model
from models.train import train_model
from models.eval import evaluate_model, save_evaluation_results

from utils.logger import setup_logger
logger = setup_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_optuna(cfg: DictConfig):
    logger.info("Starting\033[1;36m Optuna\033[0m hyperparameter optimization...")
    logger.info(f"Running training using Fake News Detection using \033[1;36m \"{cfg.model.type}\" \033[0mModel")

    checkpoints_dir = f"output/checkpoints/"
    results_dir = f"output/results/"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize", study_name="hyperparameter_tuning")
    study.optimize(lambda trial: objective(cfg, trial), n_trials=cfg.optuna.n_trials)

    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")

    best_trial = study.best_trial
    best_params = best_trial.params
    best_batch_size = best_params['batch_size']
    best_hidden_size = best_params['hidden_size']
    best_dropout_rate = best_params['dropout_rate']
    best_learning_rate = best_params['learning_rate']
    
    # Updae cfg
    cfg.data.batch_size = best_batch_size
    cfg.model.hidden_size = best_hidden_size
    cfg.model.dropout_rate = best_dropout_rate
    cfg.training.learning_rate = best_learning_rate
    cfg.training.num_epochs = cfg.optuna.eval_epochs
    

    logger.info("Retraining with best parameters from Optuna...")
    # 最良パラメータが反映されたcfgを用いて通常トレーニングフローを再実行
    run_training_for_optuna(cfg)

    # Optuna結果保存
    results_df = study.trials_dataframe()
    results_df["best_hidden_size"] = best_hidden_size
    results_df["best_learning_rate"] = best_learning_rate
    results_df["best_batch_size"] = best_batch_size
    results_df.to_csv(f"{results_dir}/optuna_results.csv", index=False)

def run_training_for_optuna(cfg: DictConfig):    
    logger.info(f"Running training model..")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    logger.info("Running creating model...")
    hidden_size = cfg.model.hidden_size
    dropout_rate = cfg.model.dropout_rate
    model = create_model(cfg, hidden_size, dropout_rate)
    
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