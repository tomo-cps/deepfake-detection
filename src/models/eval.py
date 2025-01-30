import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from tabulate import tabulate

from utils.logger import setup_logger
logger = setup_logger(__name__)

console = Console()

def evaluate_model(model, test_loader, loader_name="Data Loader"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            try:
                text_embedding = batch['text_embedding'].to(device)
                image_embedding = batch['image_embedding'].to(device)
                caption_embedding = batch['caption_embedding'].to(device)
                labels = batch['label'].to(device)

                outputs = model(text_embedding, image_embedding, caption_embedding)
                if isinstance(outputs, tuple):  # Handle tuple output
                    outputs, attention_weights = outputs
                preds = (outputs > 0.5).float().cpu().numpy().astype(int).flatten()

                true_labels.extend(labels.cpu().numpy().flatten())
                predictions.extend(preds)
            except Exception as e:
                logger.exception("Error during batch processing in evaluation.")
                raise

    # sklearn to calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    _display_metrics(metrics, table_name=f"Metrics Table for {loader_name}")

    results = pd.DataFrame({
        "true_label": true_labels,
        "prediction": [int(pred) for pred in predictions]  # Fixed
    })

    logger.info(f"Evaluation completed for {loader_name}. Metrics: {metrics}")
    return metrics, results

def _display_metrics(metrics, table_name="Metrics Table"):
    headers = ["Metric", "Value"]
    table = [(k.capitalize(), f"{v:.4f}") for k, v in metrics.items()]
    logger.info(f"{table_name}: {metrics}")
    print("-" * len(table_name))
    print(f"{table_name}\n" + "-" * len(table_name))
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

def evaluate_model_for_visualize(model, test_loader, loader_name="Test Loader"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    attention_weights_list = []  # To store attention weights for visualization

    with torch.no_grad():
        for batch in tqdm(test_loader):
            try:
                text_embedding = batch['text_embedding'].to(device)
                image_embedding = batch['image_embedding'].to(device)
                caption_embedding = batch['caption_embedding'].to(device)
                labels = batch['label'].to(device)

                outputs, attention_weights = model(text_embedding, image_embedding, caption_embedding)
                preds = (outputs > 0.5).float().cpu().numpy().astype(int).flatten()

                true_labels.extend(labels.cpu().numpy().flatten())
                predictions.extend(preds)
                attention_weights_list.append(attention_weights.cpu().numpy())  # Store attention weights
            except Exception as e:
                logger.exception("Error during batch processing in evaluation.")
                raise

    # sklearn to calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    _display_metrics(metrics, table_name=f"Metrics Table for {loader_name}")

    results = pd.DataFrame({
        "true_label": true_labels,
        "prediction": [int(pred) for pred in predictions]
    })

    logger.info(f"Evaluation completed for {loader_name}. Metrics: {metrics}")

    # Visualize attention weights
    _visualize_attention(attention_weights_list)

    return metrics, results

def _visualize_attention(attention_weights_list):
    """
    Visualize attention weights for a single batch
    """
    if not attention_weights_list:
        logger.warning("No attention weights found for visualization.")
        return

    # Example: Visualize the first batch of attention weights
    attention_weights = attention_weights_list[0][0]  # First example in first batch

    # Create dummy tokens for visualization
    text_tokens = [f"T{i}" for i in range(attention_weights.shape[1])]  # Example text tokens
    caption_tokens = [f"C{i}" for i in range(attention_weights.shape[0])]  # Example caption tokens

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=text_tokens, yticklabels=caption_tokens, cmap="viridis", annot=True)
    plt.xlabel("Text Tokens")
    plt.ylabel("Caption Tokens")
    plt.title("Attention Weights")
    plt.savefig('output.png')
    plt.show()

def save_evaluation_results(cfg: DictConfig, model, metrics, predictions):
    checkpoints_dir = f"output/checkpoints"
    results_dir = f"output/results"
    predictions_dir = f"output/results"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    model_name = cfg.model.type

    model_path = f"{checkpoints_dir}/{model_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    
    now = datetime.now()
    metrics_with_info = {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "optimize": cfg.training_config.is_optimizing,
        "method": model_name,
        **metrics
    }

    results_path = f"{results_dir}/metrics_{model_name}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics_with_info, f, indent=4)

    predictions_path = f"{predictions_dir}/predictions_{model_name}.csv"
    predictions.to_csv(predictions_path, index=False)

    logger.info(f"Model saved at {model_path}")
    logger.info(f"Metrics saved at {results_path}")
    logger.info(f"Predictions saved at {predictions_path}")

    _display_save_messages(model_path, results_path, predictions_path)

def _display_save_messages(model_path, results_path, predictions_path):
    content = (
        f"[bold green]Model state_dict saved to:[/bold green] {model_path}\n"
        f"[bold cyan]Results saved to:[/bold cyan] {results_path}\n"
        f"[bold magenta]Predictions saved to:[/bold magenta] {predictions_path}"
    )
    logger.info("Displaying save information panel.")
    console.print(Panel.fit(content, title="Save Information", border_style="white"))
