import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm 
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from tabulate import tabulate

console = Console()

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            text_embedding = batch['text_embedding']
            image_embedding = batch['image_embedding']
            caption_embedding = batch['caption_embedding']
            labels = batch['label'].cpu().numpy()

            outputs = model(text_embedding, image_embedding, caption_embedding)
            preds = (outputs.cpu().numpy() > 0.5).astype(int)

            predictions.extend(preds)
            true_labels.extend(labels)

    # sklearnを使って各評価指標を計算
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
    
    _display_metrics(metrics)
    
    results = pd.DataFrame({
        "true_label": true_labels,
        "prediction": [int(pred[0]) for pred in predictions]  # フラット化
    })

    return metrics, results

def _display_metrics(metrics):
    headers = ["Metric", "Value"]
    table = [(k.capitalize(), f"{v:.4f}") for k, v in metrics.items()]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    
def save_evaluation_results(cfg: DictConfig, model, metrics, predictions):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    checkpoints_dir = f"output/checkpoints/{date_str}"
    results_dir = f"output/results/{date_str}"
    predictions_dir = f"output/results/{date_str}"

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    model_name = cfg.model.type

    model_path = f"{checkpoints_dir}/{model_name}_model_{time_str}.pth"
    torch.save(model.state_dict(), model_path)
    
    metrics_with_info = {
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),  # Add time information
        "method": model_name,                      # Add model name
        **metrics                                  # Include existing metrics
    }
    
    results_path = f"{results_dir}/metrics_{model_name}_{time_str}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics_with_info, f, indent=4)
        
    predictions_path = f"{predictions_dir}/predictions_{model_name}_{time_str}.csv"
    predictions.to_csv(predictions_path, index=False)
    
    _display_save_messages(model_path, results_path, predictions_path)

def _display_save_messages(model_path, results_path, predictions_path):
    content = (
        f"[bold green]Model state_dict saved to:[/bold green] {model_path}\n"
        f"[bold cyan]Results saved to:[/bold cyan] {results_path}\n"
        f"[bold magenta]Predictions saved to:[/bold magenta] {predictions_path}"
    )
    console.print(Panel.fit(content, title="Save Information", border_style="white"))