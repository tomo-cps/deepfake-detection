import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re

def evaluate_model_for_eval(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
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

    # 結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

def evaluate_model(model, test_loader, model_name):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
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

    # 結果を表示
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    save_evaluation_results(
        model=model,
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_labels=true_labels,
        predictions=predictions,
    )

    return accuracy, precision, recall, f1


def save_evaluation_results(model, model_name, accuracy, precision, recall, f1, true_labels, predictions):
    """
    モデル、評価結果、予測結果を指定のディレクトリ構造に保存する関数

    Parameters:
        model: PyTorchモデル
        model_name: str, モデルの名前 (例: "MultiModalClassifier")
        accuracy: float, 正確性 (Accuracy)
        precision: float, 適合率 (Precision)
        recall: float, 再現率 (Recall)
        f1: float, F1スコア
        true_labels: list, 真のラベル
        predictions: list, モデルの予測
    """
    # 現在の日時を取得
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")  # "YYYY-MM-DD"
    time_str = now.strftime("%H%M%S")  # "HHMMSS"

    # ディレクトリ構造を定義
    checkpoints_dir = f"output/checkpoints/{date_str}"
    results_dir = f"output/results/{date_str}"
    predictions_dir = f"output/results/{date_str}"

    # 必要なディレクトリを作成（存在しない場合）
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    model_name = get_model_name(model, model_name)

    # ファイルパスを定義
    # モデル名をファイルに含める
    model_path = f"{checkpoints_dir}/{model_name}_{time_str}.pth"
    results_path = f"{results_dir}/metrics_{model_name}_{time_str}.json"
    predictions_path = f"{predictions_dir}/predictions_{model_name}_{time_str}.csv"

    # モデルの状態を保存
    torch.save(model.state_dict(), model_path)
    print(f"Model state_dict saved to {model_path}")

    # 結果を保存
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    # リスト内の単一要素を抽出して保存
    flat_true_labels = [int(label) for label in true_labels]
    flat_predictions = [int(pred[0]) if isinstance(pred, (list, np.ndarray)) else int(pred) 
                        for pred in predictions]

    # データフレームを作成して保存
    predictions_df = pd.DataFrame({
        'true_labels': flat_true_labels,
        'predictions': flat_predictions
    })
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

def get_model_name(model, model_name: str) -> str:
    # "MultiModalClassifier" → "multi_modal_classifier"
    model_name = model.__class__.__name__
    model_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
    return model_name
