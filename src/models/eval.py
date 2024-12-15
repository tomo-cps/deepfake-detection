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
from tqdm import tqdm 

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
    
    results = pd.DataFrame({
        "true_label": true_labels,
        "prediction": [int(pred[0]) for pred in predictions]  # フラット化
    })

    return metrics, results