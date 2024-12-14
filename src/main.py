import torch
from torch.utils.data import DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path

from features.build_features import MultiModalDataset
from models.model_arch import MultiModalClassifier
from models.train import train_model
from models.eval import evaluate_model, save_evaluation_results

from utils.logger import default_logger

default_logger.info("Main script started.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    base_path = Path('/mnt/data')
    train_data = 'train_fake_news.csv'
    val_data = 'val_fake_news.csv'
    test_data = 'test_fake_news.csv'
    
    train_dataset = MultiModalDataset(base_path=base_path, data_name=train_data)
    val_dataset = MultiModalDataset(base_path=base_path, data_name=val_data)
    test_dataset = MultiModalDataset(base_path=base_path, data_name=test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=62, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=62, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=62, shuffle=False)
    
    input_size = 768 + 1000 + 768  # テキスト（768）+ 画像（1000）+ キャプション（768）
    hidden_size = 512
    output_size = 1  # 2値分類
    model = MultiModalClassifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = model.to(device)
    
    

    # トレーニング
    train_model(model, train_loader, val_loader)

    # 評価
    evaluate_model(model, test_loader, model_name)

if __name__ == "__main__":
    main()

