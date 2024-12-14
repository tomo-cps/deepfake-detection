import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.eval import evaluate_model_for_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, num_epochs=10, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCELoss()
    model.train()
    best_f1 = 0.0  # F1スコアが最も良い場合のモデルを記録
    patience_counter = 0  # 改善が見られないエポック数を記録
    
    for epoch in range(num_epochs):
        model.train()  # トレーニングモードに切り替え
        total_loss = 0
        for batch in train_loader:
            text_embedding = batch['text_embedding']
            image_embedding = batch['image_embedding']
            caption_embedding = batch['caption_embedding']
            labels = batch['label'].float().unsqueeze(1).to(device)  # ラベルをfloat型に変換

            # 順伝播
            outputs = model(text_embedding, image_embedding, caption_embedding)
            loss = criterion(outputs, labels)

            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')

        # テストデータで評価
        accuracy, precision, recall, f1 = evaluate_model_for_eval(model, val_loader)

        # F1スコアの改善をチェック
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0  # 改善があった場合、カウンタをリセット
            print(f"New best F1 Score: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in F1 score for {patience_counter} epoch(s)")

        # 早期終了のチェック
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
