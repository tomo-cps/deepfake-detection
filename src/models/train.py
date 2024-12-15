import traceback
from tqdm import tqdm 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from models.eval import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, num_epochs=1, patience=3, learning_rate=1e-5):
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        model.to(device)
        model.train()

        best_f1 = 0.0  # F1スコアが最も良い場合のモデルを記録
        patience_counter = 0  # 改善が見られないエポック数を記録
        
        for epoch in range(num_epochs):
            model.train()  # トレーニングモードに切り替え
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # データ形式を確認しながら取得
                try:
                    text_embedding = batch['text_embedding'].to(device)
                    image_embedding = batch['image_embedding'].to(device)
                    caption_embedding = batch['caption_embedding'].to(device)
                    labels = batch['label'].float().unsqueeze(1).to(device)
                except (TypeError, KeyError):
                    print(f"Batch type: {type(batch)}, Batch content: {batch}")
                    raise

                # 順伝播
                outputs = model(text_embedding, image_embedding, caption_embedding)
                loss = criterion(outputs, labels)

                # 逆伝播と最適化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')

            # 検証データで評価
            metrics, _ = evaluate_model(model, val_loader)

            # F1スコアの改善をチェック
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                patience_counter = 0  # 改善があった場合、カウンタをリセット
                print(f"New best F1 Score: {best_f1:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in F1 score for {patience_counter} epoch(s)")

            # 早期終了のチェック
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        return best_f1  # 最良の損失を返す
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        return float('inf')
