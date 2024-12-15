
import traceback
from tqdm import tqdm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from models.eval import evaluate_model

from utils.logger import setup_logger
logger = setup_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(cfg: DictConfig, model, train_loader, val_loader):
    try:
        num_epochs = cfg.training.num_epochs
        patience = cfg.training.patience
        learning_rate = cfg.training.learning_rate

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        model.to(device)
        model.train()

        best_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                try:
                    text_embedding = batch['text_embedding'].to(device)
                    image_embedding = batch['image_embedding'].to(device)
                    caption_embedding = batch['caption_embedding'].to(device)
                    labels = batch['label'].float().unsqueeze(1).to(device)
                except (TypeError, KeyError):
                    logger.error(f"Batch type: {type(batch)}, Batch content: {batch}")
                    raise

                outputs = model(text_embedding, image_embedding, caption_embedding)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

            # 検証データで評価
            metrics, _ = evaluate_model(model, val_loader, loader_name="Val Loader")

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                patience_counter = 0
                logger.info(f"New best F1 Score: {best_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement in F1 score for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        return best_f1
    except Exception as e:
        logger.exception("Error during training")
        return float('inf')

def train_model_for_optuna(model, train_loader, val_loader, num_epochs, patience, learning_rate):
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        model.to(device)
        model.train()

        best_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                try:
                    text_embedding = batch['text_embedding'].to(device)
                    image_embedding = batch['image_embedding'].to(device)
                    caption_embedding = batch['caption_embedding'].to(device)
                    labels = batch['label'].float().unsqueeze(1).to(device)
                except (TypeError, KeyError):
                    logger.error(f"Batch type: {type(batch)}, Batch content: {batch}")
                    raise

                outputs = model(text_embedding, image_embedding, caption_embedding)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

            metrics, _ = evaluate_model(model, val_loader, loader_name="Val Loader")

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                patience_counter = 0
                logger.info(f"F1 Score: {best_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement in F1 score for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        return best_f1
    except Exception as e:
        logger.exception("Error during training")
        return float('inf')
