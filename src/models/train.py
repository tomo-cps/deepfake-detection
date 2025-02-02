from tqdm import tqdm
import torch
import torch.nn as nn
from omegaconf import DictConfig

from dataloaders.dataset import create_dataloaders
from models.model_arch import create_model
from models.eval import evaluate_model, save_evaluation_results

from utils.logger import setup_logger
logger = setup_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(cfg: DictConfig):    
    logger.info(f"Running training using \033[1;36m Fake News Detection using \"{cfg.model.type}\" Model\033[0m")
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

def train_model_for_visualize(cfg: DictConfig, model, train_loader, val_loader):
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

                outputs, attention_weights = model(text_embedding, image_embedding, caption_embedding)
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
