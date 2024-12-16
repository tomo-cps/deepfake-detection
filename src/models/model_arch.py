import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiModalClassifier(nn.Module):
    def __init__(self, cfg: DictConfig, hidden_size, dropout_rate):
        super(MultiModalClassifier, self).__init__()
        input_size = cfg.model.input_size.text + cfg.model.input_size.image
        output_size = cfg.model.output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # 2値分類用

    def forward(self, text_embedding, image_embedding, caption_embedding=None):
        # 埋め込みを結合 (caption_embeddingは無視)
        combined = torch.cat((text_embedding, image_embedding), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
class MultiModalClassifierWithCaption(nn.Module):
    def __init__(self, cfg: DictConfig, hidden_size, dropout_rate):
        super(MultiModalClassifierWithCaption, self).__init__()
        input_size = cfg.model.input_size.text + cfg.model.input_size.image + cfg.model.input_size.caption
        output_size = cfg.model.output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # 2値分類用

    def forward(self, text_embedding, image_embedding, caption_embedding):
        # 埋め込みを結合
        combined = torch.cat((text_embedding, image_embedding, caption_embedding), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class MultiModalClassifierWithCaptionUsingAttention(nn.Module):
    def __init__(self, cfg: DictConfig, hidden_size, dropout_rate, num_heads=3):
        super(MultiModalClassifierWithCaptionUsingAttention, self).__init__()
        self.cfg = cfg
        input_size_text = cfg.model.input_size.text
        input_size_image = cfg.model.input_size.image
        output_size = cfg.model.output_size
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_size_text, num_heads=num_heads, batch_first=True)
        
        combined_input_size = input_size_image + input_size_text
        self.fc1 = nn.Linear(combined_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_embedding, image_embedding, caption_embedding):
        text_embedding = text_embedding.unsqueeze(1)  # [Batch, 1, TextFeature]
        caption_embedding = caption_embedding.unsqueeze(1)  # [Batch, 1, CaptionFeature]
        
        # Cross-Attention: Query = caption, Key & Value = text
        attended_text, attention_weights = self.cross_attention(
            query=caption_embedding, key=text_embedding, value=text_embedding
        )
        attended_text = attended_text.squeeze(1)  # Remove sequence dimension after attention
        
        self.attention_weights = attention_weights
        
        combined = torch.cat((attended_text, image_embedding), dim=1)  # [Batch, CombinedFeature]
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        if self.cfg.training_config.is_visualizing:
            return x, attention_weights
        
        return x
    
    def visualize_attention(self, text_tokens, caption_tokens):
        """
        注意重みを可視化するメソッド
        - text_tokens: テキストトークンのリスト
        - caption_tokens: キャプショントークンのリスト
        """
        if self.attention_weights is None:
            raise ValueError("Attention weights are not available. Forward pass must be run first.")
        
        # 注意重みを取り出す（最初のバッチ）
        attention = self.attention_weights[0].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=text_tokens, yticklabels=caption_tokens, cmap="viridis", annot=True)
        plt.xlabel("Text Tokens")
        plt.ylabel("Caption Tokens")
        plt.title("Attention Weights")
        plt.show()

def create_model(cfg: DictConfig, hidden_size, dropout_rate):
    model_type = cfg.model.type
    if model_type == "multi_modal":
        model = MultiModalClassifier(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
    elif model_type == "multi_modal_with_caption":
        model = MultiModalClassifierWithCaption(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate)
    elif model_type == "multi_modal_with_caption_using_cross_attention":
        num_heads = cfg.model.get("num_heads", 3)
        model = MultiModalClassifierWithCaptionUsingAttention(cfg, hidden_size=hidden_size, dropout_rate=dropout_rate, num_heads=num_heads)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model.to(device)
