import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50

class MultiModalClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiModalClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # 2値分類用

    def forward(self, text_embedding, image_embedding, caption_embedding):
        # 埋め込みを結合
        combined = torch.cat((text_embedding, image_embedding, caption_embedding), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
