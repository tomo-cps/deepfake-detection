from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torchvision.models import resnet50
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
class MultiModalDataset(Dataset):
    def __init__(self, base_path: Path, data_name:str)-> None:
        self.base_path = base_path
        self.df = self._load_data(base_path, data_name)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.bert_model.eval()
        
        self.resnet_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        self.resnet_model.eval()
        
        self.transform = get_image_transform()
        
    def _load_data(self, base_path: Path, data_name: str) -> pd.DataFrame:
        df = pd.read_csv(base_path / data_name)
        df = df.head(100)
        return df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        text_inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=128
            ).to(device)
        
        text_embedding = self.bert_model(**text_inputs).last_hidden_state.mean(dim=1)
        
        caption = self.df.iloc[index]['imgcaption']
        caption_inputs = self.tokenizer(
            caption, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=128
            ).to(device)
        
        caption_embedding = self.bert_model(**caption_inputs).last_hidden_state.mean(dim=1)
        
        image_path = self.base_path / 'selected_public_image_set' / f"{self.df.iloc[index]['id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = self.resnet_model(image_tensor).squeeze()

        caption = self.df.iloc[index]['imgcaption']
        if not isinstance(caption, str):
            caption = ""  
        
        label = torch.tensor(self.df.iloc[index]['2_way_label'], dtype=torch.long)

        return {
            'text_embedding': text_embedding.squeeze(),
            'image_embedding': image_embedding,
            'caption_embedding': caption_embedding.squeeze(),
            'label': label
        }
