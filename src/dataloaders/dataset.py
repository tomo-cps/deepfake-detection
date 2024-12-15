# dataloaders/dataset.py
from pathlib import Path
from torch.utils.data import Dataset
import torch
import pandas as pd
from features.build_features import TextFeatureExtractor, ImageFeatureExtractor
import logging
class MultiModalDataset(Dataset):
    def __init__(self, base_path: Path, data_name: str, cfg) -> None:
        self.base_path = base_path
        self.cfg = cfg
        self.df = self._load_data(base_path, data_name)
        self.text_extractor = TextFeatureExtractor(self.cfg.data.text_feature.method) if self.cfg.data.text_feature.use else None
        self.caption_extractor = TextFeatureExtractor(self.cfg.data.caption_feature.method) if self.cfg.data.caption_feature.use else None
        self.image_extractor = ImageFeatureExtractor(self.cfg.data.image_feature.method) if self.cfg.data.image_feature.use else None

    def _load_data(self, base_path: Path, data_name: str) -> pd.DataFrame:
        df = pd.read_csv(base_path / data_name)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # テキスト特徴量抽出
        text_embedding = torch.tensor([])
        if self.text_extractor is not None:
            text = row['text']
            text_embedding = self.text_extractor.extract_features(text)

        # キャプション特徴量抽出
        caption_embedding = torch.tensor([])
        if self.caption_extractor is not None:
            caption = row.get('imgcaption', "")
            if not isinstance(caption, str) or not caption.strip():
                logging.warning(f"Invalid or empty caption for row: {row}")
                caption = text
            caption_embedding = self.caption_extractor.extract_features(caption)

        # 画像特徴量抽出
        image_embedding = torch.tensor([])
        if self.image_extractor is not None:
            image_id = row['id']
            image_path = self.base_path / 'selected_public_image_set' / f"{image_id}.jpg"
            image_embedding = self.image_extractor.extract_features(image_path)

        label = torch.tensor(row['2_way_label'], dtype=torch.long)

        return {
            'text_embedding': text_embedding,
            'image_embedding': image_embedding,
            'caption_embedding': caption_embedding,
            'label': label
        }