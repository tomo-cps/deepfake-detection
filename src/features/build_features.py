import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from utils.logger import setup_logger
logger = setup_logger(__name__)

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextFeatureExtractor:
    def __init__(self, method: str):
        self.method = method

        if self.method == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
            self.model.eval()
            # logger.info("Initialized BERT model for text feature extraction.")

        elif self.method == "TF-IDF":
            logger.warning("TF-IDF extraction is not fully implemented.")

        elif self.method == "LSTM":
            try:
                self.lstm_model = torch.load("lstm_model.pth", map_location=device)
                self.lstm_model.eval()
                # logger.info("Loaded pre-trained LSTM model for text feature extraction.")
            except Exception as e:
                logger.exception("Failed to load LSTM model.")
                raise

        else:
            # logger.error(f"Unknown text feature extraction method: {self.method}")
            raise ValueError(f"Unknown text feature extraction method: {self.method}")

    def extract_features(self, text: str) -> torch.Tensor:
        try:
            if self.method == "BERT":
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding='max_length', 
                    truncation=True, 
                    max_length=64
                ).to(device)
                with torch.no_grad():
                    emb = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze()
                return emb

            elif self.method == "TF-IDF":
                vec = self.vectorizer.transform([text])  # shape: (1, tfidf_dim)
                vec = torch.tensor(vec.toarray(), dtype=torch.float).to(device).squeeze(0)
                return vec

            elif self.method == "LSTM":
                tokens = text.split()[:128]
                with torch.no_grad():
                    emb = self.lstm_model(tokens)
                return emb

            else:
                logger.warning(f"No extraction method matched for text: {text}")
                return torch.zeros(768)

        except Exception as e:
            logger.exception(f"Error during text feature extraction for input: {text}")
            raise

def create_image_feature_extractor(model: nn.Module, remove_last_layer: bool = True) -> nn.Module:
    try:
        if isinstance(model, models.ResNet):
            layers = list(model.children())[:-1]
            feature_model = nn.Sequential(*layers)
            return feature_model

        if isinstance(model, models.VGG):
            feature_model = nn.Sequential(
                *list(model.features.children()),
                nn.Flatten(),
                *list(model.classifier.children())[:-1]
            )
            return feature_model

        if isinstance(model, models.DenseNet):
            feature_model = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Flatten()
            )
            return feature_model

        class ViTFeatureModel(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model
            def forward(self, x):
                with torch.no_grad():
                    features = self.vit.forward_features(x)
                return features

        if hasattr(model, 'heads'):
            return ViTFeatureModel(model)

        logger.warning("Unknown model type for image feature extraction.")
        return model

    except Exception as e:
        logger.exception("Failed to create image feature extractor.")
        raise

class ImageFeatureExtractor:
    def __init__(self, method: str):
        self.method = method
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            if self.method == "ResNet-50":
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
                base_model.eval()
                self.feature_model = create_image_feature_extractor(base_model)
                # logger.info("Initialized ResNet-50 for image feature extraction.")

            elif self.method == "VGG-16":
                base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
                base_model.eval()
                self.feature_model = create_image_feature_extractor(base_model)
                # logger.info("Initialized VGG-16 for image feature extraction.")

            elif self.method == "DenseNet-201":
                base_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).to(device)
                base_model.eval()
                self.feature_model = create_image_feature_extractor(base_model)
                # logger.info("Initialized DenseNet-201 for image feature extraction.")

            elif self.method == "ViT-L/32":
                vit_l_32 = models.vit_l_32(weights="IMAGENET1K_V1").to(device)
                vit_l_32.eval()
                self.feature_model = create_image_feature_extractor(vit_l_32)
                # logger.info("Initialized ViT-L/32 for image feature extraction.")

            else:
                logger.error(f"Unknown image feature extraction method: {self.method}")
                raise ValueError(f"Unknown image feature extraction method: {self.method}")

        except Exception as e:
            logger.exception(f"Failed to initialize image feature extractor with method: {self.method}")
            raise

    def extract_features(self, image_path) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
            if image.mode == "P" and "transparency" in image.info:
                # logger.warning(f"Image with transparency found at {image_path}")
                image = image.convert("RGBA")
            else:
                image = image.convert("RGB")

            image_tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.feature_model(image_tensor)
                return features.squeeze()

        except Exception as e:
            logger.exception(f"Error during image feature extraction for image: {image_path}")
            raise
