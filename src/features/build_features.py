# features/build_features.py
import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
from torchvision import transforms, models
import joblib
import torch.nn as nn
Image.MAX_IMAGE_PIXELS = None

# BLIP関連
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################
# テキスト特徴抽出クラス
########################################

class TextFeatureExtractor:
    def __init__(self, method: str):
        self.method = method

        if self.method == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
            self.model.eval()

        elif self.method == "TF-IDF":
            pass

        elif self.method == "LSTM":
            # 学習済みのLSTMモデルをロード
            # LSTMモデルはテキストをトークナイズ後にEmbedding→LSTMで変換し、
            # 最終的に768次元程度の特徴ベクトルを返す想定
            self.lstm_model = torch.load("lstm_model.pth", map_location=device)
            self.lstm_model.eval()
            # トークナイザ等が必要ならここで初期化
            # self.lstm_tokenizer = ... # 簡略化のため省
            pass

        else:
            raise ValueError(f"Unknown text feature extraction method: {self.method}")

    def extract_features(self, text: str) -> torch.Tensor:
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
            # TF-IDFを適用
            vec = self.vectorizer.transform([text])  # shape: (1, tfidf_dim)
            vec = torch.tensor(vec.toarray(), dtype=torch.float).to(device).squeeze(0)
            # vecはtfidf_dim次元(例: 3000次元など)
            return vec

        elif self.method == "LSTM":
            # テキストをLSTMモデルで特徴抽出
            # 簡易的にスペース分割のエンコーディングなど（実運用は適切なトークナイザ必須）
            tokens = text.split()[:128]
            # ここでtokensをid化しembeddingしてLSTMに通す処理が必要だが省略
            # 仮にlstm_modelが tokens(list of str) -> 768-dim tensor を直接返すと仮定
            with torch.no_grad():
                emb = self.lstm_model(tokens)
            return emb

        else:
            return torch.zeros(768)


########################################
# 画像特徴抽出クラス
########################################

def create_image_feature_extractor(model: nn.Module, remove_last_layer: bool = True) -> nn.Module:
    # 分類層を取り除き、中間特徴を取得するための関数
    # モデルによって最後の層が異なるためモデルごとに処理分岐
    # ResNetはmodel.fcが最終層
    # VGGはmodel.classifier[-1]が最終層
    # DenseNetはmodel.classifierが最終層
    # ViTはmodel.headsが最終層
    # ここでは単純に最後の分類層を除去
    # なお、ViTの場合、出力はcls_tokenの埋め込み(768次元)が得られる設計
    
    # ResNet
    if isinstance(model, models.ResNet):
        # 最終fc層を除く
        layers = list(model.children())[:-1]
        feature_model = nn.Sequential(*layers)
        return feature_model

    # VGG-16
    if isinstance(model, models.VGG):
        # classifierの最後(fc8)を除いた部分まで
        feature_model = nn.Sequential(
            *list(model.features.children()),
            nn.Flatten(),
            *list(model.classifier.children())[:-1] # 最終層(1000クラスのfc)を除く -> 4096次元
        )
        return feature_model

    # DenseNet-201
    if isinstance(model, models.DenseNet):
        # classifierを除く
        # DenseNetのfeatures出力は(?, 1920, 7,7), avgpool後(?,1920)
        # classifier前が特徴ベクトル1920次元
        feature_model = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        return feature_model

    # ViT-L/32
    # Vision Transformerはmodel.headsが最後の分類層
    # clsトークン埋め込みはmodel.forward_features(image_tensor)で得られる
    # 適当にクラスをラップして特徴のみ返す
    class ViTFeatureModel(nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit = vit_model
        def forward(self, x):
            # forward_featuresが内部でCLSトークンを抽出する
            # 出力は clsトークン埋め込み 768次元程度
            with torch.no_grad():
                features = self.vit.forward_features(x) # (B, 768)
            return features

    if hasattr(model, 'heads'):
        # vitモデル
        return ViTFeatureModel(model)

    # 万一対応外モデル
    return model


class ImageFeatureExtractor:
    def __init__(self, method: str):
        self.method = method
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if self.method == "ResNet-50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
            base_model.eval()
            self.feature_model = create_image_feature_extractor(base_model)
            # ResNet-50の特徴ベクトルは2048次元
            
        elif self.method == "VGG-16":
            base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
            base_model.eval()
            self.feature_model = create_image_feature_extractor(base_model)
            # VGG-16特徴は4096次元

        elif self.method == "DenseNet-201":
            base_model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1).to(device)
            base_model.eval()
            self.feature_model = create_image_feature_extractor(base_model)
            # DenseNet-201は1920次元特徴

        elif self.method == "ViT-L/32":
            # Vision Transformer の large/32 モデル
            # PyTorch 2.0+であれば
            # vit_l_32 = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
            # ここでは仮に存在するとして記述
            vit_l_32 = models.vit_l_32(weights="IMAGENET1K_V1").to(device)
            vit_l_32.eval()
            self.feature_model = create_image_feature_extractor(vit_l_32)
            # ViT-L/32は768次元程度
            
        else:
            raise ValueError(f"Unknown image feature extraction method: {self.method}")

    def extract_features(self, image_path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        if image.mode == "P" and "transparency" in image.info:
            print(f"Warning: Image with transparency found at {image_path}")
        image_tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = self.feature_model(image_tensor)
            # featuresはモデルによって次元が異なる
            # ResNet-50: (1, 2048)
            # VGG-16: (1, 4096)
            # DenseNet-201: (1, 1920)
            # ViT-L/32: (1, 768)
            # squeezeで(次元数,)に
            return features.squeeze()
