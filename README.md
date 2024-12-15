# Multimodal Deepfake Detection
## 使用方法

### 1. 基本的な実行方法


### 2.モデルの選択
```
base_path: "/mnt/data"

data:
  text_feature:
    use: true
    method: BERT        # TF-IDF, LSTM, BERT
  image_feature:
    use: true
    method: ResNet-50   # VGG-16, DenseNet-201, ViT-L/32, ResNet-50
  caption_feature:
    use: true
    method: BERT        # TF-IDF, LSTM, BERT, BERT+BLIP

  train_data: "train_fake_news.csv"
  val_data: "val_fake_news.csv"
  test_data: "test_fake_news.csv"
  batch_size: 256

model:
  type: multi_modal_with_caption_using_cross_attention  # multi_modal, multi_modal_with_caption, multi_modal_with_caption_using_cross_attention
  input_size:
    text: 768
    image: 2048
    caption: 768
  hidden_size: 512
  output_size: 1
  num_heads: 3  # Cross-Attentionに使用（必要な場合）

training:
  learning_rate: 1e-3
  patience: 3
  num_epochs: 1

optimize: false # true false
optuna:
  n_trials: 1 #20
  trial_epochs: 1 #5
  eval_epochs: 1 #1
```


### 3. 入力データ形式


### 4. 出力データ形式

### 5. ファイル構成

```
.
├── README.md
├── data
│   ├── raw/                # 生データ
│   └── processed/          # 前処理済みデータや中間ファイル
├── output
│   ├── logs/               # ログファイル
│   ├── models/             # 学習済みモデルの保存先
│   ├── figures/            # 可視化結果の出力先 (Attention可視化画像など)
│   └── predictions/        # 予測結果保存先
├── configs
│   ├── default_config.yaml # デフォルト設定
│   ├── model_config.yaml   # モデルパラメータ
│   └── data_config.yaml    # データ関連パラメータ
├── requirements.txt
└── src
    ├── main.py                      # 全体のエントリーポイント (学習・推論・評価を起動)
    ├── dataloaders
    │   ├── __init__.py
    │   └── dataset.py              # データセット定義やデータローダークラス
    ├── features
    │   ├── __init__.py
    │   └── build_features.py       # 特徴量抽出や前処理スクリプト
    ├── models
    │   ├── __init__.py
    │   ├── model_arch.py           # モデルアーキテクチャ定義
    │   ├── train.py                # モデル学習用スクリプト
    │   ├── eval.py                 # 評価用スクリプト (trainやpredictと同階層に)
    │   └── inference.py            # 推論用スクリプト (predict_model.pyを一般化)
    ├── utils
    │   ├── __init__.py
    │   ├── logger.py               # ログ設定
    │   ├── config.py               # コンフィグ読み込みユーティリティ
    │   └── helpers.py              # その他共通関数
    └── visualization
        ├── __init__.py
        ├── visualize.py            # 一般的な可視化機能
        └── visualize_attention.py  # Attentionマップ等、説明性に特化した可視化

# (オプション) テスト用ディレクトリ
# tests/
#   ├── __init__.py
#   ├── test_dataloaders.py
#   ├── test_features.py
#   ├── test_models.py
#   └── test_utils.py

# (オプション) Notebook用
# notebooks/
#   ├── data_exploration.ipynb
#   └── model_analysis.ipynb

```
