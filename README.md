# Multimodal Deepfake Detection
## 使用方法

### 1. 基本的な実行方法

```
pip install -r requirements.txt
```
```
python src/main.py
```
### 2.モデルの選択

- configs/config.yaml で変更することができます．


### 3. 入力データ形式

|    text     |      id     | imgcaption  | 2_way_label | 6_way_label |
|-------------|-------------|-------------|-------------|-------------|
|    `str`     |   `str`    |     `str`   |    `int`    |    `int`    |

- **`text`**: 検出対象のテキスト
- **`id`**: 検出対象の画像 id
- **`imgcaption`**: 検出対象の画像キャプション
- **`2_way_label`**: 2 種類のラベル
  - `0`True: リアルニュース
  - `1`False: フェイクニュース
- **`6_way_label`**: 6 種類のラベル
  - `0`True: 事実に基づいた正確な情報
  - `1`Satire/Parody: 風刺的やパロディ的な要素を含む，虚偽の情報
  - `2`Misleading Content: 意図的にユーザーを誤導する情報
  - `3`Imposter Content: ボットによって生成された情報
  - `4`False Connection: テキストと画像の内容が一致しない情報
  - `5`Manipulated Content: 意図的に編集された情報

### 4. 出力データ形式
`./outputs/{実行した日にち}/{保存した時間}/`
- `.hydra/config.yaml`
  - 実行時の config.yaml 情報
- `output/`
  - `checkpoints/{検出モデル名}.pth` モデル保存パス
- `results/`
  - `metrics_{検出モデル名}.json` モデル保存パス
```
{
    "time": "実行時の時間",
    "optimize": true or false,
    "method": "検出モデル名",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score
}
```
  - `predictions_{config.yamlのmodel.typeの検出モデル名}.csv`
```
true_label,prediction
1,1
0,1
1,1
```
- `main.log`
```
[2024-12-17 02:23:25,841][optimization.optuna_tuner][INFO] - Starting Optuna hyperparameter optimization...
[2024-12-17 02:23:25,841][optimization.optuna_tuner][INFO] - Running training using Fake News Detection using "multi_modal" Model
[2024-12-17 02:23:25,842][optimization.objective][INFO] - Starting trial 0:
[2024-12-17 02:23:25,842][optimization.objective][INFO] - Proposed hyperparameters - Learning Rate: 2.9811659606981014e-05, Batch Size: 128, Hidden Size: 256
[2024-12-17 02:23:30,073][optimization.objective][INFO] - Using MultiModalClassifier model
[2024-12-17 02:23:30,074][optimization.objective][INFO] - Model loaded to cuda
```
  - Terminal に出力された log 情報


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
