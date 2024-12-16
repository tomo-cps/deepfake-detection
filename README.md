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
  - {`0`: True} リアルニュース
  - {`1`: False} フェイクニュース
- **`6_way_label`**: 6 種類のラベル
  - {`0`: True} 事実に基づいた正確な情報
  - {`1`: Satire/Parody} 風刺的やパロディ的な要素を含む，虚偽の情報
  - {`2`: Misleading Content} 意図的にユーザーを誤導する情報
  - {`3`: Imposter Content} ボットによって生成された情報
  - {`4`: False Connection} テキストと画像の内容が一致しない情報
  - {`5`: Manipulated Content} 意図的に編集された情報

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
Terminal に出力された log 情報
```
[2024-12-17 02:23:25,841][optimization.optuna_tuner][INFO] - Starting Optuna hyperparameter optimization...
[2024-12-17 02:23:25,841][optimization.optuna_tuner][INFO] - Running training using Fake News Detection using "multi_modal" Model
[2024-12-17 02:23:25,842][optimization.objective][INFO] - Starting trial 0:
[2024-12-17 02:23:25,842][optimization.objective][INFO] - Proposed hyperparameters - Learning Rate: 2.9811659606981014e-05, Batch Size: 128, Hidden Size: 256
[2024-12-17 02:23:30,073][optimization.objective][INFO] - Using MultiModalClassifier model
[2024-12-17 02:23:30,074][optimization.objective][INFO] - Model loaded to cuda
```



### 5. ファイル構成

```
.
├── README.md
├── data
│   ├── raw/                
│   └── processed/          
├── output
│   ├── checkpoints/         
│   ├── results/             
├── configs
│   ├── config.yaml 
├── requirements.txt
├── scripts
│   ├── download_data.sh
│   ├── inference_pipeline.sh
│   └── run_training.sh
└── src
    ├── dataloaders
    │   ├── __init__.py
    │   └── dataset.py
    ├── features
    │   ├── __init__.py
    │   └── build_features.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   ├── eval.py
    │   ├── inference.py
    │   ├── model_arch.py
    │   └── train.py
    ├── optimization
    │   ├── __init__.py
    │   ├── objective.py
    │   └── optuna_tuner.py
    ├── setup
    │   └── download_dataset.py
    ├── utils
    │   ├── __init__.py
    │   ├── config.py
    │   ├── helpers.py
    │   └── logger.py
    └── visualization
        ├── __init__.py
        └── visualize.py
```
