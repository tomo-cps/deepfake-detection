import json
from pathlib import Path
from typing import Dict, List

# データの選択
# モデルの選択




def main():
    # データの選択
    # モデルの選択
    # 学習させるか推論か
    input_path = Path("data/MedB_MedC_updated_20241205.json")
    prompt_type = PromptType.ZERO_SHOT  # 必要に応じて変更可能
    parallel_mode = True  # True: 並列実行、False:逐次実行

    # プロンプトタイプに基づいて出力パスを生成
    output_path = _get_output_path(input_path, prompt_type)

    client = create_client("openai")
    service = MedicalCorrectionService(client)
    console = Console()

    processor = MedicalTextProcessor(
        service,
        console,
        prompt_type=prompt_type,
        parallel_mode=parallel_mode
    )
    processor.process_file(input_path, output_path)

if __name__ == "__main__":
    main()
