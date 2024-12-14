import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    セットアップ済みのロガーを返す関数。
    Args:
        name (str): ロガーの名前（モジュール名など）。
        log_file (str, optional): ログを保存するファイルパス。指定がなければコンソールに出力。
        level (int): ログレベル（例: logging.INFO）。
    Returns:
        logging.Logger: セットアップされたロガー。
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # ロガーの生成
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー（オプション）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

default_logger = setup_logger(
    name="default_logger",
    log_file=os.path.join("output", "logs", f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"),
    level=logging.DEBUG
)
