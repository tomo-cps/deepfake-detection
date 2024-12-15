import logging
# import sys

def setup_logger(__name__):
    return logging.getLogger(__name__)

# def setup_logger(name: str = None, log_level: int = logging.INFO, log_file: str = None) -> logging.Logger:
#     """
#     ロガーの共通設定を行う関数。

#     Args:
#         name (str): ロガー名（デフォルトはルートロガーを使用）。
#         log_level (int): ログレベル（デフォルト: INFO）。
#         log_file (str): ログファイルのパス（指定がある場合、ファイルにも出力）。

#     Returns:
#         logging.Logger: 設定済みのロガー。
#     """
#     # ルートロガーからすべてのハンドラを削除（干渉を防ぐ）
#     for handler in logging.root.handlers[:]:
#         logging.root.removeHandler(handler)

#     logger = logging.getLogger(name)
#     logger.setLevel(log_level)

#     # すでにハンドラが設定されている場合は再設定を防ぐ
#     if logger.hasHandlers():
#         return logger

#     # コンソール出力用のハンドラ
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(log_level)
#     console_formatter = logging.Formatter(
#         "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     )
#     console_handler.setFormatter(console_formatter)
#     logger.addHandler(console_handler)

#     # ファイル出力用のハンドラ（必要な場合）
#     if log_file:
#         file_handler = logging.FileHandler(log_file, encoding="utf-8")
#         file_handler.setLevel(log_level)
#         file_formatter = logging.Formatter(
#             "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#         )
#         file_handler.setFormatter(file_formatter)
#         logger.addHandler(file_handler)

#     return logger
