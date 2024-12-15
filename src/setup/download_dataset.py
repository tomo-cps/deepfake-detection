import os
import gdown
import zipfile

from utils.logger import setup_logger
logger = setup_logger(__name__)

def download_data_from_gdrive():
    try:
        file_id = "1zJ8_d6B62ZnuOqRhie6kTgzBtHDbCJbp"  # 実際のファイルIDを記入
        url = f"https://drive.google.com/uc?id={file_id}"
        zip_path = "data/fakenews_data_241214.zip"
        extract_to = "data"

        # ZIPファイルのダウンロード
        if not os.path.exists(zip_path):
            logger.info(f"Downloading {zip_path} from Google Drive...")
            gdown.download(url, zip_path, quiet=False)
            logger.info("Download completed.")
        else:
            logger.info(f"{zip_path} already exists. Skipping download.")

        # ZIPファイルの解凍
        if os.path.exists(zip_path):
            logger.info(f"Unzipping {zip_path} to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info("Unzipping completed.")

            # 解凍後にZIPファイルを削除
            logger.info(f"Removing the ZIP file: {zip_path}")
            os.remove(zip_path)
            logger.info("ZIP file removed.")
        else:
            logger.warning(f"{zip_path} not found. Cannot unzip.")
    except Exception as e:
        logger.exception("An error occurred during the data download or extraction process.")

if __name__ == "__main__":
    download_data_from_gdrive()