import os
import gdown
import zipfile

def download_data_from_gdrive():
    # Google Drive上のファイルの共有リンクからidを取得する必要があります。
    # 例えば、共有リンクが
    # "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQRsTuVaWxyz/view?usp=sharing"
    # の場合、idは "1AbCdEfGhIjKlMnOpQRsTuVaWxyz" となります。
    file_id = "1zJ8_d6B62ZnuOqRhie6kTgzBtHDbCJbp"  # 実際のファイルIDを記入
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = "data/fakenews_data_241214.zip"
    extract_to = "data"

    # ZIPファイルのダウンロード
    if not os.path.exists(zip_path):
        print(f"Downloading {zip_path} from Google Drive...")
        gdown.download(url, zip_path, quiet=False)
        print("Download completed.")
    else:
        print(f"{zip_path} already exists. Skipping download.")

    # ZIPファイルの解凍
    if os.path.exists(zip_path):
        print(f"Unzipping {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzipping completed.")

        # 解凍後にZIPファイルを削除
        print(f"Removing the ZIP file: {zip_path}")
        os.remove(zip_path)
        print("ZIP file removed.")
    else:
        print(f"{zip_path} not found. Cannot unzip.")

if __name__ == "__main__":
    download_data_from_gdrive()
