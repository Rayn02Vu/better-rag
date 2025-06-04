import requests
from pathlib import Path
import datetime
import logging

def download_docs(name, url):
    OUTPUT_DIR = Path("./data")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    txt_path = OUTPUT_DIR / f"{name}.txt"

    response = requests.get(url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"PDF downloaded to: {pdf_path}")
    else:
        print(f"Failed to download PDF: {response.status_code}")
        exit(1)

if __name__ == "__main__":
    download_docs("LichsuDang", "https://www.iuv.edu.vn/cms/plugin_upload/preview/news/1468/1113/gt-lich-su-dang-csvn-ban-tuyen-giao-tw.pdf")