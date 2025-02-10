import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def download_data(data_urls, raw_dir="data/raw"):
    """下载数据并保存到 raw 文件夹"""
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    for url in data_urls:
        file_name = url.split("/")[-1]  # 从URL中提取文件名
        save_path = os.path.join(raw_dir, file_name)
        print(f"Downloading {file_name}...")
        download_file(url, save_path)
        print(f"Saved to {save_path}")

# 示例：从网上下载数据
if __name__ == "__main__":
    data_urls = [
        "https://example.com/dataset1.zip",
        "https://example.com/dataset2.zip"
    ]
    download_data(data_urls)
