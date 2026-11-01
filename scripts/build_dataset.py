import os
import sys
import pandas as pd
from src.preprocessing.split_data import split_dataset

def build_dataset():
    raw_path = "data/raw/vn_news_merged.csv"
    if not os.path.exists(raw_path):
        print("Không tìm thấy dữ liệu thô, hãy đặt file thật vào data/raw/")
        sys.exit(1)
    split_dataset(raw_path)

if __name__ == "__main__":
    build_dataset()
