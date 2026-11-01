import os
import pandas as pd
from src.preprocessing.split_data import split_dataset

def build_dataset():
    os.makedirs("data/raw", exist_ok=True)
    raw_path = "data/raw/vn_news_merged.csv"
    
    if not os.path.exists(raw_path):
        print(f"⚠️ Không tìm thấy {raw_path}. Đang tạo dữ liệu mẫu thô ban đầu...")
        texts = [
            f"Bản tin thời sự chính trị xã hội phản ánh tình hình phát triển kinh tế đất nước năm {2020 + i%5} với nhiều thành tựu quan trọng đáng ghi nhận trong nước và quốc tế." for i in range(100)
        ] + [
            f"Tin nóng lan truyền trên mạng xã hội về sự kiện giật gân chưa được kiểm chứng số {i} khiến dư luận xôn xao bàn tán." for i in range(100)
        ]
        labels = [0]*100 + [1]*100
        df = pd.DataFrame({"text": texts, "label": labels})
        df.to_csv(raw_path, index=False, encoding="utf-8")
        print(f"✅ Đã tạo file dữ liệu thô mẫu tại {raw_path}")

    split_dataset(raw_path)

if __name__ == "__main__":
    build_dataset()
