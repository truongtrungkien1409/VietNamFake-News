import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_csv: str, output_dir: str = "data/splits", random_state: int = 42):
    """
    Phân chia tập dữ liệu chuẩn 80% Train, 10% Validation, 10% Test theo tỷ lệ phân tầng (Stratified Split).
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_csv):
        print(f"⚠️ Không tìm thấy file dữ liệu tại {input_csv}. Vui lòng đặt file CSV chứa cột 'text' và 'label'.")
        return

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    # Bước 1: Tách Train (80%) và Temp (20%)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=random_state, stratify=df['label']
    )

    # Bước 2: Tách Temp thành Validation (10%) và Test (10%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['label']
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"✅ Đã phân chia dữ liệu thành công:")
    print(f"   - Train (80%): {len(train_df)} mẫu -> {train_path}")
    print(f"   - Validation (10%): {len(val_df)} mẫu -> {val_path}")
    print(f"   - Test (10%): {len(test_df)} mẫu -> {test_path}")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/vn_news_merged.csv"
    split_dataset(csv_file)
