import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocess import preprocess_dataframe

from src.config import get_paths

def split_dataset(input_csv: str, output_dir: str = None, processed_dir: str = None, random_state: int = 42):
    """
    Đọc dữ liệu thô, tiến hành tiền xử lý 5 bước toàn bộ (lưu clean.csv vào processed_dir),
    sau đó phân chia phân tầng (Stratified Split) 80% Train, 10% Validation, 10% Test.
    """
    paths = get_paths()
    if output_dir is None: output_dir = paths.splits
    if processed_dir is None: processed_dir = paths.processed

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        print(f"⚠️ Không tìm thấy file dữ liệu tại {input_csv}. Vui lòng đặt file CSV chứa cột 'text' và 'label'.")
        return

    print(f"📂 Đang đọc dữ liệu thô từ {input_csv}...")
    df_raw = pd.read_csv(input_csv)
    
    # Tiền xử lý 5 bước chuẩn hóa và loại bỏ trùng lặp trước khi split (tránh rò rỉ dữ liệu)
    df_clean = preprocess_dataframe(df_raw, text_col="text", label_col="label")
    
    clean_csv_path = os.path.join(processed_dir, "clean.csv")
    df_clean.to_csv(clean_csv_path, index=False, encoding="utf-8")
    print(f"💾 Đã lưu dữ liệu sạch toàn bộ tại {clean_csv_path} ({len(df_clean)} mẫu)")

    # Phân chia phân tầng 80% Train, 20% Temp (để tách Val 10% và Test 10%)
    train_df, temp_df = train_test_split(
        df_clean, test_size=0.2, random_state=random_state, stratify=df_clean['label']
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['label']
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"\n📊 THỐNG KÊ PHÂN BỐ TẬP DỮ LIỆU (So sánh Bảng 2.2):")
    for name, dset in [("Train (80%)", train_df), ("Validation (10%)", val_df), ("Test (10%)", test_df)]:
        n_total = len(dset)
        n_real = (dset['label'] == 0).sum()
        n_fake = (dset['label'] == 1).sum()
        print(f"   - {name}: {n_total} mẫu | Tin thật (0): {n_real} ({n_real/n_total*100:.1f}%) | Tin giả (1): {n_fake} ({n_fake/n_total*100:.1f}%)")

if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/vn_news_merged.csv"
    split_dataset(csv_file)
