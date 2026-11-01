import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.config import get_paths

def extract_embeddings_for_split(csv_path: str, output_prefix: str):
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy file {csv_path}. Bỏ qua trích xuất đặc trưng cho split này.")
        return

    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("File CSV phải chứa hai cột 'text' và 'label'.")

    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(int).values

    print(f"🤖 Đang tải mô hình {MODEL_NAME} (đóng băng trọng số, requires_grad=False)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Kiểm tra thử nếu có model cache cục bộ hoặc tải từ hub
    try:
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        print(f"⚠️ Không thể tải từ hub (có thể do giới hạn ổ cứng). Lỗi: {e}")
        raise e

    model.eval()
    # Đóng băng hoàn toàn trọng số PhoBERT
    for param in model.parameters():
        param.requires_grad = False

    all_embeddings = []
    print(f"🔄 Đang trích xuất đặc trưng [CLS] (768 chiều) cho {len(texts)} mẫu từ {csv_path}...")
    start_time = time.time()

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        if (i // BATCH_SIZE + 1) % 10 == 0 or (i + BATCH_SIZE >= len(texts)):
            print(f"   - Đã xử lý {min(i + BATCH_SIZE, len(texts))}/{len(texts)} mẫu...")

    features = np.concatenate(all_embeddings, axis=0)
    
    paths = get_paths()
    os.makedirs(paths.saved_models, exist_ok=True)
    np.save(f"{output_prefix}_features_768.npy", features)
    np.save(f"{output_prefix}_labels.npy", labels)
    print(f"✅ Hoàn thành! Đã lưu {output_prefix}_features_768.npy ({features.shape}) và {output_prefix}_labels.npy trong {time.time() - start_time:.2f} giây.")

if __name__ == "__main__":
    paths = get_paths()
    for split in ["train", "val", "test"]:
        csv_p = os.path.join(paths.splits, f"{split}.csv")
        prefix = os.path.join(paths.splits, split)
        extract_embeddings_for_split(csv_p, prefix)
