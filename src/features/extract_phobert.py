import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from src.preprocessing.preprocess import TextPreprocessor

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_embeddings_for_split(csv_path: str, output_prefix: str):
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy file {csv_path}. Bỏ qua trích xuất đặc trưng cho split này.")
        return

    df = pd.read_csv(csv_path)
    preprocessor = TextPreprocessor(use_word_segmentation=True)
    
    print(f"🧹 Đang tiền xử lý {len(df)} mẫu từ {csv_path}...")
    texts = [preprocessor.clean_text(t) for t in df['text'].astype(str).tolist()]
    labels = df['label'].astype(int).values

    print(f"🤖 Đang tải mô hình {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    all_embeddings = []
    print(f"🔄 Đang trích xuất đặc trưng [CLS] (768 chiều)...")
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
    
    os.makedirs("results/saved_models", exist_ok=True)
    np.save(f"{output_prefix}_features_768.npy", features)
    np.save(f"{output_prefix}_labels.npy", labels)
    print(f"✅ Hoàn thành! Đã lưu {output_prefix}_features_768.npy ({features.shape}) và {output_prefix}_labels.npy trong {time.time() - start_time:.2f} giây.")

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        csv_p = f"data/splits/{split}.csv"
        prefix = f"data/splits/{split}"
        extract_embeddings_for_split(csv_p, prefix)
