
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from src.config import get_paths

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _timing_path() -> str:
    return os.path.join(get_paths().tables, "phobert_extraction_timing.csv")


def _write_timing(split: str, num_samples: int, duration_sec: float):
    timing_path = _timing_path()
    os.makedirs(os.path.dirname(timing_path), exist_ok=True)
    row = pd.DataFrame([{
        "split": split,
        "num_samples": num_samples,
        "device": str(device),
        "duration_sec": round(duration_sec, 4),
        "samples_per_sec": round(num_samples / duration_sec, 4) if duration_sec > 0 else 0.0,
    }])
    if os.path.exists(timing_path):
        existing = pd.read_csv(timing_path)
        existing = existing[existing["split"] != split]
        row = pd.concat([existing, row], ignore_index=True)
    row.sort_values("split").to_csv(timing_path, index=False, encoding="utf-8")
    print(f"Timing: {timing_path}")


def extract_embeddings_for_split(csv_path: str, output_prefix: str, force: bool = False):

    if not os.path.exists(csv_path):
        print(f"Khong tim thay: {csv_path}")
        return

    # Duong dan output
    feature_path = f"{output_prefix}_features_768.npy"
    label_path = f"{output_prefix}_labels.npy"

    # Neu da co feature va label thi bo qua
    if not force and os.path.exists(feature_path) and os.path.exists(label_path):
        print(f"Da co feature: {feature_path}. Bo qua.")
        return

    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV phai co cot text va label")

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).values

    print(f"Device: {device}")
    print(f"Loading PhoBERT offline: {MODEL_NAME}")

    # Load model tu cache, KHONG truy cap HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    ).to(device)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    # Cap phat mang embedding
    features = np.empty(
        (len(texts), 768),
        dtype=np.float32
    )

    print(f"Extracting {len(texts)} samples...")
    start_time = time.time()

    for i in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            outputs = model(**encoded)

            cls_emb = outputs.last_hidden_state[:, 0, :]

            features[i:i + len(batch_texts)] = (
                cls_emb.cpu().numpy()
            )

        processed = min(i + BATCH_SIZE, len(texts))

        print(
            f"\rProgress: {processed}/{len(texts)} "
            f"({processed / len(texts) * 100:.1f}%)",
            end="",
            flush=True
        )

    print()

    # Luu feature va label
    os.makedirs(os.path.dirname(feature_path), exist_ok=True)

    np.save(feature_path, features)
    np.save(label_path, labels)

    print(f"Feature: {features.shape}")
    print(f"Label: {labels.shape}")
    print(f"Saved: {feature_path}")
    duration_sec = time.time() - start_time
    print(f"Time: {duration_sec:.2f}s")
    split = os.path.splitext(os.path.basename(output_prefix))[0]
    _write_timing(split, len(texts), duration_sec)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trích xuất embedding PhoBERT cho các split.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Trích xuất lại kể cả khi file embedding đã tồn tại để đo thời gian thực tế."
    )
    args = parser.parse_args()

    paths = get_paths()

    for split in ["train", "val", "test"]:

        csv_path = os.path.join(
            paths.splits,
            f"{split}.csv"
        )

        output_prefix = os.path.join(
            paths.splits,
            split
        )

        extract_embeddings_for_split(
            csv_path,
            output_prefix,
            force=args.force
        )
