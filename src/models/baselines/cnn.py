import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from src.models.baselines.lstm import SimpleTextDataset, build_vocab, SEED
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.config import get_paths

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, kernel_sizes=[3, 4, 5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (ks, embed_dim)) for ks in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)
        conved = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(cv, dim=2)[0] for cv in conved]
        cat = torch.cat(pooled, dim=1)
        logits = self.fc(cat)
        return self.sigmoid(logits).squeeze(1)

def run_cnn_baseline(
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    epochs: int = 15,
    batch_size: int = 32,
    max_samples: int = None,
    metrics_output: str = None,
    output_csv: str = None,
    pred_csv: str = None,
    cm_path: str = None,
    model_path: str = None,
    min_freq: int = 2,
    max_words: int = 30000,
    **kwargs
):
    paths = get_paths()
    if train_csv is None: train_csv = os.path.join(paths.splits, "train.csv")
    if val_csv is None: val_csv = os.path.join(paths.splits, "val.csv")
    if test_csv is None: test_csv = os.path.join(paths.splits, "test.csv")
    if metrics_output is None: metrics_output = output_csv or os.path.join(paths.tables, "cnn_metrics.csv")
    if pred_csv is None: pred_csv = os.path.join(paths.tables, "cnn_preds.csv")
    if cm_path is None: cm_path = os.path.join(paths.figures, "cm_cnn.png")
    if model_path is None: model_path = os.path.join(paths.saved_models, "cnn_model.pt")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        print("⚠️ Thiếu file dữ liệu train/val/test cho CNN baseline.")
        return {}

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    val_df = pd.read_csv(val_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples)
        test_df = test_df.head(max_samples)

    vocab = build_vocab(train_df['text'].values, min_freq=min_freq, max_words=max_words)

    train_dataset = SimpleTextDataset(train_df['text'].values, train_df['label'].values, vocab)
    val_dataset = SimpleTextDataset(val_df['text'].values, val_df['label'].values, vocab)
    test_dataset = SimpleTextDataset(test_df['text'].values, test_df['label'].values, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size=len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_t = time.time()
    best_val_f1 = -1.0
    patience = 3
    patience_counter = 0
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device)
                probs = model(x_v).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                val_preds.extend(preds)
                val_true.extend(y_v.numpy())

        _, _, f_cls, _ = precision_recall_fscore_support(val_true, val_preds, labels=[0, 1], zero_division=0)
        val_f1_fake = f_cls[1] if len(f_cls) > 1 else 0.0

        if val_f1_fake > best_val_f1:
            best_val_f1 = val_f1_fake
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 CNN Early stopping tại epoch {epoch+1} (Val F1 Fake: {val_f1_fake*100:.2f}%)")
                break

    train_time = time.time() - start_t

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    model.eval()
    all_preds, all_probs, all_true = [], [], []
    start_pred = time.time()
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            probs = model(x_b).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_b.numpy())
    pred_time = time.time() - start_pred
    inference_time_per_sample = pred_time / len(test_df) if len(test_df) > 0 else 0.0

    metrics = compute_metrics(all_true, all_preds, all_probs)
    metrics["training_time_sec"] = round(train_time, 2)
    metrics["inference_sec_per_sample"] = round(inference_time_per_sample, 6)

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output, index=False)

    df_preds = pd.DataFrame({
        "original_text": test_df['original_text'].tolist() if 'original_text' in test_df.columns else test_df['text'].tolist(),
        "text": test_df['text'].tolist(),
        "true_label": all_true,
        "predicted_label": all_preds,
        "fake_probability": all_probs
    })
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    df_preds.to_csv(pred_csv, index=False)

    save_confusion_matrix(all_true, all_preds, cm_path)
    return metrics

train_and_evaluate_cnn = run_cnn_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    run_cnn_baseline(epochs=args.epochs, max_samples=args.max_samples)
