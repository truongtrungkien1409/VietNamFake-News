import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from src.models.baselines.lstm import SimpleTextDataset, SEED
from src.evaluation.metrics import compute_metrics, save_confusion_matrix

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
    train_csv: str = "data/splits/train.csv",
    test_csv: str = "data/splits/test.csv",
    epochs: int = 2,
    batch_size: int = 32,
    metrics_output: str = "results/tables/cnn_metrics.csv"
):
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("⚠️ Thiếu file dữ liệu train/test cho CNN baseline.")
        return

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

    vocab = {"<pad>": 0, "<unk>": 1}
    for txt in train_df['text'].astype(str):
        for w in txt.split():
            if w not in vocab:
                vocab[w] = len(vocab)

    train_dataset = SimpleTextDataset(train_df['text'].values, train_df['label'].values, vocab)
    test_dataset = SimpleTextDataset(test_df['text'].values, test_df['label'].values, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size=len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_t = time.time()
    model.train()
    for epoch in range(epochs):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_t

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
    inference_time_per_sample = pred_time / len(test_df)

    metrics = compute_metrics(all_true, all_preds, all_probs)
    metrics["training_time_sec"] = round(train_time, 2)
    metrics["inference_sec_per_sample"] = round(inference_time_per_sample, 4)

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output, index=False)
    return metrics

def train_and_evaluate_cnn(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    output_csv: str = "results/tables/cnn_metrics.csv",
    pred_csv: str = "results/tables/cnn_preds.csv",
    cm_path: str = "results/figures/cm_cnn.png",
    model_path: str = "results/saved_models/cnn_model.pt",
    epochs: int = 1,
    batch_size: int = 32,
    max_samples: int = None
):
    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])
    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples)

    vocab = {"<pad>": 0, "<unk>": 1}
    for txt in train_df['text'].astype(str):
        for w in txt.split():
            if w not in vocab:
                vocab[w] = len(vocab)

    train_dataset = SimpleTextDataset(train_df['text'].values, train_df['label'].values, vocab)
    test_dataset = SimpleTextDataset(test_df['text'].values, test_df['label'].values, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCNN(vocab_size=len(vocab)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            probs = model(x_b).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_b.numpy())

    metrics = compute_metrics(all_true, all_preds, all_probs)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)

    df_pred = pd.DataFrame({"true": all_true, "pred": all_preds, "prob": all_probs})
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    df_pred.to_csv(pred_csv, index=False)

    save_confusion_matrix(all_true, all_preds, cm_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return metrics

if __name__ == "__main__":
    run_cnn_baseline()
