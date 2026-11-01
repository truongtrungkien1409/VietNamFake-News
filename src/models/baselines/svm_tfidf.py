import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.config import get_paths

def train_and_evaluate_svm_tfidf(
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    output_csv: str = None,
    metrics_output: str = None,
    pred_csv: str = None,
    cm_path: str = None,
    max_samples: int = None,
    **kwargs
):
    paths = get_paths()
    if train_csv is None: train_csv = os.path.join(paths.splits, "train.csv")
    if val_csv is None: val_csv = os.path.join(paths.splits, "val.csv")
    if test_csv is None: test_csv = os.path.join(paths.splits, "test.csv")
    if metrics_output is None: metrics_output = output_csv or os.path.join(paths.tables, "baseline_svm_tfidf.csv")
    if pred_csv is None: pred_csv = os.path.join(paths.tables, "predictions_svm_tfidf.csv")
    if cm_path is None: cm_path = os.path.join(paths.figures, "svm_tfidf_confusion_matrix.png")

    print("📊 [SVM + TF-IDF Baseline] Đang tải dữ liệu...")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv) or not os.path.exists(test_csv):
        print("❌ Lỗi: Thiếu file dữ liệu chia tách. Hãy chạy split_data trước.")
        return {}

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    val_df = pd.read_csv(val_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max_samples)
        test_df = test_df.head(max_samples)

    X_train_text = train_df['text'].astype(str)
    y_train = train_df['label'].astype(int)

    X_val_text = val_df['text'].astype(str)
    y_val = val_df['label'].astype(int)

    X_test_text = test_df['text'].astype(str)
    y_test = test_df['label'].astype(int)

    print("📊 Đang trích xuất đặc trưng TF-IDF (max_features=10000, ngram=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    C_values = [0.1, 1.0, 10.0]
    best_C = 1.0
    best_val_f1 = -1.0

    print("🔬 Đang chọn C tối ưu trên tập Validation...")
    for C in C_values:
        svm = SVC(kernel='linear', C=C, probability=True, random_state=42)
        svm.fit(X_train, y_train)
        y_val_pred = svm.predict(X_val)
        val_metrics = compute_metrics(y_val, y_val_pred, svm.predict_proba(X_val)[:, 1])
        f1_fake = val_metrics['fake_f1']
        print(f"   - C={C} -> Validation F1 (Fake): {f1_fake*100:.2f}%")
        if f1_fake > best_val_f1:
            best_val_f1 = f1_fake
            best_C = C

    print(f"🏆 C tốt nhất: {best_C}")

    print("⚙️ Huấn luyện mô hình cuối cùng với C tốt nhất...")
    start_t = time.time()
    best_svm = SVC(kernel='linear', C=best_C, probability=True, random_state=42)
    best_svm.fit(X_train, y_train)
    train_time = time.time() - start_t

    start_inf = time.time()
    y_probs = best_svm.predict_proba(X_test)[:, 1]
    y_pred = best_svm.predict(X_test)
    inf_time = (time.time() - start_inf) / X_test.shape[0] if X_test.shape[0] > 0 else 0.0

    test_metrics = compute_metrics(y_test, y_pred, y_probs)
    test_metrics["training_time_sec"] = round(train_time, 2)
    test_metrics["inference_sec_per_sample"] = round(inf_time, 6)
    test_metrics["n_parameters"] = int(X_train.shape[1])

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    pd.DataFrame([test_metrics]).to_csv(metrics_output, index=False)

    df_preds = pd.DataFrame({
        "original_text": test_df['original_text'].tolist() if 'original_text' in test_df.columns else test_df['text'].tolist(),
        "text": test_df['text'].tolist(),
        "true_label": y_test.values,
        "predicted_label": y_pred,
        "fake_probability": y_probs
    })
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    df_preds.to_csv(pred_csv, index=False)
    save_confusion_matrix(y_test, y_pred, cm_path)

    print(f"✅ SVM + TF-IDF Baseline hoàn tất! Đã lưu kết quả tại {metrics_output}")
    return test_metrics

run_svm_tfidf_baseline = train_and_evaluate_svm_tfidf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train_and_evaluate_svm_tfidf(max_samples=args.max_samples)
