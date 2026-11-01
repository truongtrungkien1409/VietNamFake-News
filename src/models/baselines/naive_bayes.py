import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.evaluation.metrics import compute_metrics, save_confusion_matrix
from src.config import get_paths

def run_naive_bayes_baseline(
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    output_csv: str = None,
    metrics_output: str = None,
    pred_csv: str = None,
    cm_path: str = None,
    max_samples: int = None,
    model_output: str = None,
    **kwargs
):
    paths = get_paths()
    if train_csv is None: train_csv = os.path.join(paths.splits, "train.csv")
    if test_csv is None: test_csv = os.path.join(paths.splits, "test.csv")
    if model_output is None: model_output = os.path.join(paths.saved_models, "nb_model.pkl")
    if metrics_output is None: metrics_output = output_csv or os.path.join(paths.tables, "nb_metrics.csv")
    if pred_csv is None: pred_csv = os.path.join(paths.tables, "nb_preds.csv")
    if cm_path is None: cm_path = os.path.join(paths.figures, "cm_nb.png")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("⚠️ Thiếu file dữ liệu train/test cho Naive Bayes baseline.")
        return {}

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples)

    print("📊 [Naive Bayes] Đang trích xuất đặc trưng TF-IDF (ngram 1-2, max_features=10000)...")
    start_t = time.time()
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    y_train = train_df['label'].astype(int)

    X_test = vectorizer.transform(test_df['text'].astype(str))
    y_test = test_df['label'].astype(int)
    train_time = time.time() - start_t

    print("⚙️ [Naive Bayes] Đang huấn luyện mô hình MultinomialNB...")
    start_fit = time.time()
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_pred = nb_model.predict(X_test)
    y_probs = nb_model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred
    inference_time_per_sample = pred_time / X_test.shape[0] if X_test.shape[0] > 0 else 0.0

    metrics = compute_metrics(y_test, y_pred, y_probs)
    metrics["training_time_sec"] = round(train_time + fit_time, 2)
    metrics["inference_sec_per_sample"] = round(inference_time_per_sample, 6)

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output, index=False)
    
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

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump((nb_model, vectorizer), model_output)
    print(f"✅ Naive Bayes Baseline hoàn tất! Accuracy: {metrics['accuracy']*100:.2f}% | F1 (Fake): {metrics['fake_f1']*100:.2f}%")
    return metrics

train_and_evaluate_nb = run_naive_bayes_baseline

if __name__ == "__main__":
    run_naive_bayes_baseline()
