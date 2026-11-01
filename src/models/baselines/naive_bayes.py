import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.evaluation.metrics import compute_metrics, save_confusion_matrix

def run_naive_bayes_baseline(
    train_csv: str = "data/splits/train.csv",
    test_csv: str = "data/splits/test.csv",
    model_output: str = "results/saved_models/nb_model.pkl",
    metrics_output: str = "results/tables/nb_metrics.csv"
):
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("⚠️ Thiếu file dữ liệu train/test cho Naive Bayes baseline.")
        return

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

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
    inference_time_per_sample = pred_time / X_test.shape[0]

    metrics = compute_metrics(y_test, y_pred, y_probs)
    metrics["training_time_sec"] = round(train_time + fit_time, 2)
    metrics["inference_sec_per_sample"] = round(inference_time_per_sample, 4)

    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_output, index=False)
    
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump((nb_model, vectorizer), model_output)
    print(f"✅ Naive Bayes Baseline hoàn tất! Accuracy: {metrics['accuracy']*100:.2f}% | F1 (Fake): {metrics['fake_f1']*100:.2f}%")
    return metrics

def train_and_evaluate_nb(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    output_csv: str = "results/tables/nb_metrics.csv",
    pred_csv: str = "results/tables/nb_preds.csv",
    cm_path: str = "results/figures/cm_nb.png",
    max_samples: int = None
):
    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])
    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    y_train = train_df['label'].astype(int)

    X_test = vectorizer.transform(test_df['text'].astype(str))
    y_test = test_df['label'].astype(int)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    y_probs = nb.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_probs)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)

    df_pred = pd.DataFrame({"true": y_test, "pred": y_pred, "prob": y_probs})
    os.makedirs(os.path.dirname(pred_csv), exist_ok=True)
    df_pred.to_csv(pred_csv, index=False)

    save_confusion_matrix(y_test, y_pred, cm_path)
    return metrics

def train_and_evaluate_svm_tfidf(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    output_csv: str = "results/tables/svm_tfidf_metrics.csv",
    pred_csv: str = "results/tables/svm_tfidf_preds.csv",
    cm_path: str = "results/figures/cm_svm_tfidf.png",
    max_samples: int = None
):
    from sklearn.svm import LinearSVC
    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])
    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    y_train = train_df['label'].astype(int)

    X_test = vectorizer.transform(test_df['text'].astype(str))
    y_test = test_df['label'].astype(int)

    clf = LinearSVC(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_probs = 1.0 / (1.0 + np.exp(-clf.decision_function(X_test)))

    metrics = compute_metrics(y_test, y_pred, y_probs)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_csv, index=False)
    save_confusion_matrix(y_test, y_pred, cm_path)
    return metrics

if __name__ == "__main__":
    run_naive_bayes_baseline()
