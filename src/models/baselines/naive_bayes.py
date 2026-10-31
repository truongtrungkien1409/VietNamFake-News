import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_naive_bayes_baseline(
    train_csv: str = "data/splits/train.csv",
    test_csv: str = "data/splits/test.csv",
    model_output: str = "results/saved_models/nb_model.pkl"
):
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print("⚠️ Thiếu file dữ liệu train/test cho Naive Bayes baseline.")
        return

    train_df = pd.read_csv(train_csv).dropna(subset=['text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['text', 'label'])

    print("📊 Đang trích xuất đặc trưng TF-IDF cho Naive Bayes...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['text'].astype(str))
    y_train = train_df['label'].astype(int)

    X_test = vectorizer.transform(test_df['text'].astype(str))
    y_test = test_df['label'].astype(int)

    print("⚙️ Đang huấn luyện mô hình Naive Bayes (MultinomialNB)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Naive Bayes Baseline Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Tin thật (0)", "Tin giả (1)"]))

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump((nb_model, vectorizer), model_output)
    print(f"💾 Đã lưu Naive Bayes baseline tại {model_output}.")

if __name__ == "__main__":
    train_naive_bayes_baseline()
