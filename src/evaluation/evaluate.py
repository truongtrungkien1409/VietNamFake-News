import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def evaluate_model(
    test_feat_path: str = "data/splits/test_features_768.npy",
    test_label_path: str = "data/splits/test_labels.npy",
    model_path: str = "results/saved_models/svm_model.pkl",
    scaler_path: str = "results/saved_models/scaler.pkl",
    output_table_path: str = "results/tables/evaluation_metrics.csv"
):
    if not os.path.exists(test_feat_path) or not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("⚠️ Thiếu file đặc trưng test, mô hình SVM hoặc scaler. Hãy huấn luyện và trích xuất trước.")
        return

    print("📂 Đang tải dữ liệu kiểm thử và mô hình...")
    X_test = np.load(test_feat_path)
    y_test = np.load(test_label_path)

    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print("⚖️ Đang chuẩn hóa tập test bằng StandardScaler...")
    X_test_scaled = scaler.transform(X_test)

    print("🔮 Đang dự đoán trên tập kiểm thử...")
    y_pred = svm_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print("\n📈 KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM TRÊN TẬP TEST:")
    print(f"   - Accuracy  : {acc*100:.2f}%")
    print(f"   - Precision : {precision*100:.2f}%")
    print(f"   - Recall    : {recall*100:.2f}%")
    print(f"   - F1-score  : {f1*100:.2f}%")
    print("\n📋 Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Tin thật (0)", "Tin giả (1)"])
    print(report)
    print("\n🔢 Confusion Matrix:")
    print(cm)

    os.makedirs(os.path.dirname(output_table_path), exist_ok=True)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-score (Weighted)"],
        "Score": [acc, precision, recall, f1]
    })
    metrics_df.to_csv(output_table_path, index=False)
    print(f"💾 Đã lưu bảng kết quả đánh giá tại {output_table_path}.")

if __name__ == "__main__":
    evaluate_model()
