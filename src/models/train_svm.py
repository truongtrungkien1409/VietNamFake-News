import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_svm_model(
    train_feat_path: str = "data/splits/train_features_768.npy",
    train_label_path: str = "data/splits/train_labels.npy",
    model_output_path: str = "results/saved_models/svm_model.pkl",
    scaler_output_path: str = "results/saved_models/scaler.pkl",
    kernel: str = "linear",
    C: float = 1.0,
    random_state: int = 42
):
    if not os.path.exists(train_feat_path) or not os.path.exists(train_label_path):
        print(f"⚠️ Không tìm thấy file dữ liệu huấn luyện tại {train_feat_path}. Hãy chạy trích xuất đặc trưng trước.")
        return

    print("📂 Đang tải ma trận đặc trưng huấn luyện...")
    X_train = np.load(train_feat_path)
    y_train = np.load(train_label_path)

    print("⚖️ Đang chuẩn hóa dữ liệu bằng StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"⚙️ Đang huấn luyện SVM (kernel='{kernel}', C={C}, probability=True)...")
    svm_model = SVC(kernel=kernel, C=C, probability=True, random_state=random_state)
    svm_model.fit(X_train_scaled, y_train)

    # Đánh giá nhanh trên tập train
    y_pred = svm_model.predict(X_train_scaled)
    acc = accuracy_score(y_train, y_pred)
    print(f"✅ Huấn luyện hoàn tất! Accuracy trên train set: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(svm_model, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"💾 Đã lưu mô hình tại {model_output_path} và scaler tại {scaler_output_path}.")

if __name__ == "__main__":
    train_svm_model()
