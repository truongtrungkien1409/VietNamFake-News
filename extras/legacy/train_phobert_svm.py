import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import joblib
from core.preprocess import clean_and_preprocess_text

# ==========================================
# CẤU HÌNH THAM SỐ THỰC NGHIỆM (THEO DACN.PDF)
# ==========================================
MODEL_NAME = "vinai/phobert-base-v2"
DATA_PATH = "vn_news_merged.csv"
MAX_LENGTH = 256                  # max_length = 256 tokens
BATCH_SIZE = 32                   # batch_size = 32
TEST_SIZE = 0.1                   # 10% Test (hoặc 0.2 tùy cấu hình, DACN.pdf dùng 80-10-10 hoặc 80-20)
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Cấu hình SVM
SVM_KERNEL = 'linear'             # Linear Kernel theo DACN.pdf mục 2.4.2.2
SVM_C = 1.0                       # C = 1.0 (Soft margin tối ưu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Sử dụng thiết bị: {device}")


def extract_cls_features(texts, tokenizer, model, batch_size=32):
    """
    Trích xuất đặc trưng 768 chiều từ [CLS] token của PhoBERT (theo mục 2.3.2.2 & 2.3.3.1 DACN.pdf).
    """
    model.eval()
    all_embeddings = []

    print(f"🔄 Đang trích xuất đặc trưng [CLS] cho {len(texts)} văn bản...")
    start_time = time.time()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize văn bản với padding và truncation max_length=256
        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded_inputs)
            # Lấy véc-tơ tại token đặc biệt [CLS] (chỉ số 0 trên chiều sequence_length)
            last_hidden_state = outputs.last_hidden_state
            cls_embeddings = last_hidden_state[:, 0, :]  # Kích thước: (batch_size, 768)
            all_embeddings.append(cls_embeddings.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0 or (i + batch_size >= len(texts)):
            print(f"   - Đã xử lý {min(i + batch_size, len(texts))}/{len(texts)} mẫu...")

    print(f"✅ Hoàn thành trích xuất đặc trưng [CLS] trong {time.time() - start_time:.2f} giây.")
    return np.concatenate(all_embeddings, axis=0)


def main():
    # ==========================================
    # BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU (5 BƯỚC)
    # ==========================================
    if not os.path.exists(DATA_PATH):
        print(f"❌ Không tìm thấy file dữ liệu tại: {DATA_PATH}. Vui lòng đặt file vn_news_merged.csv vào thư mục gốc.")
        return

    print(f"📂 Đang đọc dữ liệu từ {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("File CSV phải chứa hai cột 'text' và 'label'.")
    
    df = df.dropna(subset=['text', 'label'])
    
    print("🧹 Đang thực hiện quy trình tiền xử lý 5 bước chuyên biệt cho văn bản mạng xã hội...")
    df['cleaned_text'] = df['text'].apply(clean_and_preprocess_text)
    
    texts = df['cleaned_text'].astype(str).tolist()
    labels = df['label'].astype(int).values

    print(f"📊 Tổng số mẫu dữ liệu: {len(df)}")
    print(f"   - Tin thật (Label 0): {np.sum(labels == 0)}")
    print(f"   - Tin giả (Label 1): {np.sum(labels == 1)}")

    # ==========================================
    # BƯỚC 2: TẢI MODEL PHOBERT-BASE-V2 VÀ TOKENIZER
    # ==========================================
    print(f"🤖 Đang tải mô hình {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    except Exception as e:
        print(f"⚠️ Không tải được {MODEL_NAME}, chuyển sang 'vinai/phobert-base'... Lỗi: {e}")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        bert_model = AutoModel.from_pretrained("vinai/phobert-base").to(device)

    # ==========================================
    # BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG 768 CHIỀU & LƯU .NPY
    # ==========================================
    features = extract_cls_features(texts, tokenizer, bert_model, batch_size=BATCH_SIZE)
    
    print(f"📐 Kích thước ma trận đặc trưng: {features.shape}")
    assert features.shape[1] == 768, "Lỗi: Đặc trưng trích xuất không phải 768 chiều!"

    # Lưu trữ ra file .npy (theo DACN.pdf mục 2.3.3)
    np.save("phobert_features_768.npy", features)
    np.save("phobert_labels.npy", labels)
    print("💾 Đã lưu thành công 'phobert_features_768.npy' và 'phobert_labels.npy'.")

    # ==========================================
    # BƯỚC 4: PHÂN CHIA TẬP TRAIN / TEST (Stratified Split)
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"🔀 Kích thước tập huấn luyện (Train): {X_train.shape[0]} mẫu")
    print(f"🔀 Kích thước tập kiểm thử (Test): {X_test.shape[0]} mẫu")

    # ==========================================
    # BƯỚC 5: CHUẨN HÓA DỮ LIỆU BẰNG STANDARDSCALER (Mục 2.4.1.2)
    # ==========================================
    print("⚖️ Đang chuẩn hóa tập véc-tơ đặc trưng bằng StandardScaler (Z-score scaling)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # BƯỚC 6: HUẤN LUYỆN MÔ HÌNH SVM (Linear Kernel, C=1.0, probability=True)
    # ==========================================
    print(f"⚙️ Đang huấn luyện mô hình SVM (kernel='{SVM_KERNEL}', C={SVM_C}, probability=True)...")
    svm_model = SVC(kernel=SVM_KERNEL, C=SVM_C, probability=True, random_state=RANDOM_STATE)
    
    start_train = time.time()
    svm_model.fit(X_train_scaled, y_train)
    print(f"✅ Huấn luyện SVM hoàn tất trong {time.time() - start_train:.2f} giây.")

    # ==========================================
    # BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH & XUẤT SỐ LIỆU (DACN.pdf Chương 3)
    # ==========================================
    print("\n" + "="*50)
    print("📈 KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM TRÊN TẬP TEST")
    print("="*50)
    
    y_pred = svm_model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"• Accuracy  : {acc * 100:.2f}%")
    print(f"• Precision : {precision * 100:.2f}%")
    print(f"• Recall    : {recall * 100:.2f}%")
    print(f"• F1-Score  : {f1 * 100:.2f}%")
    
    print("\n📋 Báo cáo phân loại chi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["Tin thật (0)", "Tin giả (1)"]))
    
    print("🧮 Ma trận nhầm lẫn (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Lưu mô hình SVM và StandardScaler đã huấn luyện (Mục 2.4.3.3)
    model_dir = os.path.join("ML core", "ML", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    svm_path = os.path.join(model_dir, "svm_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(svm_model, svm_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n💾 Đã lưu mô hình SVM vào '{svm_path}' và StandardScaler vào '{scaler_path}'.")


if __name__ == "__main__":
    main()
