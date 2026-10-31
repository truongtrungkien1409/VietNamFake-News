# Đồ Án Chuyên Ngành: Nhận Diện Tin Giả Tiếng Việt Trên Mạng Xã Hội Sử Dụng PhoBERT và SVM

Hệ thống nhận diện tin giả tiếng Việt dựa trên mô hình học sâu kết hợp (PhoBERT - Support Vector Machine) tuân thủ tuyệt đối cấu trúc và phương pháp luận trong báo cáo đồ án chính thức (`DACN.pdf`).

---

## 🏗️ 1. Kiến Trúc Hệ Thống (Pipeline)

1. **Tiền xử lý văn bản (5 bước chuẩn):** Chuẩn hóa Unicode NFC/HTML, chuẩn hóa URL thành `HTTPURL` và mentions thành `@USER`, xử lý emoji/teencode, tách từ ghép tiếng Việt bằng `pyvi` (mô phỏng RDRSegmenter), và loại bỏ bản ghi rỗng/trùng lặp.
2. **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mô hình tiền huấn luyện **PhoBERT-base-v2** (đóng băng trọng số) với `max_length=256` để trích xuất véc-tơ trạng thái ẩn 768 chiều từ token đặc biệt `[CLS]`.
3. **Chuẩn hóa & Phân loại (StandardScaler + SVM):** Chuẩn hóa véc-tơ bằng `StandardScaler` (Z-score scaling) và đưa vào bộ phân loại **Support Vector Machine (SVM)** với `kernel='linear'`, `C=1.0`, `probability=True`.
4. **Triển khai ứng dụng:** Flask REST API (`app/app.py`) kết hợp Tiện ích mở rộng trình duyệt Chrome Extension (`app/extension/`).

---

## 📁 2. Cấu Trúc Thư Mục Dự Án

```
F:\DACN\Vietnamese-Fake-News-detection-master\
├── data/
│   ├── raw/                  # Dữ liệu thô ban đầu (.csv)
│   ├── processed/            # Dữ liệu sau tiền xử lý
│   └── splits/               # Phân chia train.csv, val.csv, test.csv + .npy
├── src/
│   ├── preprocessing/        # Tiền xử lý 5 bước & script chia tập dữ liệu
│   ├── features/             # Trích xuất đặc trưng [CLS] 768 chiều từ PhoBERT
│   ├── models/               # Huấn luyện SVM & các mô hình baseline (Naive Bayes...)
│   └── evaluation/           # Đánh giá hiệu năng (Accuracy, Precision, Recall, F1, CM)
├── experiments/              # Script chạy các kịch bản thực nghiệm
├── results/                  # Bảng số liệu, biểu đồ và trọng số đã lưu (svm_model.pkl, scaler.pkl)
├── app/                      # Flask REST API và Chrome Extension
├── extras/                   # Các module mở rộng cũ (Ensemble, SerpAPI)
├── DACN.pdf                  # Báo cáo đồ án chính thức (Nguồn sự thật)
└── README.md                 # Tài liệu hướng dẫn dự án
```

---

## 🚀 3. Hướng Dẫn Tái Lập Thí Nghiệm & Sử Dụng

### Bước 3.1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3.2: Phân chia tập dữ liệu (80% Train, 10% Validation, 10% Test)
```bash
python src/preprocessing/split_data.py data/raw/vn_news_merged.csv
```

### Bước 3.3: Trích xuất đặc trưng PhoBERT [CLS] (768 chiều)
```bash
python src/features/extract_phobert.py
```

### Bước 3.4: Huấn luyện mô hình SVM
```bash
python src/models/train_svm.py
```

### Bước 3.5: Đánh giá mô hình trên tập Test
```bash
python src/evaluation/evaluate.py
```

### Bước 3.6: Khởi chạy Flask REST API
```bash
python app/app.py
```

---

## 📊 4. Kết Quả Thực Nghiệm & Đánh Giá
Tham khảo chi tiết các bảng số liệu, biểu đồ ma trận nhầm lẫn (Confusion Matrix), đường cong ROC/AUC được lưu tự động trong thư mục `results/tables/`.
