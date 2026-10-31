# Đồ Án Chuyên Ngành: Nhận Diện Tin Giả Tiếng Việt Trên Mạng Xã Hội Sử Dụng PhoBERT và SVM

Hệ thống nhận diện tin giả tiếng Việt dựa trên mô hình học sâu kết hợp (PhoBERT - Support Vector Machine) tuân thủ tuyệt đối cấu trúc và phương pháp luận trong báo cáo đồ án chính thức (`DACN.pdf`).

---

## 🏗️ 1. Kiến Trúc Hệ Thống (Pipeline)

1. **Tiền xử lý văn bản (5 bước chuẩn theo Mục 2.2.2.2 & Bảng 2.1):** Chuẩn hóa Unicode NFC/HTML/lowercase, chuẩn hóa URL thành `HTTPURL` và mentions thành `@USER` / xử lý hashtag, xử lý teencode & emoji thành từ diễn đạt, tách từ ghép tiếng Việt bằng **pyvi (ViTokenizer)** *(Lưu ý: Báo cáo mô tả RDRSegmenter/VnCoreNLP, nhưng thực nghiệm triển khai sử dụng pyvi)*, và loại bỏ bản ghi rỗng, trùng lặp, câu dưới 3 từ.
2. **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mô hình tiền huấn luyện **PhoBERT-base-v2** (đóng băng trọng số, `requires_grad=False`) với `max_length=256` để trích xuất véc-tơ trạng thái ẩn 768 chiều từ token đặc biệt `[CLS]`.
3. **Chuẩn hóa & Phân loại (StandardScaler + SVM):** Chuẩn hóa véc-tơ bằng `StandardScaler` (fit trên Train) và đưa vào bộ phân loại **Support Vector Machine (SVM)** với `kernel='linear'`, `C=1.0`, `probability=True`.
4. **Triển khai ứng dụng:** Flask REST API (`app/app.py`) kết hợp Tiện ích mở rộng trình duyệt Chrome Extension (`app/extension/`).

---

## 📁 2. Cấu Trúc Thư Mục Dự Án

```
F:\DACN\Vietnamese-Fake-News-detection-master\
├── data/
│   ├── raw/                  # Dữ liệu thô ban đầu (.csv)
│   ├── processed/            # Dữ liệu sau tiền xử lý (clean.csv)
│   └── splits/               # Phân chia train, val, test (.csv & .npy)
├── src/
│   ├── preprocessing/        # Tiền xử lý 5 bước (preprocess.py, teencode.py, split_data.py)
│   ├── features/             # Trích xuất đặc trưng [CLS] 768 chiều (extract_phobert.py)
│   ├── models/               # Huấn luyện SVM & baselines (train_svm.py, baselines/)
│   └── evaluation/           # Đánh giá hiệu năng và sinh biểu đồ (evaluate.py)
├── experiments/              # Script chạy pipeline tự động (run_all.py)
├── results/                  # Bảng số liệu, biểu đồ và trọng số (svm_model.pkl, scaler.pkl)
├── app/                      # Flask REST API và Chrome Extension (app.py, extension/)
├── extras/                   # Các module mở rộng cũ & legacy code (extras/legacy/)
├── DACN.pdf                  # Báo cáo đồ án chính thức (Nguồn sự thật)
└── README.md                 # Tài liệu hướng dẫn dự án
```

---

## 🚀 3. Hướng Dẫn Tái Lập Thí Nghiệm & Sử Dụng (Chạy từ thư mục gốc)

### Bước 3.1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3.2: Phân chia tập dữ liệu (80% Train, 10% Validation, 10% Test)
```bash
python -m src.preprocessing.split_data data/raw/vn_news_merged.csv
```

### Bước 3.3: Trích xuất đặc trưng PhoBERT [CLS] (768 chiều)
```bash
python -m src.features.extract_phobert
```

### Bước 3.4: Huấn luyện mô hình SVM
```bash
python -m src.models.train_svm
```

### Bước 3.5: Đánh giá mô hình trên tập Test
```bash
python -m src.evaluation.evaluate
```

### Bước 3.6: Khởi chạy Flask REST API
```bash
python -m app.app
```

### 💡 Chạy toàn bộ tự động:
```bash
python -m experiments.run_all
```
