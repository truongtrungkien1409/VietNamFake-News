# Đồ Án Chuyên Ngành: Nhận Diện Tin Giả Tiếng Việt Sử Dụng PhoBERT và SVM

Mã nguồn và tài liệu thực nghiệm cho đề tài đồ án chuyên ngành: **"Tiền xử lý văn bản và PhoBERT-SVM cho nhận diện tin giả tiếng Việt"** (Dựa theo nguồn tài liệu chính thức `DACN.pdf`).

---

## 1. Giới Thiệu & Kiến Trúc Hệ Thống

Hệ thống cung cấp một pipeline end-to-end hoàn chỉnh nhằm phát hiện tin giả trên văn bản tiếng Việt bằng cách kết hợp:
1. **Tiền xử lý văn bản 5 bước chuẩn hóa tiếng Việt** (Chuẩn hóa Unicode, làm sạch URL/HTML, xử lý hashtag & dấu câu lặp, xử lý teencode/emoji, và phân tách từ ghép bằng `pyvi`).
2. **Trích xuất đặc trưng ngữ nghĩa sâu** bằng mô hình ngôn ngữ lớn **PhoBERT-base-v2** (đóng băng trọng số, lấy véc-tơ `[CLS]` 768 chiều).
3. **Phân loại nhị phân** bằng mô hình **Support Vector Machine (SVM)** với chuẩn hóa `StandardScaler` (Nhãn: `0` = Tin thật, `1` = Tin giả).

---

## 2. Bảng Thông Số Kỹ Thuật (Khớp Bảng 2.3 & 2.8 trong DACN.pdf)

### Bảng 2.3: Thông số kiến trúc PhoBERT-base-v2
| Thành phần | Thông số cấu hình |
| :--- | :--- |
| **Model Architecture** | Transformer Encoder (RoBERTa architecture) |
| **Pre-trained Weights** | `vinai/phobert-base-v2` |
| **Số lớp Transformer (Layers)** | 12 |
| **Kích thước ẩn (Hidden Size)** | 768 |
| **Số đầu Attention (Attention Heads)** | 12 |
| **Độ dài chuỗi tối đa (Max Length)** | 256 tokens |
| **Kích thước Batch (Batch Size)** | 32 |

### Bảng 2.8: Thông số siêu tham số mô hình SVM
| Siêu tham số | Giá trị chuẩn |
| :--- | :--- |
| **Algorithm** | Support Vector Classifier (SVC) |
| **Kernel Function** | Linear (`kernel='linear'`) |
| **Hằng số phạt lỗi ($C$)** | 1.0 |
| **Gamma** | `scale` |
| **Probability Estimation** | `True` (hỗ trợ tính xác suất dự báo) |
| **Random Seed** | 42 (Cố định toàn hệ thống) |

---

## 3. Cấu Trúc Thư Mục Thực Tế Của Repo

```text
F:\DACN\Vietnamese-Fake-News-detection-master\
├── .env.example
├── .gitignore
├── DACN.pdf
├── Makefile
├── README.md
├── requirements.txt
├── app/
│   ├── app.py
│   └── extension/
│       ├── background.js
│       ├── manifest.json
│       ├── popup.html
│       └── popup.js
├── data/
│   ├── raw/
│   └── splits/
├── docs/
│   ├── ablation.md
│   └── dataset.md
├── experiments/
│   ├── ablation_preprocessing.py
│   ├── make_results_index.py
│   └── run_all.py
├── extras/
│   ├── api.py
│   ├── app_logic.py
│   ├── main.py
│   ├── webapp.py
│   └── legacy/
├── ML core/
│   └── (Jupyter notebooks & extracted models)
├── notebook/
│   └── fake-checker.ipynb
├── scripts/
│   └── run.ps1
├── src/
│   ├── evaluation/
│   │   ├── compare_models.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── features/
│   │   └── extract_phobert.py
│   ├── models/
│   │   ├── train_svm.py
│   │   └── baselines/
│   │       ├── cnn.py
│   │       ├── lstm.py
│   │       ├── naive_bayes.py
│   │       └── svm_tfidf.py
│   └── preprocessing/
│       ├── preprocess.py
│       ├── split_data.py
│       └── teencode.py
└── tests/
    ├── test_baselines.py
    ├── test_metrics.py
    ├── test_preprocessing_table21.py
    └── test_train_svm.py
```

---

## 4. Chuẩn Bị Dữ Liệu (`docs/dataset.md`)
Dữ liệu thô đặt tại `data/raw/vn_news_merged.csv` bao gồm các cột `text` (hoặc `original_text`) và `label` (`0` = Tin thật, `1` = Tin giả).
Hệ thống sử dụng chiến lược **Stratified Split (80% Train, 10% Validation, 10% Test)**.
Chi tiết hướng dẫn chuẩn bị xem tại [docs/dataset.md](docs/dataset.md).

---

## 5. Hướng Dẫn Chạy & Lệnh Thực Thi

Bạn có thể sử dụng **Makefile** (trên Linux/macOS/WSL) hoặc **PowerShell Script** (`scripts/run.ps1` trên Windows):

### Cài đặt thư viện:
```bash
make install
# Hoặc trên PowerShell:
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command install
```

### Chạy kiểm thử (Unit Tests):
```bash
make test
# Hoặc: python -m pytest tests -q
```

### Chạy toàn bộ quy trình thực nghiệm (Full Mode):
```bash
make all
# Hoặc: python -m experiments.run_all
```

### Chạy nhanh kiểm tra quy trình (Quick Mode):
```bash
make quick
# Hoặc: python -m experiments.run_all --quick
```
*(Chế độ quick sử dụng mẫu nhỏ và ghi toàn bộ kết quả vào thư mục `results_quick/` thay vì ghi đè kết quả thật).*

### Chạy từng bước riêng lẻ:
1. Phân chia dữ liệu: `make data`
2. Trích xuất đặc trưng PhoBERT: `make features`
3. Huấn luyện SVM: `make train`
4. Đánh giá mô hình: `make evaluate`
5. So sánh Baseline: `make baselines`
6. Ablation Study: `make ablation`

---

## 6. Danh Sách Kết Quả Thực Nghiệm (`results/`)
Toàn bộ các bảng số liệu, biểu đồ và phân tích định tính được lưu trữ tập trung tại thư mục `results/` (chi tiết xem tại [results/RESULTS_INDEX.md](results/RESULTS_INDEX.md)):
- `results/tables/evaluation_metrics.csv`
- `results/figures/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
- `results/tables/predictions_test.csv` & `error_analysis.csv`
- `results/error.md` (Top FP & FN)
- `results/tables/model_comparison.csv` (So sánh với Naive Bayes, TextLSTM, TextCNN)
- `results/tables/ablation_preprocessing.csv` & biểu đồ `ablation_preprocessing.png`

---

## 7. Triển Khai: Flask REST API & Chrome Extension

### Khởi chạy Flask API (`app/app.py`):
```bash
make serve
# Hoặc: python -m app.app
```
API khởi chạy tại `http://127.0.0.1:5000` phục vụ endpoint nhận diện tin giả qua JSON payload.

### Cài đặt Chrome Extension (`app/extension/`):
1. Mở trình duyệt Chrome, truy cập `chrome://extensions/`.
2. Bật **Developer mode** (Chế độ nhà phát triển) ở góc trên bên phải.
3. Nhấn **Load unpacked** (Tải tiện ích đã giải nén) và chọn thư mục `app/extension/`.

---

## 8. Hạn Chế Đã Biết & Khác Biệt So Với Báo Cáo

### Hạn chế đã biết:
1. **Thư viện tách từ:** Sử dụng `pyvi` (ViTokenizer) thay vì `RDRSegmenter` do sự khác biệt về môi trường cài đặt py_vncorenlp trên một số hệ điều hành.
2. **Nhãn dữ liệu (ReINTEL / Mạng xã hội):** Dữ liệu có thể chứa độ nhiễu ngữ cảnh từ các bài viết ngắn trên mạng xã hội.
3. **Lệch miền dữ liệu (Domain Shift):** Mô hình huấn luyện trên tập tin tức tổng hợp có thể suy giảm độ chính xác khi áp dụng vào các bài viết lóng chuyên biệt hoặc tin tức dạng đa phương thức (hình ảnh/video).
4. **Không đa phương thức:** Hệ thống hiện chỉ xử lý văn bản thuần túy (Text-only).

### Khác biệt so với báo cáo:
- Mọi kết quả, biểu đồ và kiểm định thống kê (McNemar, Bootstrap CI 95%) được tái tạo hoàn toàn tự động qua các script trong thư mục `experiments/` và lưu trữ tại `results/`, khớp 100% với cấu trúc nội dung của `DACN.pdf`.
