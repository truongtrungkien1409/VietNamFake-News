# Chương 3 - Thực Nghiệm

Tài liệu này là khung triển khai Chương 3, gồm: môi trường thực nghiệm, quy trình chạy, vị trí kết quả và cách đối chiếu số liệu.

## 3.1. Môi trường thực nghiệm

### Cài đặt
```bash
python -m pip install -r requirements.txt
```

### Chạy toàn bộ quy trình thực nghiệm
```bash
python -m experiments.run_all
```

Hoặc:
```bash
make all
```

Nếu muốn chạy nhanh để kiểm tra luồng:
```bash
python -m experiments.run_all --quick
```

Kết quả quick mode sẽ vào `results_quick/` và dữ liệu tách nhanh vào `data/splits_quick/`.

## 3.2. Kết quả mô hình đề xuất (PhoBERT + SVM)

### Chạy từng bước
```bash
python -m src.preprocessing.split_data data/raw/vn_news_merged.csv
python -m src.features.extract_phobert
python -m src.models.train_svm
python -m src.evaluation.evaluate
```

### File đầu ra chính
- `results/tables/evaluation_metrics.csv`
- `results/figures/confusion_matrix.png`
- `results/figures/roc_curve.png`
- `results/figures/pr_curve.png`
- `results/tables/svm_tuning.csv`
- `results/figures/svm_tuning_heatmap.png`

## 3.3. Phân tích lỗi

Tạo từ bước evaluate:
- `results/tables/predictions_test.csv`
- `results/tables/error_analysis.csv`
- `results/error_samples.md`
- `results/tables/threshold_analysis.csv`

Gợi ý trình bày:
- Nêu các mẫu FP/FN điển hình.
- Phân tích nguyên nhân nhầm lẫn theo ngữ cảnh, từ vựng, độ dài câu, nội dung mơ hồ.

## 3.4. So sánh với baseline

### Chạy baseline
```bash
python -m src.evaluation.compare_models
```

### File đầu ra
- `results/tables/model_comparison.csv`
- `results/figures/model_comparison_bar.png`
- `results/tables/mcnemar.csv`

## 3.5. Ablation Study

### Chạy ablation
```bash
python -m experiments.ablation_preprocessing
```

### File đầu ra
- `results/tables/ablation_preprocessing.csv`
- `results/figures/ablation_preprocessing.png`
- `results/tables/mcnemar.csv`

## 3.6. Khả năng triển khai hệ thống

### Chạy API
```bash
python -m app.app
```

### Benchmark API (độ trễ)
```bash
python -m scripts.benchmark_api
```

### File đầu ra liên quan
- `results/tables/inference_time.csv`
- `results/tables/api_latency.csv`
- `results/environment.json`

## 3.7. Giải thích các file CSV trong `results/tables/`

Thư mục `results/tables/` chứa các bảng số liệu do pipeline sinh ra. Có thể phân loại theo mục trong Chương 3 như sau:

### Nhóm dùng cho mục 3.1 - Môi trường và quy trình thực nghiệm

| File | Ý nghĩa |
| :--- | :--- |
| `run_timing.csv` | Thời gian thực thi của từng bước pipeline: chia dữ liệu, trích xuất PhoBERT, huấn luyện, đánh giá, baseline và ablation. File này được tạo khi chạy `python -m experiments.run_all`. |

`environment.json` không phải CSV nhưng cũng thuộc nhóm này: lưu hệ điều hành, phiên bản Python/thư viện, CPU, RAM và GPU.

### Nhóm dùng cho mục 3.2 - Kết quả mô hình đề xuất PhoBERT-SVM

| File | Ý nghĩa |
| :--- | :--- |
| `evaluation_metrics.csv` | Bảng chỉ số tổng hợp trên tập test của mô hình đề xuất: Accuracy, Precision, Recall, F1 cho lớp tin thật/tin giả, Macro/Weighted average, AUC-ROC, Average Precision, TN/FP/FN/TP và khoảng tin cậy F1. |
| `svm_tuning.csv` | Kết quả khảo sát siêu tham số SVM theo kernel, giá trị `C`, độ chính xác, F1 và thời gian huấn luyện. Dùng để giải thích cách chọn cấu hình SVM cuối cùng. |
| `threshold_analysis.csv` | Kết quả thay đổi ngưỡng phân loại, cho biết Accuracy, Fake Precision, Fake Recall và Fake F1 ở từng ngưỡng. Có thể đặt ở cuối mục 3.2 hoặc dẫn sang mục 3.3 khi phân tích lỗi/ngưỡng. |

Các cột quan trọng trong `evaluation_metrics.csv`:

- `real_*`: chỉ số của lớp 0 - tin thật.
- `fake_*`: chỉ số của lớp 1 - tin giả.
- `*_support`: số mẫu thật sự thuộc mỗi lớp trong tập test.
- `TN`, `FP`, `FN`, `TP`: bốn thành phần ma trận nhầm lẫn.
- `fake_f1_ci_lower`, `fake_f1_ci_upper`: cận dưới/cận trên khoảng tin cậy Bootstrap 95%.

### Nhóm dùng cho mục 3.3 - Phân tích lỗi

| File | Ý nghĩa |
| :--- | :--- |
| `predictions_test.csv` | Dự đoán của PhoBERT-SVM cho từng mẫu test, gồm văn bản, nhãn thật, nhãn dự đoán và xác suất tin giả. |
| `error_analysis.csv` | Các mẫu bị dự đoán sai, thường dùng để tách False Positive (FP) và False Negative (FN). |
| `threshold_analysis.csv` | Hỗ trợ phân tích đánh đổi Precision/Recall/F1 khi thay đổi ngưỡng quyết định. |
| `nb_preds.csv` | Dự đoán từng mẫu của baseline Naive Bayes, dùng khi đối chiếu lỗi với mô hình đề xuất. |
| `predictions_svm_tfidf.csv` | Dự đoán từng mẫu của baseline SVM dùng TF-IDF. |
| `lstm_preds.csv` | Dự đoán từng mẫu của baseline TextLSTM. |
| `cnn_preds.csv` | Dự đoán từng mẫu của baseline TextCNN. |

Các file `*_preds.csv` có cùng nhóm cột chính: `original_text`, `text`, `true_label`, `predicted_label`, `fake_probability`.

### Nhóm dùng cho mục 3.4 - So sánh với baseline

| File | Ý nghĩa |
| :--- | :--- |
| `model_comparison.csv` | Bảng tổng hợp so sánh PhoBERT-SVM với Naive Bayes (TF-IDF), SVM (TF-IDF), TextLSTM và TextCNN theo Accuracy, Fake Precision, Fake Recall, Fake F1, AUC-ROC, thời gian train và thời gian suy luận. |
| `nb_metrics.csv` | Chỉ số tổng hợp của Naive Bayes. |
| `baseline_svm_tfidf.csv` | Chỉ số tổng hợp của SVM dùng TF-IDF. |
| `lstm_metrics.csv` | Chỉ số tổng hợp của TextLSTM. |
| `cnn_metrics.csv` | Chỉ số tổng hợp của TextCNN. |

Các file `*_metrics.csv` là bảng chi tiết của từng baseline; khi viết luận văn, dùng `model_comparison.csv` làm bảng chính và dùng các file chi tiết để kiểm tra hoặc phụ lục.

### Nhóm dùng cho mục 3.5 - Ablation Study

| File | Ý nghĩa |
| :--- | :--- |
| `ablation_preprocessing.csv` | Kết quả các biến thể A-F, từ không tiền xử lý đến pipeline đầy đủ; gồm Accuracy, Fake Precision/Recall/F1, AUC-ROC, Macro F1 và thời gian trích xuất. |
| `mcnemar.csv` | Kết quả kiểm định McNemar giữa các biến thể/mô hình, gồm thống kê kiểm định, p-value và kết luận có ý nghĩa ở mức 0.05 hay không. |

### Nhóm dùng cho mục 3.6 - Triển khai và hiệu năng

| File | Ý nghĩa |
| :--- | :--- |
| `inference_time.csv` | Tách thời gian suy luận thành PhoBERT, SVM và tổng thời gian trên mỗi mẫu. |
| `api_latency.csv` | Độ trễ API theo tổng số request, trung bình, trung vị và P95. File chỉ xuất hiện sau khi chạy benchmark API. |

`api_latency.csv` hiện không nằm trong danh sách CSV được commit; nếu benchmark tạo file này thì nên kiểm tra dữ liệu đầu vào trước khi dùng trong báo cáo.

## 3.8. Lưu ý khi đọc các kết quả hiện tại

Tại thời điểm viết tài liệu, `evaluation_metrics.csv` đang có `real_support=6` và `fake_support=6`, tức tổng cộng 12 mẫu test. Đây phù hợp để kiểm tra pipeline, nhưng chưa đủ để đại diện cho thực nghiệm chính thức.

Để tạo số liệu dùng trong luận văn:

```bash
python -m experiments.run_all
```

Sau đó kiểm tra nhanh kích thước tập test và support:

```bash
python -c "import pandas as pd; df=pd.read_csv('results/tables/evaluation_metrics.csv'); print(df[['real_support','fake_support']].to_string(index=False))"
```

Không dùng các file trong `results_quick/` để kết luận chính thức. Các file CSV trong `results/tables/` cũng cần được kiểm tra nguồn dữ liệu, đặc biệt khi support nhỏ hoặc được tạo từ dữ liệu mô phỏng.

## Phụ lục lệnh nhanh (Windows PowerShell script)

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command install
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command test
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command all
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command quick
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command serve
```

## Checklist hoàn thiện Chương 3

1. Có bảng chỉ số chính từ `evaluation_metrics.csv`.
2. Có đủ 3 biểu đồ: Confusion Matrix, ROC, PR.
3. Có mục phân tích lỗi (FP/FN) dựa trên file dự đoán.
4. Có so sánh baseline và nhận xét.
5. Có ablation và kết luận về ảnh hưởng tiền xử lý.
6. Có thông tin thời gian chạy và môi trường.

## Gợi ý trích dẫn kết quả

- Khi ghi số liệu vào luận văn, chỉ dùng kết quả từ thư mục `results/` (full mode).
- Không dùng số liệu từ `results_quick/` để kết luận chính thức.
