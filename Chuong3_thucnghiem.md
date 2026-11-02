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

