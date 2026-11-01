# Dữ Liệu Thực Nghiệm Chương 3: Kết Quả và Đánh Giá
*Tài liệu được tự động tổng hợp từ thư mục `results_quick/` lúc 2026-11-01T17:49:53.370565.*

## 3.1 Cấu Hình Môi Trường & Siêu Tham Số Thực Nghiệm
### Bảng 3.1.1: Thông tin phần cứng và phần mềm hệ thống
| Thông số | Giá trị |
|:---|:---|
| Hệ điều hành (OS) | `Windows 10` |
| Python Version | `3.8.0` |
| PyTorch Version | `2.4.1+cpu` |
| Transformers Version | `4.46.3` |
| Scikit-Learn Version | `1.3.2` |
| CPU | `AMD64 Family 23 Model 160 Stepping 0, AuthenticAMD (8 cores)` |
| RAM | `0.0 GB` |
| GPU | `None` |

### Bảng 3.1.2: Cấu hình trích xuất đặc trưng và mô hình SVM cuối cùng
| Cấu hình / Tham số | Giá trị |
|:---|:---|
| `kernel` | `linear` |
| `C` | `1.0` |
| `gamma` | `scale` |
| `class_weight` | `None` |
| `n_support` | `35` |
| `seed` | `42` |
| `training_time_sec` | `0.01` |
| `sklearn_version` | `1.3.2` |

## 3.2 Kết Quả Đánh Giá Mô Hình PhoBERT-SVM Trên Tập Test
### Bảng 3.2.1: Hiệu năng chi tiết theo lớp (Nhãn 0: Thật, Nhãn 1: Giả)
| Lớp (Class) | Precision (%) | Recall (%) | F1-Score (%) |
|:---|:---:|:---:|:---:|
| Tin thật (Real - 0) | `100.00` | `100.00` | `100.00` |
| **Tin giả (Fake - 1)** | **`100.00`** | **`100.00`** | **`100.00`** |

- **Accuracy toàn cục**: `100.00%`
- **AUC-ROC**: `1.0000`
- **Average Precision (AP)**: `1.0000`
- **Khoảng tin cậy 95% F1 (Tin giả - Bootstrap CI)**: `[100.00%, 100.00%]`

### Bảng 3.2.2: Ma trận nhầm lẫn (Confusion Matrix)
| Metric | Giá trị |
|:---|:---:|
| True Negatives (TN - Thật đoán Thật) | `6` |
| False Positives (FP - Thật đoán Giả) | `0` |
| False Negatives (FN - Giả đoán Thật) | `0` |
| True Positives (TP - Giả đoán Giả) | `6` |

### Hình minh họa đánh giá
- Biểu đồ Confusion Matrix: `results_quick/figures/confusion_matrix.png`
- Đường cong ROC-AUC: `results_quick/figures/roc_curve.png`
- Đường cong Precision-Recall: `results_quick/figures/pr_curve.png`

## 3.3 Khảo Sát Siêu Tham Số & Phân Tích Ngưỡng (Threshold Analysis)
### Bảng 3.3.1: Kết quả Grid Search SVM trên tập Validation (`svm_tuning.csv`)
| kernel   |    C | class_weight   |   train_accuracy |   accuracy |   precision_fake |   recall_fake |   f1_fake |   f1_macro |   auc |   n_support |   training_time_sec |
|:---------|-----:|:---------------|-----------------:|-----------:|-----------------:|--------------:|----------:|-----------:|------:|------------:|--------------------:|
| linear   |  0.1 | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0.02 |
| linear   |  0.1 | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0.01 |
| linear   |  1   | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0    |
| linear   |  1   | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0.01 |
| linear   | 10   | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0.01 |
| linear   | 10   | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          35 |                0.01 |
| rbf      |  0.1 | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          96 |                0.01 |
| rbf      |  0.1 | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          96 |                0.01 |
| rbf      |  1   | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          84 |                0.01 |
| rbf      |  1   | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          84 |                0.01 |
| rbf      | 10   | nan            |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          84 |                0.01 |
| rbf      | 10   | balanced       |                1 |          1 |                1 |             1 |         1 |          1 |     1 |          84 |                0.01 |

### Bảng 3.3.2: Phân tích hiệu năng theo các ngưỡng quyết định khác nhau (`threshold_analysis.csv`)
|   threshold |   precision_fake |   recall_fake |   f1_fake |
|------------:|-----------------:|--------------:|----------:|
|         0.3 |                1 |             1 |         1 |
|         0.4 |                1 |             1 |         1 |
|         0.5 |                1 |             1 |         1 |
|         0.6 |                1 |             1 |         1 |
|         0.7 |                1 |             1 |         1 |

## 3.4 Phân Tích Mẫu Lỗi (Error Analysis)
- **Tổng số mẫu False Positive (FP)**: `0`
- **Tổng số mẫu False Negative (FN)**: `0`
- **Chi tiết mẫu lỗi định tính**: Xem tại file `results_quick/error_samples.md`.

## 3.5 So Sánh Với Các Mô Hình Baseline & Kiểm Định Thống Kê
### Bảng 3.5.1: So sánh hiệu năng PhoBERT-SVM và các baseline (`model_comparison.csv`)
| Model                 |   Accuracy |   Fake_Precision |   Fake_Recall |   Fake_F1 |   AUC_ROC |   Training_Time_sec |   Inference_Sec_Per_Sample |
|:----------------------|-----------:|-----------------:|--------------:|----------:|----------:|--------------------:|---------------------------:|
| Naive Bayes (TF-IDF)  |        1   |              1   |             1 |  1        |         1 |                0.04 |                     0.0002 |
| TextLSTM              |        0.5 |              0   |             0 |  0        |         1 |                3.3  |                     0.0213 |
| TextCNN               |        0.5 |              0.5 |             1 |  0.666667 |         1 |                1.16 |                     0.0039 |
| PhoBERT-SVM (Đề xuất) |        1   |              1   |             1 |  1        |         1 |               45    |                     0.008  |

### Bảng 3.5.2: Kết quả kiểm định McNemar so sánh biến thể chuẩn với các biến thể khác (`mcnemar.csv`)
| comparison             |   mcnemar_statistic |   p_value | significant_at_0.05   |
|:-----------------------|--------------------:|----------:|:----------------------|
| Variant E vs Variant A |                   0 |         1 | False                 |
| Variant E vs Variant B |                   0 |         1 | False                 |
| Variant E vs Variant C |                   0 |         1 | False                 |
| Variant E vs Variant D |                   0 |         1 | False                 |
| Variant E vs Variant F |                   0 |         1 | False                 |

