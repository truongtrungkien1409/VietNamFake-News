# Chỉ Mục Kết Quả Thực Nghiệm (Results Index - RESULTS)

Tài liệu này tự động liệt kê toàn bộ các bảng dữ liệu và biểu đồ được sinh ra trong thư mục `results/` phục vụ cho Chương 3 của đồ án.

| Đường dẫn tệp | Mô tả ngắn gọn | Mục Chương 3 | Trạng thái |
| :--- | :--- | :--- | :--- |
| `results\tables/evaluation_metrics.csv` | Bảng chỉ số đánh giá tổng hợp trên tập Test (Accuracy, P/R/F1, AUC, CI 95%) | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\figures/confusion_matrix.png` | Biểu đồ Ma trận nhầm lẫn (Confusion Matrix) trên tập Test | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\figures/roc_curve.png` | Đường cong ROC-AUC | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\figures/pr_curve.png` | Đường cong Precision-Recall | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\tables/predictions_test.csv` | Chi tiết dự đoán trên toàn bộ tập Test (True label, Pred, Probability) | 3.3 Phân tích lỗi | ✅ Đã tạo |
| `results\tables/error_analysis.csv` | Danh sách chi tiết các mẫu dự đoán sai (FP, FN) | 3.3 Phân tích lỗi | ✅ Đã tạo |
| `results\error_samples.md` | 10 mẫu FP và 10 mẫu FN xác suất cao nhất (cắt 300 ký tự đầu) | 3.3 Phân tích lỗi | ✅ Đã tạo |
| `results\tables/threshold_analysis.csv` | Phân tích Precision/Recall/F1 theo các ngưỡng quyết định (0.3 - 0.7) | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\tables/inference_time.csv` | Thời gian suy luận trung bình mỗi mẫu và kích thước trọng số | 3.6 Triển khai hệ thống | ✅ Đã tạo |
| `results\environment.json` | Thông tin phần cứng, OS, phiên bản thư viện môi trường | 3.1 Môi trường thực nghiệm | ✅ Đã tạo |
| `results\tables/model_comparison.csv` | So sánh hiệu năng giữa PhoBERT-SVM và các baseline (NB, LSTM, CNN) | 3.4 So sánh với baseline | ✅ Đã tạo |
| `results\tables/ablation_preprocessing.csv` | Kết quả thí nghiệm Ablation Study kiểm chứng giả thuyết H2 | 3.5 Ablation Study | ✅ Đã tạo |
| `results\figures/ablation_preprocessing.png` | Biểu đồ F1-score các biến thể ablation kèm khoảng tin cậy | 3.5 Ablation Study | ✅ Đã tạo |
| `results\tables/mcnemar.csv` | Kết quả kiểm định thống kê McNemar so sánh các biến thể | 3.5 Ablation Study | ✅ Đã tạo |
| `results\tables/svm_tuning.csv` | Bảng khảo sát siêu tham số SVM (Linear vs RBF, C) | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\tables/svm_final_config.json` | Metadata cấu hình huấn luyện SVM cuối cùng | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\figures/svm_tuning_heatmap.png` | Biểu đồ nhiệt F1-score theo Kernel và C | 3.2 Kết quả thực nghiệm | ✅ Đã tạo |
| `results\tables/run_timing.csv` | Thống kê thời gian thực thi từng bước trong pipeline | 3.1 Môi trường thực nghiệm | ✅ Đã tạo |
| `results\tables/phobert_extraction_timing.csv` | Thời gian trích xuất embedding PhoBERT theo split | 3.1 Môi trường thực nghiệm | ⏳ Chưa chạy |
| `results\tables/preprocessing_examples.csv` | Ví dụ văn bản trước và sau tiền xử lý | 3.1 Môi trường thực nghiệm | ✅ Đã tạo |
| `results\tables/preprocessing_summary.csv` | Tóm tắt định lượng các bước lọc và chuẩn hóa | 3.1 Môi trường thực nghiệm | ✅ Đã tạo |