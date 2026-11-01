import os

def generate_results_index(output_path: str = None):
    res_base = os.environ.get("RESULTS_BASE", "results")
    if output_path is None:
        output_path = os.path.join(res_base, "RESULTS_INDEX.md")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    expected_items = [
        (os.path.join(res_base, "tables/evaluation_metrics.csv"), "Bảng chỉ số đánh giá tổng hợp trên tập Test (Accuracy, P/R/F1, AUC, CI 95%)", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "figures/confusion_matrix.png"), "Biểu đồ Ma trận nhầm lẫn (Confusion Matrix) trên tập Test", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "figures/roc_curve.png"), "Đường cong ROC-AUC", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "figures/pr_curve.png"), "Đường cong Precision-Recall", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "tables/predictions_test.csv"), "Chi tiết dự đoán trên toàn bộ tập Test (True label, Pred, Probability)", "3.3 Phân tích lỗi"),
        (os.path.join(res_base, "tables/error_analysis.csv"), "Danh sách chi tiết các mẫu dự đoán sai (FP, FN)", "3.3 Phân tích lỗi"),
        (os.path.join(res_base, "error_samples.md"), "10 mẫu FP và 10 mẫu FN xác suất cao nhất (cắt 300 ký tự đầu)", "3.3 Phân tích lỗi"),
        (os.path.join(res_base, "tables/threshold_analysis.csv"), "Phân tích Precision/Recall/F1 theo các ngưỡng quyết định (0.3 - 0.7)", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "tables/inference_time.csv"), "Thời gian suy luận trung bình mỗi mẫu và kích thước trọng số", "3.6 Triển khai hệ thống"),
        (os.path.join(res_base, "environment.json"), "Thông tin phần cứng, OS, phiên bản thư viện môi trường", "3.1 Môi trường thực nghiệm"),
        (os.path.join(res_base, "tables/model_comparison.csv"), "So sánh hiệu năng giữa PhoBERT-SVM và các baseline (NB, LSTM, CNN)", "3.4 So sánh với baseline"),
        (os.path.join(res_base, "tables/ablation_preprocessing.csv"), "Kết quả thí nghiệm Ablation Study kiểm chứng giả thuyết H2", "3.5 Ablation Study"),
        (os.path.join(res_base, "figures/ablation_preprocessing.png"), "Biểu đồ F1-score các biến thể ablation kèm khoảng tin cậy", "3.5 Ablation Study"),
        (os.path.join(res_base, "tables/mcnemar.csv"), "Kết quả kiểm định thống kê McNemar so sánh các biến thể", "3.5 Ablation Study"),
        (os.path.join(res_base, "tables/svm_tuning.csv"), "Bảng khảo sát siêu tham số SVM (Linear vs RBF, C)", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "tables/svm_final_config.json"), "Metadata cấu hình huấn luyện SVM cuối cùng", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "figures/svm_tuning_heatmap.png"), "Biểu đồ nhiệt F1-score theo Kernel và C", "3.2 Kết quả thực nghiệm"),
        (os.path.join(res_base, "tables/run_timing.csv"), "Thống kê thời gian thực thi từng bước trong pipeline", "3.1 Môi trường thực nghiệm")
    ]

    lines = [
        f"# Chỉ Mục Kết Quả Thực Nghiệm (Results Index - {res_base.upper()})\n",
        f"Tài liệu này tự động liệt kê toàn bộ các bảng dữ liệu và biểu đồ được sinh ra trong thư mục `{res_base}/` phục vụ cho Chương 3 của đồ án.\n",
        "| Đường dẫn tệp | Mô tả ngắn gọn | Mục Chương 3 | Trạng thái |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for path, desc, section in expected_items:
        exists = os.path.exists(path)
        status = "✅ Đã tạo" if exists else "⏳ Chưa chạy"
        lines.append(f"| `{path}` | {desc} | {section} | {status} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Đã tạo chỉ mục kết quả tại {output_path}")

if __name__ == "__main__":
    generate_results_index()
