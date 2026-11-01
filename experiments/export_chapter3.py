import os
import json
import pandas as pd

def export_chapter3():
    # Xác định thư mục kết quả (ưu tiên results/, nếu không có thì dùng results_quick/)
    res_base = "results"
    if not os.path.exists(os.path.join(res_base, "environment.json")) and os.path.exists("results_quick/environment.json"):
        res_base = "results_quick"
    
    print(f"📂 Đang đọc dữ liệu kết quả từ thư mục: {res_base}/")

    output_md = "docs/chapter3_data.md"
    os.makedirs(os.path.dirname(output_md), exist_ok=True)

    # Đọc các file dữ liệu
    env_path = os.path.join(res_base, "environment.json")
    config_path = os.path.join(res_base, "tables/svm_final_config.json")
    metrics_path = os.path.join(res_base, "tables/evaluation_metrics.csv")
    tuning_path = os.path.join(res_base, "tables/svm_tuning.csv")
    thresh_path = os.path.join(res_base, "tables/threshold_analysis.csv")
    error_analysis_path = os.path.join(res_base, "tables/error_analysis.csv")
    comp_path = os.path.join(res_base, "tables/model_comparison.csv")
    mcnemar_path = os.path.join(res_base, "tables/mcnemar.csv")

    env_data = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env_data = json.load(f)

    final_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            final_config = json.load(f)

    df_metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    df_tuning = pd.read_csv(tuning_path) if os.path.exists(tuning_path) else None
    df_thresh = pd.read_csv(thresh_path) if os.path.exists(thresh_path) else None
    df_errors = pd.read_csv(error_analysis_path) if os.path.exists(error_analysis_path) else None
    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    df_mcnemar = pd.read_csv(mcnemar_path) if os.path.exists(mcnemar_path) else None

    # Tạo nội dung Markdown cho Chương 3
    lines = []
    lines.append("# Dữ Liệu Thực Nghiệm Chương 3: Kết Quả và Đánh Giá\n")
    lines.append(f"*Tài liệu được tự động tổng hợp từ thư mục `{res_base}/` lúc {env_data.get('timestamp', 'N/A')}.*\n\n")

    # --- 3.1 Cấu hình môi trường và tham số ---
    lines.append("## 3.1 Cấu Hình Môi Trường & Siêu Tham Số Thực Nghiệm\n")
    lines.append("### Bảng 3.1.1: Thông tin phần cứng và phần mềm hệ thống\n")
    lines.append("| Thông số | Giá trị |\n")
    lines.append("|:---|:---|\n")
    lines.append(f"| Hệ điều hành (OS) | `{env_data.get('os', 'N/A')}` |\n")
    lines.append(f"| Python Version | `{env_data.get('python_version', 'N/A')}` |\n")
    lines.append(f"| PyTorch Version | `{env_data.get('torch_version', 'N/A')}` |\n")
    lines.append(f"| Transformers Version | `{env_data.get('transformers_version', 'N/A')}` |\n")
    lines.append(f"| Scikit-Learn Version | `{env_data.get('sklearn_version', 'N/A')}` |\n")
    lines.append(f"| CPU | `{env_data.get('cpu_name', 'N/A')} ({env_data.get('cpu_cores', 'N/A')} cores)` |\n")
    lines.append(f"| RAM | `{env_data.get('ram_gb', 'N/A')} GB` |\n")
    lines.append(f"| GPU | `{env_data.get('gpu_name', 'None')}` |\n\n")

    lines.append("### Bảng 3.1.2: Cấu hình trích xuất đặc trưng và mô hình SVM cuối cùng\n")
    if final_config:
        lines.append("| Cấu hình / Tham số | Giá trị |\n")
        lines.append("|:---|:---|\n")
        for k, v in final_config.items():
            lines.append(f"| `{k}` | `{v}` |\n")
    else:
        lines.append("*Không tìm thấy file svm_final_config.json.*\n")
    lines.append("\n")

    # --- 3.2 Kết quả PhoBERT-SVM trên Test ---
    lines.append("## 3.2 Kết Quả Đánh Giá Mô Hình PhoBERT-SVM Trên Tập Test\n")
    if df_metrics is not None and not df_metrics.empty:
        m = df_metrics.iloc[0]
        lines.append("### Bảng 3.2.1: Hiệu năng chi tiết theo lớp (Nhãn 0: Thật, Nhãn 1: Giả)\n")
        lines.append("| Lớp (Class) | Precision (%) | Recall (%) | F1-Score (%) |\n")
        lines.append("|:---|:---:|:---:|:---:|\n")
        lines.append(f"| Tin thật (Real - 0) | `{m.get('real_precision', 0)*100:.2f}` | `{m.get('real_recall', 0)*100:.2f}` | `{m.get('real_f1', 0)*100:.2f}` |\n")
        lines.append(f"| **Tin giả (Fake - 1)** | **`{m.get('fake_precision', 0)*100:.2f}`** | **`{m.get('fake_recall', 0)*100:.2f}`** | **`{m.get('fake_f1', 0)*100:.2f}`** |\n\n")

        lines.append(f"- **Accuracy toàn cục**: `{m.get('accuracy', 0)*100:.2f}%`\n")
        lines.append(f"- **AUC-ROC**: `{m.get('auc_roc', 0):.4f}`\n")
        lines.append(f"- **Average Precision (AP)**: `{m.get('average_precision', 0):.4f}`\n")
        lines.append(f"- **Khoảng tin cậy 95% F1 (Tin giả - Bootstrap CI)**: `[{m.get('fake_f1_ci_lower', 0)*100:.2f}%, {m.get('fake_f1_ci_upper', 0)*100:.2f}%]`\n\n")

        lines.append("### Bảng 3.2.2: Ma trận nhầm lẫn (Confusion Matrix)\n")
        lines.append("| Metric | Giá trị |\n")
        lines.append("|:---|:---:|\n")
        lines.append(f"| True Negatives (TN - Thật đoán Thật) | `{int(m.get('TN', 0))}` |\n")
        lines.append(f"| False Positives (FP - Thật đoán Giả) | `{int(m.get('FP', 0))}` |\n")
        lines.append(f"| False Negatives (FN - Giả đoán Thật) | `{int(m.get('FN', 0))}` |\n")
        lines.append(f"| True Positives (TP - Giả đoán Giả) | `{int(m.get('TP', 0))}` |\n\n")

    lines.append("### Hình minh họa đánh giá\n")
    lines.append(f"- Biểu đồ Confusion Matrix: `{res_base}/figures/confusion_matrix.png`\n")
    lines.append(f"- Đường cong ROC-AUC: `{res_base}/figures/roc_curve.png`\n")
    lines.append(f"- Đường cong Precision-Recall: `{res_base}/figures/pr_curve.png`\n\n")

    # --- 3.3 Kết quả chọn siêu tham số trên Validation và Phân tích ngưỡng ---
    lines.append("## 3.3 Khảo Sát Siêu Tham Số & Phân Tích Ngưỡng (Threshold Analysis)\n")
    if df_tuning is not None and not df_tuning.empty:
        lines.append("### Bảng 3.3.1: Kết quả Grid Search SVM trên tập Validation (`svm_tuning.csv`)\n")
        lines.append(df_tuning.to_markdown(index=False))
        lines.append("\n\n")
    
    if df_thresh is not None and not df_thresh.empty:
        lines.append("### Bảng 3.3.2: Phân tích hiệu năng theo các ngưỡng quyết định khác nhau (`threshold_analysis.csv`)\n")
        lines.append(df_thresh.to_markdown(index=False))
        lines.append("\n\n")

    # --- 3.4 Phân tích lỗi ---
    lines.append("## 3.4 Phân Tích Mẫu Lỗi (Error Analysis)\n")
    fp_count = len(df_errors[df_errors['error_type'] == 'FP']) if df_errors is not None and 'error_type' in df_errors.columns else 0
    fn_count = len(df_errors[df_errors['error_type'] == 'FN']) if df_errors is not None and 'error_type' in df_errors.columns else 0
    lines.append(f"- **Tổng số mẫu False Positive (FP)**: `{fp_count}`\n")
    lines.append(f"- **Tổng số mẫu False Negative (FN)**: `{fn_count}`\n")
    lines.append(f"- **Chi tiết mẫu lỗi định tính**: Xem tại file `{res_base}/error_samples.md`.\n\n")

    # --- 3.5 So sánh baseline ---
    lines.append("## 3.5 So Sánh Với Các Mô Hình Baseline & Kiểm Định Thống Kê\n")
    if df_comp is not None and not df_comp.empty:
        lines.append("### Bảng 3.5.1: So sánh hiệu năng PhoBERT-SVM và các baseline (`model_comparison.csv`)\n")
        lines.append(df_comp.to_markdown(index=False))
        lines.append("\n\n")

    if df_mcnemar is not None and not df_mcnemar.empty:
        lines.append("### Bảng 3.5.2: Kết quả kiểm định McNemar so sánh biến thể chuẩn với các biến thể khác (`mcnemar.csv`)\n")
        lines.append(df_mcnemar.to_markdown(index=False))
        lines.append("\n\n")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"🎉 Đã xuất thành công báo cáo Chương 3 tại: {output_md}")

if __name__ == "__main__":
    export_chapter3()
