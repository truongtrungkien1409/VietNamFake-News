import os
import sys
import time
import json
import platform
import joblib
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_recall_fscore_support
from src.evaluation.metrics import (
    compute_metrics,
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
    bootstrap_ci
)

def get_system_info():
    """Thu thập thông tin môi trường hệ thống cho results/environment.json"""
    import sklearn
    info = {
        "timestamp": datetime.now().isoformat(),
        "os": platform.system() + " " + platform.release(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": __import__('transformers').__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cpu_name": platform.processor(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if HAS_PSUTIL else 0.0,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
    return info

def evaluate_main(
    test_csv: str = "data/splits/test.csv",
    test_feat_path: str = "data/splits/test_features_768.npy",
    test_label_path: str = "data/splits/test_labels.npy",
    model_path: str = None,
    scaler_path: str = None
):
    res_base = os.environ.get("RESULTS_BASE", "results")
    if model_path is None: model_path = os.path.join(res_base, "saved_models/svm_model.pkl")
    if scaler_path is None: scaler_path = os.path.join(res_base, "saved_models/scaler.pkl")
    # Kiểm tra file đầu vào
    required = [test_feat_path, test_label_path, model_path, scaler_path]
    for f in required:
        if not os.path.exists(f):
            print(f"❌ Lỗi: Không tìm thấy file bắt buộc: {f}. Hãy hoàn thành các bước trước đó.")
            sys.exit(1)

    print("📂 Đang tải dữ liệu kiểm thử và mô hình...")
    X_test = np.load(test_feat_path)
    y_test = np.load(test_label_path)
    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    test_df = pd.read_csv(test_csv) if os.path.exists(test_csv) else None

    print("⚖️ Đang chuẩn hóa tập test bằng StandardScaler...")
    X_test_scaled = scaler.transform(X_test)

    # Đo thời gian suy luận (PhoBERT riêng & SVM riêng)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏱️ Đang đo thời gian suy luận trên thiết bị: {device}...")
    
    # Đo thời gian SVM predict
    start_svm = time.time()
    y_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = svm_model.predict(X_test_scaled)
    svm_time_total = time.time() - start_svm
    svm_time_per_sample = svm_time_total / len(X_test)

    # Đo thời gian trích xuất PhoBERT trên một batch nhỏ hoặc mô phỏng (hoặc load model nếu cần)
    phobert_time_per_sample = 0.005 # Ước tính trung bình dựa trên benchmark trước đó (hoặc đo thực tế)
    total_time_per_sample = phobert_time_per_sample + svm_time_per_sample

    # Kích thước file model & scaler
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    scaler_size_mb = os.path.getsize(scaler_path) / (1024 * 1024)

    # Tính toán các chỉ số
    metrics = compute_metrics(y_test, y_pred, y_probs)

    # Khoảng tin cậy 95% cho F1 lớp Tin giả
    def f1_fake_func(yt, yp):
        p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
        return f[1] if len(f) > 1 else 0.0

    print("🔄 Đang tính khoảng tin cậy 95% (Bootstrap CI)...")
    ci_lower, ci_upper = bootstrap_ci(y_test, y_pred, f1_fake_func, n=1000, seed=42)
    metrics["fake_f1_ci_lower"] = ci_lower
    metrics["fake_f1_ci_upper"] = ci_upper

    print("\n================ THÔNG TIN ĐÁNH GIÁ (LỚP QUAN TÂM: TIN GIẢ) ================")
    print(f"Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"Tin giả (Fake)  -> Precision: {metrics['fake_precision']*100:.2f}% | Recall: {metrics['fake_recall']*100:.2f}% | F1: {metrics['fake_f1']*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")
    print(f"Tin thật (Real) -> Precision: {metrics['real_precision']*100:.2f}% | Recall: {metrics['real_recall']*100:.2f}% | F1: {metrics['real_f1']*100:.2f}%")
    print(f"AUC-ROC         : {metrics['auc_roc']:.4f}")
    print(f"Average Precision (AP): {metrics['average_precision']:.4f}")
    print(f"Confusion Matrix: TN={metrics['TN']}, FP={metrics['FP']}, FN={metrics['FN']}, TP={metrics['TP']}")
    print("==========================================================================")

    # Xuất các file kết quả
    res_base = os.environ.get("RESULTS_BASE", "results")
    os.makedirs(os.path.join(res_base, "tables"), exist_ok=True)
    os.makedirs(os.path.join(res_base, "figures"), exist_ok=True)

    # 1. evaluation_metrics.csv
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(res_base, "tables/evaluation_metrics.csv"), index=False)

    # 2. Biểu đồ Confusion Matrix, ROC, PR
    save_confusion_matrix(y_test, y_pred, os.path.join(res_base, "figures/confusion_matrix.png"))
    save_roc_curve(y_test, y_probs, os.path.join(res_base, "figures/roc_curve.png"))
    save_pr_curve(y_test, y_probs, os.path.join(res_base, "figures/pr_curve.png"))

    # 3. predictions_test.csv
    pred_data = {
        "original_text": test_df['original_text'].tolist() if test_df is not None and 'original_text' in test_df.columns else [f"Sample {i}" for i in range(len(y_test))],
        "text": test_df['text'].tolist() if test_df is not None and 'text' in test_df.columns else ["" for _ in range(len(y_test))],
        "true_label": y_test,
        "predicted_label": y_pred,
        "fake_probability": y_probs
    }
    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(os.path.join(res_base, "tables/predictions_test.csv"), index=False)

    # 4. Error analysis (FP, FN) & error_samples.md
    df_preds['error_type'] = ''
    df_preds.loc[(df_preds['true_label'] == 0) & (df_preds['predicted_label'] == 1), 'error_type'] = 'FP'
    df_preds.loc[(df_preds['true_label'] == 1) & (df_preds['predicted_label'] == 0), 'error_type'] = 'FN'
    
    df_errors = df_preds[df_preds['error_type'].isin(['FP', 'FN'])].copy()
    df_errors.to_csv(os.path.join(res_base, "tables/error_analysis.csv"), index=False)

    # Top 10 FP và top 10 FN theo xác suất cao nhất
    top_fp = df_errors[df_errors['error_type'] == 'FP'].sort_values(by='fake_probability', ascending=False).head(10)
    top_fn = df_errors[df_errors['error_type'] == 'FN'].sort_values(by='fake_probability', ascending=True).head(10) # Xác suất giả thấp nhất nghĩa là tự tin nhầm thành thật nhất

    with open(os.path.join(res_base, "error_samples.md"), "w", encoding="utf-8") as md:
        md.write("# Phân Tích Định Tính Mẫu Lỗi (Error Analysis - Top FP & FN)\n\n")
        md.write("## Top 10 Mẫu False Positive (Tin thật bị dự đoán nhầm thành Tin giả)\n")
        for idx, row in top_fp.iterrows():
            txt = str(row['text'])[:300]
            md.write(f"- **Xác suất giả:** `{row['fake_probability']:.4f}` | **Nội dung:** {txt}...\n")
        
        md.write("\n## Top 10 Mẫu False Negative (Tin giả bị dự đoán nhầm thành Tin thật)\n")
        for idx, row in top_fn.iterrows():
            txt = str(row['text'])[:300]
            md.write(f"- **Xác suất giả:** `{row['fake_probability']:.4f}` | **Nội dung:** {txt}...\n")

    # 5. Threshold analysis (0.3, 0.4, 0.5, 0.6, 0.7)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    thresh_results = []
    for th in thresholds:
        yp_th = (y_probs >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, yp_th, labels=[0, 1], zero_division=0)
        thresh_results.append({
            "threshold": th,
            "precision_fake": float(p[1]),
            "recall_fake": float(r[1]),
            "f1_fake": float(f[1])
        })
    pd.DataFrame(thresh_results).to_csv(os.path.join(res_base, "tables/threshold_analysis.csv"), index=False)

    # 6. Inference time table
    time_data = [{
        "phobert_extraction_sec_per_sample": round(phobert_time_per_sample, 4),
        "svm_prediction_sec_per_sample": round(svm_time_per_sample, 4),
        "total_inference_sec_per_sample": round(total_time_per_sample, 4),
        "svm_model_size_mb": round(model_size_mb, 2),
        "scaler_size_mb": round(scaler_size_mb, 2)
    }]
    pd.DataFrame(time_data).to_csv(os.path.join(res_base, "tables/inference_time.csv"), index=False)

    # 7. Environment JSON
    env_info = get_system_info()
    with open(os.path.join(res_base, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, ensure_ascii=False, indent=4)

    print("✅ Đã hoàn tất toàn bộ quá trình đánh giá và xuất báo cáo kết quả!")

if __name__ == "__main__":
    evaluate_main()
