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
from src.config import get_paths
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
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            import subprocess
            res = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=3)
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if len(lines) > 1:
                cpu_name = lines[1]
        except Exception:
            pass

    info = {
        "timestamp": datetime.now().isoformat(),
        "os": platform.system() + " " + platform.release(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": __import__('transformers').__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cpu_name": cpu_name,
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if HAS_PSUTIL else None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }
    return info

def evaluate_main(
    test_csv: str = None,
    test_feat_path: str = None,
    test_label_path: str = None,
    model_path: str = None,
    scaler_path: str = None
):
    paths = get_paths()
    if test_csv is None: test_csv = os.path.join(paths.splits, "test.csv")
    if test_feat_path is None: test_feat_path = os.path.join(paths.splits, "test_features_768.npy")
    if test_label_path is None: test_label_path = os.path.join(paths.splits, "test_labels.npy")
    if model_path is None: model_path = os.path.join(paths.saved_models, "svm_model.pkl")
    if scaler_path is None: scaler_path = os.path.join(paths.saved_models, "scaler.pkl")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⏱️ Đang đo thời gian suy luận trên thiết bị: {device}...")
    
    start_svm = time.time()
    y_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = svm_model.predict(X_test_scaled)
    svm_time_total = time.time() - start_svm
    svm_time_per_sample = svm_time_total / len(X_test)

    # Đo thời gian trích xuất PhoBERT thật trên 100 mẫu test đầu tiên (sau 1 lần khởi động, lấy trung bình mỗi mẫu)
    phobert_time_per_sample = float('nan')
    try:
        sample_texts = test_df['text'].head(100).astype(str).tolist() if test_df is not None and 'text' in test_df.columns else [f"Sample {i}" for i in range(min(100, len(X_test)))]
        if sample_texts:
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
            model.eval()
            # Warm-up run (1 lần khởi động)
            with torch.no_grad():
                encoded_warmup = tokenizer(sample_texts[:1], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                _ = model(**encoded_warmup)
            
            start_phobert = time.time()
            with torch.no_grad():
                encoded = tokenizer(sample_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                _ = model(**encoded)
            phobert_elapsed = time.time() - start_phobert
            phobert_time_per_sample = phobert_elapsed / len(sample_texts)
    except Exception as e:
        print(f"⚠️ Không thể đo thời gian trích xuất PhoBERT thực tế: {e}")

    total_time_per_sample = (phobert_time_per_sample if not np.isnan(phobert_time_per_sample) else 0.0) + svm_time_per_sample

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    scaler_size_mb = os.path.getsize(scaler_path) / (1024 * 1024)

    metrics = compute_metrics(y_test, y_pred, y_probs)
    metrics["inference_sec_per_sample"] = round(total_time_per_sample, 6)

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

    res_base = paths.results_base
    os.makedirs(os.path.join(res_base, "tables"), exist_ok=True)
    os.makedirs(os.path.join(res_base, "figures"), exist_ok=True)

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(res_base, "tables/evaluation_metrics.csv"), index=False)

    env_info = get_system_info()
    with open(os.path.join(res_base, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env_info, f, ensure_ascii=False, indent=4)

    save_confusion_matrix(y_test, y_pred, os.path.join(res_base, "figures/confusion_matrix.png"))
    save_roc_curve(y_test, y_probs, os.path.join(res_base, "figures/roc_curve.png"))
    save_pr_curve(y_test, y_probs, os.path.join(res_base, "figures/pr_curve.png"))

    pred_data = {
        "original_text": test_df['original_text'].tolist() if test_df is not None and 'original_text' in test_df.columns else [f"Sample {i}" for i in range(len(y_test))],
        "text": test_df['text'].tolist() if test_df is not None and 'text' in test_df.columns else ["" for _ in range(len(y_test))],
        "true_label": y_test,
        "predicted_label": y_pred,
        "fake_probability": y_probs
    }
    df_preds = pd.DataFrame(pred_data)
    df_preds.to_csv(os.path.join(res_base, "tables/predictions_test.csv"), index=False)

    df_preds['error_type'] = ''
    df_preds.loc[(df_preds['true_label'] == 0) & (df_preds['predicted_label'] == 1), 'error_type'] = 'FP'
    df_preds.loc[(df_preds['true_label'] == 1) & (df_preds['predicted_label'] == 0), 'error_type'] = 'FN'
    
    df_errors = df_preds[df_preds['error_type'].isin(['FP', 'FN'])].copy()
    df_errors.to_csv(os.path.join(res_base, "tables/error_analysis.csv"), index=False)

    # Thêm inference_time.csv
    df_inf = pd.DataFrame([{
        "phobert_time_per_sample_sec": phobert_time_per_sample,
        "svm_time_per_sample_sec": svm_time_per_sample,
        "total_inference_sec_per_sample": total_time_per_sample
    }])
    df_inf.to_csv(os.path.join(res_base, "tables/inference_time.csv"), index=False)

    # Threshold analysis
    thresh_results = []
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_p_th = (y_probs >= th).astype(int)
        m_th = compute_metrics(y_test, y_p_th, y_probs)
        thresh_results.append({
            "threshold": th,
            "accuracy": m_th['accuracy'],
            "fake_precision": m_th['fake_precision'],
            "fake_recall": m_th['fake_recall'],
            "fake_f1": m_th['fake_f1']
        })
    pd.DataFrame(thresh_results).to_csv(os.path.join(res_base, "tables/threshold_analysis.csv"), index=False)

if __name__ == "__main__":
    evaluate_main()
