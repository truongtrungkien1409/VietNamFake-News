import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.baselines.naive_bayes import run_naive_bayes_baseline
from src.models.baselines.svm_tfidf import run_svm_tfidf_baseline
from src.models.baselines.lstm import run_lstm_baseline
from src.models.baselines.cnn import run_cnn_baseline
from src.evaluation.metrics import mcnemar_test
from src.config import get_paths

def compare_all_models(output_csv: str = None):
    paths = get_paths()
    res_base = paths.results_base
    if output_csv is None:
        output_csv = os.path.join(paths.tables, "model_comparison.csv")

    print("📊 [Model Comparison] Đang chạy đánh giá so sánh các mô hình trên tập Test...")
    
    nb_metrics = run_naive_bayes_baseline()
    svm_tfidf_metrics = run_svm_tfidf_baseline()
    lstm_metrics = run_lstm_baseline()
    cnn_metrics = run_cnn_baseline()

    svm_metrics_path = os.path.join(paths.tables, "evaluation_metrics.csv")
    svm_config_path = os.path.join(paths.tables, "svm_final_config.json")
    timing_path = os.path.join(paths.tables, "run_timing.csv")
    inference_time_path = os.path.join(paths.tables, "inference_time.csv")

    svm_metrics = {}
    if os.path.exists(svm_metrics_path):
        df_svm = pd.read_csv(svm_metrics_path)
        svm_metrics = df_svm.iloc[0].to_dict()

    train_time_svm = float('nan')
    if os.path.exists(svm_config_path):
        try:
            with open(svm_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                train_time_svm = float(cfg.get("training_time_sec", cfg.get("training_time", float('nan'))))
        except Exception:
            pass

    extract_time = 0.0
    if os.path.exists(timing_path):
        try:
            df_timing = pd.read_csv(timing_path)
            row_ext = df_timing[df_timing.iloc[:, 0].astype(str).str.contains("extract_phobert", case=False, na=False)]
            if len(row_ext) > 0 and 'duration_sec' in row_ext.columns:
                extract_time = float(row_ext['duration_sec'].values[0])
        except Exception:
            pass

    if not np.isnan(train_time_svm):
        svm_metrics["training_time_sec"] = train_time_svm + extract_time
    else:
        svm_metrics["training_time_sec"] = float('nan')

    inf_sec = float('nan')
    if os.path.exists(inference_time_path):
        try:
            df_inf = pd.read_csv(inference_time_path)
            if 'total_inference_sec_per_sample' in df_inf.columns:
                inf_sec = float(df_inf['total_inference_sec_per_sample'].values[0])
            elif 'inference_sec_per_sample' in df_inf.columns:
                inf_sec = float(df_inf['inference_sec_per_sample'].values[0])
        except Exception:
            pass
    svm_metrics["inference_sec_per_sample"] = inf_sec

    rows = []
    model_preds = {}
    y_true = None

    phobert_pred_path = os.path.join(paths.tables, "predictions_test.csv")
    if os.path.exists(phobert_pred_path):
        df_phobert = pd.read_csv(phobert_pred_path)
        model_preds["PhoBERT-SVM (Đề xuất)"] = df_phobert['predicted_label'].values
        y_true = df_phobert['true_label'].values

    for name, m, pred_file in [
        ("Naive Bayes (TF-IDF)", nb_metrics, "nb_preds.csv"),
        ("SVM (TF-IDF)", svm_tfidf_metrics, "predictions_svm_tfidf.csv"),
        ("TextLSTM", lstm_metrics, "lstm_preds.csv"),
        ("TextCNN", cnn_metrics, "cnn_preds.csv"),
        ("PhoBERT-SVM (Đề xuất)", svm_metrics, "predictions_test.csv")
    ]:
        if m:
            rows.append({
                "Model": name,
                "Accuracy": m.get("accuracy", float('nan')),
                "Fake_Precision": m.get("fake_precision", float('nan')),
                "Fake_Recall": m.get("fake_recall", float('nan')),
                "Fake_F1": m.get("fake_f1", float('nan')),
                "AUC_ROC": m.get("auc_roc", float('nan')),
                "Training_Time_sec": m.get("training_time_sec", float('nan')),
                "Inference_Sec_Per_Sample": m.get("inference_sec_per_sample", float('nan'))
            })
        pred_path = os.path.join(paths.tables, pred_file)
        if os.path.exists(pred_path):
            df_p = pd.read_csv(pred_path)
            pred_col = 'predicted_label' if 'predicted_label' in df_p.columns else ('pred' if 'pred' in df_p.columns else df_p.columns[1])
            model_preds[name] = df_p[pred_col].values

    os.makedirs(paths.tables, exist_ok=True)
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(output_csv, index=False)
    print(f"\n🏆 BẢNG SO SÁNH CÁC MÔ HÌNH (Lưu tại {output_csv}):")
    print(df_comp.to_string(index=False))

    mcnemar_rows = []
    if "PhoBERT-SVM (Đề xuất)" in model_preds and y_true is not None:
        phobert_preds = model_preds["PhoBERT-SVM (Đề xuất)"]
        for name, preds in model_preds.items():
            if name == "PhoBERT-SVM (Đề xuất)":
                continue
            if len(preds) == len(y_true) and len(phobert_preds) == len(y_true):
                stat, pval = mcnemar_test(y_true, phobert_preds, preds)
                correct_a = (phobert_preds == y_true)
                correct_b = (preds == y_true)
                n01 = int(np.sum(~correct_a & correct_b))
                n10 = int(np.sum(correct_a & ~correct_b))
                discordant = n01 + n10
                mcnemar_rows.append({
                    "comparison": f"PhoBERT-SVM vs {name}",
                    "mcnemar_statistic": round(stat, 4),
                    "p_value": float(pval),
                    "discordant_samples": discordant,
                    "significant_at_0.05": bool(pval < 0.05)
                })

    if mcnemar_rows:
        df_mcnemar = pd.DataFrame(mcnemar_rows)
        mcnemar_path = os.path.join(paths.tables, "mcnemar.csv")
        df_mcnemar.to_csv(mcnemar_path, index=False)
        print(f"💾 Đã lưu kết quả kiểm định McNemar tại {mcnemar_path}")

    try:
        os.makedirs(paths.figures, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_comp, x="Model", y="Fake_F1", palette="Blues_d")
        plt.title("So sánh F1-score lớp Tin giả giữa PhoBERT-SVM và các Baseline")
        plt.xlabel("Mô hình")
        plt.ylabel("F1-score (Tin giả)")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=15)
        plt.tight_layout()
        bar_path = os.path.join(paths.figures, "model_comparison_bar.png")
        plt.savefig(bar_path, dpi=300)
        plt.close()
        print(f"📊 Đã vẽ biểu đồ so sánh mô hình tại {bar_path}")
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ so sánh: {e}")

if __name__ == "__main__":
    compare_all_models()
