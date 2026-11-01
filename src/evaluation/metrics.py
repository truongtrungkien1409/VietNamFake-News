import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc
)

def compute_metrics(y_true, y_pred, y_score):
    """
    Tính toán toàn diện các chỉ số đánh giá theo DACN.pdf:
    - Accuracy, Precision, Recall, F1 theo từng lớp (0: Tin thật, 1: Tin giả)
    - Macro avg, Weighted avg
    - AUC-ROC, Average Precision (AP)
    - Confusion Matrix components: TN, FP, FN, TP (đối với lớp dương là Tin giả - 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, support, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    f1_classes = 2 * precision * recall / (precision + recall + 1e-10)

    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except Exception:
        auc_roc = 0.5

    try:
        ap = average_precision_score(y_true, y_score)
    except Exception:
        ap = 0.0

    # Confusion matrix (TN, FP, FN, TP với positive class = 1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics_dict = {
        "accuracy": float(acc),
        "real_precision": float(precision[0]),
        "real_recall": float(recall[0]),
        "real_f1": float(f1_classes[0]),
        "real_support": int(support[0]),
        "fake_precision": float(precision[1]),
        "fake_recall": float(recall[1]),
        "fake_f1": float(f1_classes[1]),
        "fake_support": int(support[1]),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "auc_roc": float(auc_roc),
        "average_precision": float(ap),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }
    return metrics_dict

def save_confusion_matrix(y_true, y_pred, path):
    """Vẽ và lưu Confusion Matrix với nhãn tiếng Việt (dpi=200)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Tin thật (0)", "Tin giả (1)"],
                yticklabels=["Tin thật (0)", "Tin giả (1)"])
    plt.title("Ma trận nhầm lẫn (Confusion Matrix)", fontsize=12, fontweight='bold')
    plt.xlabel("Nhãn dự đoán", fontsize=10)
    plt.ylabel("Nhãn thực tế", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def save_roc_curve(y_true, y_score, path):
    """Vẽ và lưu đường cong ROC kèm giá trị AUC (dpi=200)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ dương tính giả (FPR)', fontsize=10)
    plt.ylabel('Tỷ lệ dương tính thật (TPR)', fontsize=10)
    plt.title('Đường cong ROC-AUC', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def save_pr_curve(y_true, y_score, path):
    """Vẽ và lưu đường cong Precision-Recall kèm AP (dpi=200)."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Độ bao phủ (Recall)', fontsize=10)
    plt.ylabel('Độ chính xác (Precision)', fontsize=10)
    plt.title('Đường cong Precision-Recall', fontsize=12, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def bootstrap_ci(y_true, y_pred, metric_fn, n=1000, seed=42):
    """Tính khoảng tin cậy 95% bằng phương pháp Bootstrap."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    scores = []

    for _ in range(n):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        # Tránh trường hợp bootstrap chỉ có 1 lớp
        if len(np.unique(sample_true)) < 2:
            continue
        score = metric_fn(sample_true, sample_pred)
        scores.append(score)

    if not scores:
        return 0.0, 0.0

    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return float(lower), float(upper)

def mcnemar_test(y_true, pred_a, pred_b):
    """Phép kiểm định McNemar so sánh 2 mô hình A và B."""
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n01 = np.sum(~correct_a & correct_b) # A sai, B đúng
    n10 = np.sum(correct_a & ~correct_b) # A đúng, B sai

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[np.sum(correct_a & correct_b), n01],
                 [n10, np.sum(~correct_a & ~correct_b)]]
        res = mcnemar(table, exact=False, correction=True)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Fallback tính thủ công / nhị thức
        from scipy.stats import binom
        n = n01 + n10
        if n == 0:
            return 0.0, 1.0
        p_val = 2 * min(binom.cdf(min(n01, n10), n, 0.5), 1.0)
        stat = (abs(n01 - n10) - 1)**2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
        return float(stat), float(p_val)
