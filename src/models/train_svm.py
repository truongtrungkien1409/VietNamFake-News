import os
import time
import json
import random
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score
)

# Cố định random seed (Quy ước cố định 42)
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def train_and_tune_svm(
    train_feat_path: str = "data/splits/train_features_768.npy",
    train_label_path: str = "data/splits/train_labels.npy",
    val_feat_path: str = "data/splits/val_features_768.npy",
    val_label_path: str = "data/splits/val_labels.npy",
    model_output_path: str = None,
    scaler_output_path: str = None,
    tuning_table_path: str = None,
    final_config_path: str = None,
    heatmap_path: str = None,
    use_best: bool = False,
    class_weight_arg = None
):
    """
    Huấn luyện và tinh chỉnh siêu tham số SVM trên tập Validation theo DACN.pdf (Mục 2.4.2).
    """
    res_base = os.environ.get("RESULTS_BASE", "results")
    if model_output_path is None: model_output_path = os.path.join(res_base, "saved_models/svm_model.pkl")
    if scaler_output_path is None: scaler_output_path = os.path.join(res_base, "saved_models/scaler.pkl")
    if tuning_table_path is None: tuning_table_path = os.path.join(res_base, "tables/svm_tuning.csv")
    if final_config_path is None: final_config_path = os.path.join(res_base, "tables/svm_final_config.json")
    if heatmap_path is None: heatmap_path = os.path.join(res_base, "figures/svm_tuning_heatmap.png")
    required_files = [train_feat_path, train_label_path, val_feat_path, val_label_path]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ Lỗi: Không tìm thấy file dữ liệu đặc trưng tại {f}. Vui lòng chạy trích xuất đặc trưng trước.")
            return

    print("📂 Đang tải ma trận đặc trưng Train và Validation...")
    X_train = np.load(train_feat_path)
    y_train = np.load(train_label_path)
    X_val = np.load(val_feat_path)
    y_val = np.load(val_label_path)

    print("⚖️ Đang chuẩn hóa dữ liệu bằng StandardScaler (Fit CHỈ trên tập Train)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    kernels = ['linear', 'rbf']
    C_values = [0.1, 1.0, 10.0]
    class_weights = [None, 'balanced'] if class_weight_arg is None else [class_weight_arg]
    
    tuning_results = []
    print("\n🔬 Bắt đầu quá trình Grid Search / Khảo sát siêu tham số trên tập Validation:")
    best_f1_fake = -1.0
    best_config = None

    for kernel in kernels:
        for C in C_values:
            for cw in class_weights:
                cw_name = str(cw)
                print(f"   - Đang huấn luyện SVM (kernel='{kernel}', C={C}, class_weight={cw_name})...")
                start_t = time.time()
                # Dùng probability=False khi tuning cho nhanh
                svm_tune = SVC(
                    kernel=kernel,
                    C=C,
                    gamma='scale',
                    class_weight=cw,
                    probability=False,
                    random_state=RANDOM_STATE
                )
                svm_tune.fit(X_train_scaled, y_train)
                train_time = time.time() - start_t

                # Dự đoán trên tập train để kiểm tra overfitting
                y_train_pred = svm_tune.predict(X_train_scaled)
                train_acc = accuracy_score(y_train, y_train_pred)

                # Dự đoán trên tập Validation
                y_val_pred = svm_tune.predict(X_val_scaled)
                val_acc = accuracy_score(y_val, y_val_pred)

                # Chỉ số cho lớp Tin giả (label = 1)
                precision, recall, f1_cls, _ = precision_recall_fscore_support(y_val, y_val_pred, average=None, labels=[0, 1])
                prec_fake = precision[1] if len(precision) > 1 else 0.0
                rec_fake = recall[1] if len(recall) > 1 else 0.0
                f1_fake = f1_cls[1] if len(f1_cls) > 1 else 0.0
                
                f1_macro = f1_score(y_val, y_val_pred, average='macro')

                # Tính AUC bằng decision_function
                try:
                    decision_scores = svm_tune.decision_function(X_val_scaled)
                    auc_val = roc_auc_score(y_val, decision_scores)
                except Exception:
                    auc_val = 0.5

                n_support = int(np.sum(svm_tune.n_support_))

                print(f"     -> Train Acc: {train_acc*100:.2f}% (Overfitting check) | Val Acc: {val_acc*100:.2f}% | F1 (Fake): {f1_fake*100:.2f}% | AUC: {auc_val:.4f} | Support Vectors: {n_support}")

                tuning_results.append({
                    "kernel": kernel,
                    "C": C,
                    "class_weight": cw_name,
                    "train_accuracy": round(train_acc, 4),
                    "accuracy": round(val_acc, 4),
                    "precision_fake": round(prec_fake, 4),
                    "recall_fake": round(rec_fake, 4),
                    "f1_fake": round(f1_fake, 4),
                    "f1_macro": round(f1_macro, 4),
                    "auc": round(auc_val, 4),
                    "n_support": n_support,
                    "training_time_sec": round(train_time, 2)
                })

                if f1_fake > best_f1_fake:
                    best_f1_fake = f1_fake
                    best_config = (kernel, C, cw)

    # Lưu bảng kết quả tuning
    os.makedirs(os.path.dirname(tuning_table_path), exist_ok=True)
    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv(tuning_table_path, index=False)
    print(f"\n💾 Đã lưu bảng kết quả khảo sát tham số tại {tuning_table_path}")

    print(f"\n🏆 Cấu hình tốt nhất trên Validation (theo F1 Tin giả): kernel='{best_config[0]}', C={best_config[1]}, class_weight={best_config[2]} (F1 Fake: {best_f1_fake*100:.2f}%)")

    # Vẽ Heatmap F1 Tin giả theo kernel x C (với class_weight=None hoặc phổ biến)
    try:
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        df_hm = df_tuning[df_tuning['class_weight'] == 'None'].pivot(index='kernel', columns='C', values='f1_fake')
        sns.heatmap(df_hm, annot=True, fmt=".4f", cmap="Blues", cbar=True)
        plt.title("Biểu đồ nhiệt F1-score lớp Tin giả theo Kernel và C")
        plt.xlabel("Hằng số phạt lỗi C")
        plt.ylabel("Hàm nhân (Kernel)")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"📊 Đã vẽ biểu đồ nhiệt tại {heatmap_path}")
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ nhiệt: {e}")

    # Xác định cấu hình cuối cùng để huấn luyện mô hình
    final_kernel = "linear"
    final_C = 1.0
    final_cw = None

    if use_best:
        print(f"⚙️ [--use-best được kích hoạt] Sử dụng cấu hình tốt nhất: kernel='{best_config[0]}', C={best_config[1]}...")
        final_kernel, final_C, final_cw = best_config
    else:
        default_config = ("linear", 1.0, None)
        if best_config != default_config:
            # Tính chênh lệch F1
            default_row = df_tuning[(df_tuning['kernel'] == 'linear') & (df_tuning['C'] == 1.0) & (df_tuning['class_weight'] == 'None')]
            default_f1 = default_row['f1_fake'].values[0] if len(default_row) > 0 else 0.0
            diff = best_f1_fake - default_f1
            print(f"⚠️ [Cảnh báo] Cấu hình tốt nhất trên Val (F1={best_f1_fake*100:.2f}%) khác với mặc định báo cáo (Linear, C=1.0, F1={default_f1*100:.2f}%, chênh lệch: {diff*100:+.2f}%).")
            print("   Theo DACN.pdf, hệ thống giữ nguyên cấu hình chuẩn báo cáo (kernel='linear', C=1.0). Dùng --use-best nếu muốn đổi.")

    print(f"\n⚙️ Huấn luyện mô hình CUỐI CÙNG (kernel='{final_kernel}', C={final_C}, class_weight={final_cw}, probability=True)...")
    start_final = time.time()
    final_svm = SVC(
        kernel=final_kernel,
        C=final_C,
        gamma='scale',
        class_weight=final_cw,
        probability=True,
        random_state=RANDOM_STATE
    )
    final_svm.fit(X_train_scaled, y_train)
    final_train_time = time.time() - start_final
    n_support_final = int(np.sum(final_svm.n_support_))

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(final_svm, model_output_path)
    joblib.dump(scaler, scaler_output_path)
    print(f"💾 Đã lưu mô hình chính thức tại {model_output_path} và scaler tại {scaler_output_path}.")

    # Lưu metadata cấu hình cuối cùng
    import sklearn
    config_meta = {
        "kernel": final_kernel,
        "C": final_C,
        "gamma": "scale",
        "class_weight": str(final_cw),
        "n_support": n_support_final,
        "seed": RANDOM_STATE,
        "training_time_sec": round(final_train_time, 2),
        "sklearn_version": sklearn.__version__
    }
    with open(final_config_path, "w", encoding="utf-8") as f:
        json.dump(config_meta, f, ensure_ascii=False, indent=4)
    print(f"💾 Đã lưu file cấu hình cuối cùng tại {final_config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và tinh chỉnh mô hình SVM nhận diện tin giả.")
    parser.add_argument("--use-best", action="store_true", help="Sử dụng cấu hình tốt nhất từ Validation thay vì mặc định báo cáo.")
    parser.add_argument("--class-weight", type=str, default=None, help="Cấu hình class_weight (ví dụ: balanced).")
    args = parser.parse_args()

    cw_param = args.class_weight
    if cw_param == "None":
        cw_param = None

    train_and_tune_svm(use_best=args.use_best, class_weight_arg=cw_param)
