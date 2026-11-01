import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.preprocessing.preprocess import TextPreprocessor
from src.evaluation.metrics import compute_metrics, bootstrap_ci, mcnemar_test
from src.config import get_paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS_CONFIG = {
    "A": {
        "name": "A: Không tiền xử lý",
        "params": dict(lowercase=False, clean_url_html=False, hashtag_punct=False, teencode_emoji=False, repeat_chars=False, word_segmentation=False)
    },
    "B": {
        "name": "B: Chỉ URL/HTML cleanup",
        "params": dict(lowercase=False, clean_url_html=True, hashtag_punct=False, teencode_emoji=False, repeat_chars=False, word_segmentation=False)
    },
    "C": {
        "name": "C: B + hashtag & dấu câu lặp",
        "params": dict(lowercase=False, clean_url_html=True, hashtag_punct=True, teencode_emoji=False, repeat_chars=False, word_segmentation=False)
    },
    "D": {
        "name": "D: C + teencode, emoji & lặp ký tự",
        "params": dict(lowercase=False, clean_url_html=True, hashtag_punct=True, teencode_emoji=True, repeat_chars=True, word_segmentation=False)
    },
    "E": {
        "name": "E: D + tách từ pyvi + lowercase (Chuẩn báo cáo)",
        "params": dict(lowercase=True, clean_url_html=True, hashtag_punct=True, teencode_emoji=True, repeat_chars=True, word_segmentation=True)
    },
    "F": {
        "name": "F: E nhưng KHÔNG lowercase",
        "params": dict(lowercase=False, clean_url_html=True, hashtag_punct=True, teencode_emoji=True, repeat_chars=True, word_segmentation=True)
    }
}

def run_ablation_experiment(max_samples: int = None, selected_variants: list = None):
    paths = get_paths()
    train_csv = os.path.join(paths.splits, "train.csv")
    test_csv = os.path.join(paths.splits, "test.csv")

    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {train_csv} hoặc {test_csv}.")
        sys.exit(1)

    train_df = pd.read_csv(train_csv).dropna(subset=['original_text', 'label'])
    test_df = pd.read_csv(test_csv).dropna(subset=['original_text', 'label'])

    if max_samples:
        train_df = train_df.head(max_samples)
        test_df = test_df.head(max_samples)

    if not selected_variants:
        selected_variants = list(VARIANTS_CONFIG.keys())

    print(f"🤖 Đang tải mô hình PhoBERT ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("⚠️ [CẢNH BÁO THỜI GIAN] Thí nghiệm Ablation trích xuất PhoBERT qua nhiều biến thể cấu hình có thể mất nhiều thời gian!")

    results = []
    variant_preds = {}
    y_test_arr = test_df['label'].astype(int).values

    sample_tag = str(max_samples) if max_samples is not None else "full"

    for var_key in selected_variants:
        if var_key not in VARIANTS_CONFIG:
            continue
        cfg = VARIANTS_CONFIG[var_key]
        print(f"\n--- Đang xử lý biến thể {var_key}: {cfg['name']} ---")

        cache_dir = os.path.join(paths.results_base, "ablation_cache", f"{var_key}_{sample_tag}")
        os.makedirs(cache_dir, exist_ok=True)
        tr_feat_file = os.path.join(cache_dir, "train_feat.npy")
        te_feat_file = os.path.join(cache_dir, "test_feat.npy")
        tr_lab_file = os.path.join(cache_dir, "train_lab.npy")
        te_lab_file = os.path.join(cache_dir, "test_lab.npy")

        if all(os.path.exists(f) for f in [tr_feat_file, te_feat_file, tr_lab_file, te_lab_file]):
            print(f"   📂 Đã tìm thấy cache cho biến thể {var_key} ({sample_tag}), đang nạp từ đĩa...")
            X_train = np.load(tr_feat_file)
            X_test = np.load(te_feat_file)
            y_train = np.load(tr_lab_file)
            y_test = np.load(te_lab_file)
            ext_time = 0.0
        else:
            preprocessor = TextPreprocessor(**cfg["params"])
            
            print(f"   🧹 Tiền xử lý tập Train ({len(train_df)} mẫu)...")
            train_texts = [preprocessor.clean_text(t) for t in train_df['original_text'].astype(str).values]
            y_train = train_df['label'].astype(int).values

            print(f"   🧹 Tiền xử lý tập Test ({len(test_df)} mẫu)...")
            test_texts = [preprocessor.clean_text(t) for t in test_df['original_text'].astype(str).values]
            y_test = test_df['label'].astype(int).values

            print(f"   🔄 Trích xuất [CLS] embeddings từ PhoBERT cho biến thể {var_key}...")
            start_ext = time.time()
            
            batch_start = time.time()
            
            train_embeds = []
            for i in range(0, len(train_texts), BATCH_SIZE):
                b_txts = train_texts[i:i+BATCH_SIZE]
                encoded = tokenizer(b_txts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
                with torch.no_grad():
                    train_embeds.append(model(**encoded).last_hidden_state[:, 0, :].cpu().numpy())
                if i == 0 and len(train_texts) > BATCH_SIZE:
                    batch_time = time.time() - batch_start
                    est_total = batch_time * (len(train_texts) / BATCH_SIZE) * 2
                    print(f"   ⏱️ [Ước lượng] Lô đầu mất {batch_time:.2f}s. Tổng thời gian ước tính cho biến thể này: ~{est_total:.1f}s")

            X_train = np.concatenate(train_embeds, axis=0)

            test_embeds = []
            for i in range(0, len(test_texts), BATCH_SIZE):
                b_txts = test_texts[i:i+BATCH_SIZE]
                encoded = tokenizer(b_txts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
                with torch.no_grad():
                    test_embeds.append(model(**encoded).last_hidden_state[:, 0, :].cpu().numpy())
            X_test = np.concatenate(test_embeds, axis=0)

            ext_time = time.time() - start_ext

            np.save(tr_feat_file, X_train)
            np.save(te_feat_file, X_test)
            np.save(tr_lab_file, y_train)
            np.save(te_lab_file, y_test)

        print(f"   ⚖️ Huấn luyện và đánh giá SVM cho biến thể {var_key}...")
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_te_scaled = scaler.transform(X_test)

        svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        svm.fit(X_tr_scaled, y_train)

        y_pred = svm.predict(X_te_scaled)
        y_score = svm.predict_proba(X_te_scaled)[:, 1]

        variant_preds[var_key] = y_pred

        metrics = compute_metrics(y_test, y_pred, y_score)
        
        results.append({
            "variant": var_key,
            "description": cfg["name"],
            "accuracy": metrics["accuracy"],
            "precision_fake": metrics["fake_precision"],
            "recall_fake": metrics["fake_recall"],
            "f1_fake": metrics["fake_f1"],
            "auc_roc": metrics["auc_roc"],
            "f1_macro": metrics["macro_f1"],
            "extraction_time_sec": round(ext_time, 2)
        })

    res_base = paths.results_base
    os.makedirs(os.path.join(res_base, "tables"), exist_ok=True)
    os.makedirs(os.path.join(res_base, "figures"), exist_ok=True)

    df_ablation = pd.DataFrame(results)
    ablation_csv_path = os.path.join(res_base, "tables/ablation_preprocessing.csv")
    df_ablation.to_csv(ablation_csv_path, index=False)
    print(f"\n💾 Đã xuất bảng kết quả ablation tại {ablation_csv_path}")

    mcnemar_rows = []
    if "E" in variant_preds:
        base_pred = variant_preds["E"]
        for var_key in selected_variants:
            if var_key == "E" or var_key not in variant_preds:
                continue
            stat, pval = mcnemar_test(y_test_arr, base_pred, variant_preds[var_key])
            mcnemar_rows.append({
                "comparison": f"Variant E vs Variant {var_key}",
                "mcnemar_statistic": round(stat, 4),
                "p_value": pval,
                "significant_at_0.05": pval < 0.05
            })
    if mcnemar_rows:
        df_mcnemar = pd.DataFrame(mcnemar_rows)
        mcnemar_csv_path = os.path.join(res_base, "tables/mcnemar.csv")
        df_mcnemar.to_csv(mcnemar_csv_path, index=False)
        print(f"💾 Đã xuất kết quả kiểm định McNemar tại {mcnemar_csv_path}")

    print("📈 Đang vẽ biểu đồ ablation...")
    f1_vals = []
    ci_lowers = []
    ci_uppers = []
    labels = []

    for idx, row in df_ablation.iterrows():
        var_k = row["variant"]
        yp = variant_preds[var_k]
        def f1_func(yt, yp_):
            from sklearn.metrics import precision_recall_fscore_support
            _, _, f_arr, _ = precision_recall_fscore_support(y_test_arr, yp_, labels=[0, 1], zero_division=0)
            return f_arr[1] if len(f_arr) > 1 else 0.0

        low, up = bootstrap_ci(y_test_arr, yp, f1_func, n=500, seed=42)
        f1_vals.append(row["f1_fake"])
        ci_lowers.append(max(0.0, row["f1_fake"] - low))
        ci_uppers.append(max(0.0, up - row["f1_fake"]))
        labels.append(f"Var {var_k}")

    plt.figure(figsize=(9, 6))
    x_pos = np.arange(len(labels))
    errors = [ci_lowers, ci_uppers]
    plt.bar(x_pos, f1_vals, yerr=errors, align='center', alpha=0.8, ecolor='black', capsize=5, color='skyblue')
    plt.xticks(x_pos, labels, fontweight='bold')
    plt.ylabel('F1-score (Lớp Tin giả)', fontsize=11)
    plt.title('Nghiên cứu Ablation Pipeline Tiền Xử Lý (Giả thuyết H2)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    ablation_fig_path = os.path.join(res_base, "figures/ablation_preprocessing.png")
    plt.savefig(ablation_fig_path, dpi=200)
    plt.close()
    print(f"📊 Đã lưu biểu đồ ablation tại {ablation_fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thí nghiệm Ablation Study cho giả thuyết H2.")
    parser.add_argument("--max-samples", type=int, default=None, help="Chạy thử nhanh với N mẫu.")
    parser.add_argument("--variants", type=str, default=None, help="Danh sách biến thể cách nhau bởi dấu phẩy (vd: A,B,C,D,E,F).")
    args = parser.parse_args()

    variants_list = args.variants.split(",") if args.variants else None
    run_ablation_experiment(max_samples=args.max_samples, selected_variants=variants_list)
