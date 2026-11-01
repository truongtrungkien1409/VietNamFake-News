import os
import sys
import time
import argparse
import subprocess
import tempfile
from datetime import datetime
import pandas as pd

def generate_temp_dummy_data() -> str:
    """Tự động sinh file dữ liệu thô giả lập tạm thời trong thư mục temp để kiểm tra nhanh."""
    fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="vn_news_quick_")
    os.close(fd)
    
    texts = []
    labels = []
    topics = ["thời tiết", "kinh tế", "chính trị", "thể thao", "giáo dục", "y tế", "khoa học", "xã hội", "văn hóa", "công nghệ"]
    for i in range(120):
        top = topics[i % len(topics)]
        # Đảm bảo mỗi câu có đủ số từ (>=3 từ) và không trùng lặp để tránh bị bộ lọc tiền xử lý loại bỏ
        texts.append(f"Bản tin số {i} phản ánh tình hình phát triển của ngành {top} tại Việt Nam trong năm qua với nhiều điểm mới đáng chú ý.")
        labels.append(0 if i % 2 == 0 else 1)
        
    df_dummy = pd.DataFrame({"text": texts, "label": labels})
    df_dummy.to_csv(temp_path, index=False, encoding="utf-8")
    print(f"🧪 Đã tạo file dữ liệu giả lập tạm tại: {temp_path} ({len(df_dummy)} mẫu)")
    return temp_path

def main():
    parser = argparse.ArgumentParser(description="Chạy toàn bộ quy trình thực nghiệm nhận diện tin giả.")
    parser.add_argument("--skip", type=str, default="", help="Bỏ qua các bước (cách nhau bởi dấu phẩy).")
    parser.add_argument("--only", type=str, default="", help="Chỉ chạy bước được chỉ định.")
    parser.add_argument("--quick", action="store_true", help="Chạy nhanh với số mẫu nhỏ và ít epoch.")
    parser.add_argument("--data", type=str, default="", help="Đường dẫn file CSV dữ liệu thô.")
    args = parser.parse_args()

    temp_data_file = None
    if args.quick:
        res_dir = "results_quick"
        splits_dir = "data/splits_quick"
        os.environ["RESULTS_BASE"] = res_dir
        os.environ["SPLITS_BASE"] = splits_dir
        if not args.data:
            temp_data_file = generate_temp_dummy_data()
            args.data = temp_data_file
    else:
        res_dir = "results"
        splits_dir = "data/splits"
        os.environ["RESULTS_BASE"] = res_dir
        os.environ["SPLITS_BASE"] = splits_dir
        if not args.data:
            args.data = "data/raw/vn_news_merged.csv"

    os.makedirs(os.path.join(res_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(res_dir, "logs"), exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    if not os.path.exists(args.data):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu thô tại {args.data}. Vui lòng kiểm tra lại đường dẫn.")
        sys.exit(1)

    log_file_path = os.path.join(res_dir, "logs", "run_all.log")
    timing_file_path = os.path.join(res_dir, "tables", "run_timing.csv")

    print(f"🚀 Bắt đầu chạy quy trình thực nghiệm ({'QUICK MODE' if args.quick else 'FULL MODE'}). Log lưu tại {log_file_path}")

    max_samp_arg = ["--max-samples", "30"] if args.quick else []

    steps = [
        ("split_data", "Phân chia dữ liệu", ["src.preprocessing.split_data", args.data]),
        ("extract_phobert", "Trích xuất đặc trưng PhoBERT", ["src.features.extract_phobert"]),
        ("train_svm", "Huấn luyện SVM & Tuning", ["src.models.train_svm"]),
        ("evaluate", "Đánh giá mô hình trên Test", ["src.evaluation.evaluate"]),
        ("baselines", "Chạy các mô hình Baseline", ["src.evaluation.compare_models"]),
        ("ablation", "Thí nghiệm Ablation Study", ["experiments.ablation_preprocessing"] + max_samp_arg)
    ]

    skip_list = [s.strip() for s in args.skip.split(",") if s.strip()]
    timings = []

    try:
        with open(log_file_path, "w", encoding="utf-8") as log_f:
            for step_id, desc, cmd in steps:
                if args.only and step_id != args.only:
                    continue
                if step_id in skip_list:
                    print(f"⏭️ [Bỏ qua] Bước: {desc} ({step_id})")
                    continue

                print(f"\n--- Bắt đầu: {desc} ({step_id}) ---")
                log_f.write(f"\n[{datetime.now()}] --- Bắt đầu: {desc} ({step_id}) ---\n")
                log_f.flush()

                start_step = time.time()
                process = subprocess.Popen(
                    [sys.executable, "-m"] + cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"}
                )

                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()

                process.wait()
                step_duration = time.time() - start_step

                if process.returncode != 0:
                    print(f"\n❌ [LỖI] Bước '{desc}' ({step_id}) thất bại với mã lỗi {process.returncode}!")
                    print(f"💡 Để chạy lại riêng bước này, sử dụng lệnh:\n   python -m {' '.join(cmd)}")
                    sys.exit(1)

                print(f"✅ Hoàn thành: {desc} trong {step_duration:.2f} giây.")
                timings.append({
                    "step_id": step_id,
                    "description": desc,
                    "duration_sec": round(step_duration, 2)
                })

        df_timing = pd.DataFrame(timings)
        df_timing.to_csv(timing_file_path, index=False)
        print(f"⏱️ Đã lưu thời gian thực thi tại {timing_file_path}")

        from experiments.make_results_index import generate_results_index
        index_path = os.path.join(res_dir, "RESULTS_INDEX.md")
        generate_results_index(output_path=index_path)

        print(f"\n🎉 Toàn bộ quy trình thực nghiệm đã hoàn thành thành công! Kết quả lưu tại thư mục: {res_dir}/")

    finally:
        if temp_data_file and os.path.exists(temp_data_file):
            try:
                os.remove(temp_data_file)
                print(f"🧹 Đã xóa file dữ liệu giả lập tạm: {temp_data_file}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
