import os
import sys
import subprocess
from datetime import datetime

def main():
    os.makedirs("results/logs", exist_ok=True)
    log_file_path = "results/logs/run_all.log"
    
    print(f"🚀 Bắt đầu chạy quy trình thực nghiệm. Log lưu tại {log_file_path}")
    
    steps = [
        ("Phân chia dữ liệu", ["src.preprocessing.split_data"]),
        ("Trích xuất đặc trưng", ["src.features.extract_phobert"]),
        ("Huấn luyện SVM", ["src.models.train_svm"]),
        ("Đánh giá mô hình", ["src.evaluation.evaluate"])
    ]

    with open(log_file_path, "w", encoding="utf-8") as log_f:
        for desc, cmd in steps:
            print(f"\n--- {desc} ---")
            log_f.write(f"\n[{datetime.now()}] --- {desc} ---\n")
            log_f.flush()
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m"] + cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8"
                )
                print(result.stdout)
                log_f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"❌ Lỗi tại bước: {desc}")
                print(e.stdout)
                log_f.write(e.stdout)
                sys.exit(1)

    print("\n✅ Hoàn thành toàn bộ quy trình thực nghiệm thành công!")

if __name__ == "__main__":
    main()
