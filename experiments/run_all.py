import os
import sys

def main():
    print("🚀 Bắt đầu chạy toàn bộ quy trình thực nghiệm theo DACN.pdf...")
    
    # 1. Phân chia dữ liệu
    print("\n--- BƯỚC 1: Phân chia tập dữ liệu (80/10/10) ---")
    os.system("python src/preprocessing/split_data.py")
    
    # 2. Trích xuất đặc trưng
    print("\n--- BƯỚC 2: Trích xuất đặc trưng [CLS] 768 chiều từ PhoBERT ---")
    os.system("python src/features/extract_phobert.py")
    
    # 3. Huấn luyện SVM
    print("\n--- BƯỚC 3: Huấn luyện mô hình SVM ---")
    os.system("python src/models/train_svm.py")
    
    # 4. Đánh giá
    print("\n--- BƯỚC 4: Đánh giá mô hình trên tập Test ---")
    os.system("python src/evaluation/evaluate.py")
    
    print("\n✅ Hoàn thành toàn bộ quy trình thực nghiệm!")

if __name__ == "__main__":
    main()
