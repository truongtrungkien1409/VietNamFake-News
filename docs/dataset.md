# Hướng Dẫn Chuẩn Bị & Tài Liệu Dữ Liệu (Dataset Documentation)

Tài liệu này mô tả nguồn gốc, quy tắc gộp, số mẫu, tỷ lệ nhãn và quy trình xử lý tập dữ liệu nhận diện tin giả tiếng Việt (`DACN.pdf`).

---

## 1. Nguồn Dữ Liệu & Quy Tắc Gộp (Dataset Source & Merge Rules)
- **Nguồn dữ liệu:** Dữ liệu được tổng hợp từ các nguồn báo chí chính thống (Tin thật - label `0`) và các bài viết cảnh báo tin giả, mạng xã hội, trang thông tin không chính thống (Tin giả - label `1`).
- **Quy tắc gộp (Merge Rules):**
  1. Các tệp dữ liệu thô từ nhiều nguồn được gộp chung vào `data/raw/vn_news_merged.csv`.
  2. Bắt buộc có hai cột: `text` (hoặc `original_text`) và `label` (`0` cho Tin thật, `1` cho Tin giả).
  3. Lọc bỏ các dòng thiếu giá trị văn bản hoặc nhãn, các văn bản quá ngắn (< 20 ký tự hoặc < 3 từ sau khi tiền xử lý).
  4. Loại bỏ các văn bản trùng lặp nội dung hoặc có xung đột nhãn (cùng nội dung văn bản nhưng gán nhãn khác nhau giữa thật và giả).

---

## 2. Phân Chia Tập Dữ Liệu (Train / Validation / Test - 80% / 10% / 10%)
Hệ thống sử dụng phương pháp **Lấy mẫu phân tầng (Stratified Sampling)** để duy trì tỷ lệ cân bằng nhãn:
- **Tập Huấn Luyện (Train - 80%):** Dùng để trích xuất đặc trưng PhoBERT và huấn luyện mô hình phân loại SVM.
- **Tập Xác Thực (Validation - 10%):** Dùng để khảo sát, tinh chỉnh siêu tham số (Grid Search kernel & $C$).
- **Tập Kiểm Thử (Test - 10%):** Dùng để đánh giá độc lập mô hình cuối cùng.

### Lệnh thực thi xây dựng dữ liệu:
```bash
python -m scripts.build_dataset
```
Hoặc phân chia thủ công từ file thô:
```bash
python -m src.preprocessing.split_data data/raw/vn_news_merged.csv
```
Kết quả được lưu tại thư mục `data/splits/` (hoặc `data/splits_quick/` trong chế độ Quick Mode).
