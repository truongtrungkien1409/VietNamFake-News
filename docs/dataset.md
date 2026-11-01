# Hướng Dẫn Chuẩn Bị Dữ Liệu (Dataset Preparation)

Tài liệu này hướng dẫn cách chuẩn hóa và định dạng tập dữ liệu đầu vào cho hệ thống nhận diện tin giả tiếng Việt (`DACN.pdf`).

---

## 1. Cấu Trúc Tập Dữ Liệu Thô (Raw Dataset)
Tập dữ liệu thô yêu cầu định dạng tệp **CSV** (đặt tại `data/raw/vn_news_merged.csv`) và phải bao gồm tối thiểu hai cột bắt buộc:
1. **`text`** (hoặc `original_text`): Nội dung văn bản bài báo / bài viết mạng xã hội (kiểu chuỗi, độ dài tối thiểu 20 ký tự).
2. **`label`**: Nhãn phân loại nhị phân:
   - `0`: Tin thật (Real News) - Lớp âm tính.
   - `1`: Tin giả (Fake News) - Lớp dương tính.

---

## 2. Quy Trình Tiền Xử Lý & Phân Chia (80% / 10% / 10%)
Hệ thống sử dụng chiến lược **Lấy mẫu phân tầng (Stratified Sampling)** nhằm đảm bảo tỷ lệ cân bằng giữa tin thật và tin giả trên cả ba tập:
- **Tập Huấn Luyện (Train - 80%):** Dùng để trích xuất đặc trưng và huấn luyện mô hình phân loại SVM.
- **Tập Xác Thực (Validation - 10%):** Dùng để khảo sát, tinh chỉnh siêu tham số (Grid Search kernel & $C$).
- **Tập Kiểm Thử (Test - 10%):** Dùng để đánh giá độc lập năng lực tổng quát hóa của mô hình.

### Lệnh thực hiện phân chia:
```bash
python -m src.preprocessing.split_data data/raw/vn_news_merged.csv
```
Quá trình này sẽ tự động áp dụng quy trình tiền xử lý 5 bước chuẩn, loại bỏ các bản ghi xung đột nhãn / trùng lặp, và lưu các tệp kết quả vào thư mục `data/splits/`.
