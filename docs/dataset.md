# Hướng Dẫn Chuẩn Bị & Tài Liệu Dữ Liệu (Dataset Documentation)

Tài liệu này mô tả nguồn gốc, quy tắc gộp, số mẫu, tỷ lệ nhãn và quy trình xử lý tập dữ liệu nhận diện tin giả tiếng Việt (`DACN.pdf`).

---

## 1. Nguồn Dữ Liệu & Quy Tắc Gộp (Dataset Source & Merge Rules)
- **Nguồn dữ liệu:** TODO (Nguồn thật sẽ được điền sau).
- **Quy tắc gộp (Merge Rules):**
  1. Các tệp dữ liệu thô từ các nguồn được gộp vào `data/raw/vn_news_merged.csv`.
  2. Xử lý chuẩn hóa văn bản và lọc bỏ các mẫu trùng lặp hoặc xung đột nhãn.
  3. Dữ liệu sạch sau tiền xử lý được lưu tại `data/processed/clean.csv`.

---

## 2. Thống Kê Mẫu Dữ Liệu Thực Tế
Để xem số mẫu và tỷ lệ nhãn thực tế đọc từ `data/processed/clean.csv`, chạy lệnh thống kê:
```bash
python -c "import pandas as pd; df = pd.read_csv('data/processed/clean.csv'); print(f'Total: {len(df)}, Real (0): {(df[\"label\"]==0).sum()}, Fake (1): {(df[\"label\"]==1).sum()}')"
```
- **Tổng số mẫu sạch:** TODO
- **Số lượng tin thật (Label 0):** TODO
- **Số lượng tin giả (Label 1):** TODO

---

## 3. Phân Chia Tập Dữ Liệu (Train / Validation / Test - 80% / 10% / 10%)
Hệ thống sử dụng phương pháp **Lấy mẫu phân tầng (Stratified Sampling)**:
- **Tập Huấn Luyện (Train - 80%):** TODO
- **Tập Xác Thực (Validation - 10%):** TODO
- **Tập Kiểm Thử (Test - 10%):** TODO

### Lệnh thực thi xây dựng dữ liệu:
```bash
python -m scripts.build_dataset
```
