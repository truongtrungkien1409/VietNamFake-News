# Nghiên Cứu Ablation Pipeline Tiền Xử Lý (Kiểm Chứng Giả Thuyết H2)

Tài liệu này mô tả chi tiết các biến thể cấu hình trong nghiên cứu ablation nhằm đánh giá ảnh hưởng của từng bước tiền xử lý văn bản đến chất lượng biểu diễn véc-tơ PhoBERT và hiệu năng phân loại của mô hình SVM.

---

## 1. Mô Tả Các Biến Thể Cấu Hình

- **Biến thể A (Không tiền xử lý):** Sử dụng văn bản gốc hoàn toàn nguyên bản (chưa qua bất kỳ bước làm sạch, chuẩn hóa hay tách từ nào).
- **Biến thể B (Chỉ URL/HTML cleanup):** Thực hiện chuẩn hóa Unicode NFC, loại bỏ các thẻ HTML tags, chuẩn hóa URL thành `HTTPURL` và các mentions thành `@USER`.
- **Biến thể C (B + hashtag & dấu câu lặp):** Kế thừa Biến thể B, đồng thời giữ lại nội dung bên trong hashtag (loại bỏ dấu `#`) và gộp/xóa các chuỗi dấu câu lặp (`!!`, `??`).
- **Biến thể D (C + teencode, emoji & lặp ký tự):** Kế thừa Biến thể C, bổ sung thêm việc ánh xạ teencode sang từ chuẩn, xử lý/xóa emoji và rút gọn các ký tự lặp trên chữ cái (ví dụ: `vuiiii` -> `vui`).
- **Biến thể E (D + tách từ pyvi + lowercase - Cấu hình chuẩn báo cáo):** Kế thừa Biến thể D, đồng thời áp dụng phân tách từ ghép tiếng Việt bằng `pyvi` (ViTokenizer) và chuyển toàn bộ văn bản về chữ thường (`lowercase`). Đây là cấu hình chuẩn được sử dụng trong báo cáo chính thức (`DACN.pdf`).
- **Biến thể F (E nhưng KHÔNG lowercase):** Kế thừa Biến thể E nhưng giữ nguyên phân biệt hoa thường (`lowercase=False`), nhằm kiểm tra tác động của việc giữ chữ hoa đối với mô hình PhoBERT vốn được tiền huấn luyện trên văn bản có phân biệt chữ hoa/thường.

---

## 2. Kết Quả Thực Nghiệm & Phân Tích
*(Phần này sẽ được tự động cập nhật số liệu vào `results/tables/ablation_preprocessing.csv` và biểu đồ `results/figures/ablation_preprocessing.png` sau khi chạy script `python -m experiments.ablation_preprocessing`).*

### 2.1. Bảng Tổng Hợp Chỉ Số
| Biến thể | Mô tả | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) | AUC-ROC | F1 Macro |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | Không tiền xử lý | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **B** | Chỉ URL/HTML cleanup | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **C** | B + hashtag & dấu câu lặp | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **D** | C + teencode, emoji & lặp ký tự | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **E** | D + tách từ pyvi + lowercase | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |
| **F** | E nhưng KHÔNG lowercase | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* | *TODO* |

### 2.2. Kiểm Định McNemar (So sánh Biến thể E với các biến thể khác)
*(Kết quả lưu tại `results/tables/mcnemar.csv`)*

---
*Lưu ý: Không đưa ra kết luận trước khi có số liệu thực nghiệm. Phần phân tích định tính và định lượng chi tiết sẽ được điền vào đây dựa trên kết quả chạy thực tế.*
