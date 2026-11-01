# Hướng Dẫn Triển Khai Hệ Thống (Deployment Guide)

Tài liệu này hướng dẫn chi tiết các bước khởi chạy mô hình nhận diện tin giả qua **Flask REST API** và cài đặt **Chrome Extension** trên trình duyệt để kiểm tra trực tiếp.

---

## 1. Khởi Chạy Flask REST API

1. **Chuẩn bị mô hình:** Đảm bảo bạn đã huấn luyện mô hình và các file trọng số đã được lưu tại `results/saved_models/`:
   - `svm_model.pkl`
   - `scaler.pkl`
2. **Khởi chạy API:**
   ```bash
   python -m app.app
   ```
   Hoặc sử dụng Makefile / PowerShell:
   ```bash
   make serve
   # Hoặc: powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command serve
   ```
3. API sẽ khởi chạy tại `http://127.0.0.1:5000`. Bạn có thể kiểm tra trạng thái hoạt động bằng cách truy cập:
   `http://127.0.0.1:5000/health` (Trả về `{"model_loaded": true, ...}`).

---

## 2. Cài Đặt Chrome Extension

1. Mở trình duyệt Google Chrome hoặc Microsoft Edge, truy cập trang quản lý tiện ích:
   `chrome://extensions/` (hoặc `edge://extensions/`).
2. Bật **Developer mode** (Chế độ nhà phát triển) ở góc trên bên phải.
3. Nhấn vào nút **Load unpacked** (Tải tiện ích đã giải nén) ở góc trên bên trái.
4. Trỏ tới thư mục `app/extension/` của dự án và nhấn **Select Folder**.
5. Tiện ích **Vietnamese Fake News Detector** sẽ xuất hiện trên thanh công cụ của trình duyệt.

---

## 3. Danh Sách Ảnh Chụp Màn Hình Cần Thiết Cho Chương 3

Để minh họa trong chương báo cáo đồ án (Mục 3.6 Triển khai hệ thống), bạn cần tự chụp các màn hình sau:
1. **Giao diện Popup khi phát hiện Tin Thật (Real News):** Hiển thị nhãn xanh, độ tin cậy và thanh tiến trình xác suất.
2. **Giao diện Popup khi phát hiện Tin Giả (Fake News):** Hiển thị cảnh báo đỏ (`⚠️ CẢNH BÁO TIN GIẢ!`) và xác suất chi tiết.
3. **Giao diện Popup khi chưa bật API:** Hiển thị thông báo lỗi thân thiện kèm hướng dẫn (`Hãy chạy: python -m app.app`).
4. **Tính năng "Lấy chữ chọn":** Màn hình popup tự động điền đoạn văn bản đang bôi đen trên một trang báo điện tử bất kỳ.

---

## 4. Hạn Chế Của Hệ Thống Triển Khai

1. **Chạy Cục Bộ (Localhost):** Hiện tại hệ thống được thiết kế chạy trên môi trường cục bộ (`127.0.0.1:5000`), chưa được đóng gói lên các nền tảng Cloud công khai (như Render, AWS, Hugging Face Spaces).
2. **Tài Nguyên Phần Cứng:** Quá trình suy luận sử dụng mô hình ngôn ngữ lớn PhoBERT yêu cầu bộ nhớ RAM / VRAM phù hợp (khuyến nghị có GPU hoặc CPU đa nhân để đạt độ trễ thấp).
3. **Tính Độc Lập:** Hệ thống chỉ đưa ra cảnh báo dựa trên phân tích văn bản ngôn ngữ tự nhiên, người dùng cần kết hợp xác minh từ các nguồn báo chí chính thống khác.
