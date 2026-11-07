# Hướng Dẫn Test

Tài liệu này hướng dẫn cách chạy kiểm thử cho dự án nhận diện tin giả tiếng Việt.

## 0)Test thủ công

python -m app.app

## 1) Chuẩn bị môi trường

### Cài thư viện

```bash
python -m pip install -r requirements.txt
```

### (Tùy chọn) Dùng PowerShell script có sẵn

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command install
```

## 2) Chạy toàn bộ test

Lệnh chuẩn:

```bash
python -m pytest tests -q
```

Hoặc:

```bash
make test
```

Hoặc PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run.ps1 -Command test
```

## 3) Chạy test theo nhóm (khi cần debug nhanh)

### Test metrics

```bash
python -m pytest tests/test_metrics.py -q
```

### Test chặn kết quả giả (không cho commit nhầm dữ liệu mô phỏng)

```bash
python -m pytest tests/test_no_fake_results.py -q
```

### Test train SVM

```bash
python -m pytest tests/test_train_svm.py -q
```

### Test baseline

```bash
python -m pytest tests/test_baselines.py -q
```

## 4) Giải thích test `test_no_fake_results.py`

File test này đọc `results/tables/evaluation_metrics.csv` (nếu tồn tại) và kiểm tra:

- `fake_support + real_support >= 100`

Mục đích là tránh commit nhầm kết quả từ dữ liệu giả lập quá nhỏ.

Lưu ý:

- Nếu file `results/tables/evaluation_metrics.csv` chưa tồn tại, test sẽ `skip` (bỏ qua), không fail.

## 5) Quy trình khuyến nghị trước khi commit

1. Chạy test nhanh:
   ```bash
   python -m pytest tests/test_metrics.py tests/test_no_fake_results.py -q
   ```
2. Chạy full test:
   ```bash
   python -m pytest tests -q
   ```
3. Chỉ commit khi tất cả test pass (hoặc có lý do rõ ràng cho test bị skip).

## 6) Một số lỗi thường gặp

- **Thiếu thư viện**: chạy lại `python -m pip install -r requirements.txt`.
- **Thiếu dữ liệu tách sẵn** (`data/splits/...`): chạy pipeline dữ liệu trong [Chuong3_thucnghiem.md](./Chuong3_thucnghiem.md).
- **Khác phiên bản Python/thư viện**: nên thống nhất môi trường trước khi so sánh kết quả thực nghiệm.
