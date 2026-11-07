# Checklist sửa Chương 1 và Chương 2

Tài liệu này là danh sách các vị trí cần đối chiếu và sửa trong [docs/Do_an/DACN.pdf](docs/Do_an/DACN.pdf) sau khi có kết quả thực nghiệm. Đây không phải là nội dung thay thế trực tiếp cho báo cáo PDF.

## 1. Các quy ước phải thống nhất trong Chương 1

### 1.1. Mục 1.2.2.1 — biểu diễn bài toán phân loại nhị phân

**Cần sửa:** rà soát mọi câu mô tả mã nhãn và sửa về cùng một quy ước với mã nguồn:

| Mã | Ý nghĩa đúng |
|---:|---|
| 0 | Tin thật |
| 1 | Tin giả |

Không được để lại câu diễn giải ngược như “1 = tin thật”. Quy ước này phải khớp với các cột `label`, `true_label`, `predicted_label` trong các file kết quả.

### 1.2. Mục 1.3.1.1–1.3.1.2 — nguồn dữ liệu

**Cần sửa:** mô tả dữ liệu thực nghiệm là sự kết hợp của:

- ReINTEL: các bài đăng mạng xã hội được sử dụng trong dữ liệu đầu vào;
- VFND: 254 bài thuộc các nhóm nguồn báo và Facebook trong bản dữ liệu đã tải;
- dữ liệu đầu vào sau làm sạch: 4.505 mẫu, theo các file chia tập trong `data/splits/`.

Không ghi rằng bộ thực nghiệm có 10.000 mẫu nếu không kèm chú thích đó là con số thiết kế cũ. Không gọi dữ liệu hiện tại là toàn bộ dữ liệu cào tự động khi báo cáo không có bằng chứng tương ứng.

### 1.3. Mục 1.3.4.2 — tiêu chí đánh giá

**Cần sửa:** bổ sung hoặc điều chỉnh cách nhấn mạnh tiêu chí đánh giá do dữ liệu mất cân bằng. Chỉ số trọng tâm là F1 và Recall của lớp tin giả; Accuracy chỉ dùng để tham khảo. Kết quả test hiện tại có 83 mẫu tin giả, nên khoảng tin cậy Bootstrap của F1 lớp này khá rộng.

Không dùng các cụm khẳng định tuyệt đối như “vượt trội rõ rệt” nếu chưa đi kèm khoảng tin cậy và kiểm định McNemar.

## 2. Các nội dung cần sửa trong Chương 2

### 2.1. Mục 2.2.1.3 — thống kê dữ liệu và phân bố nhãn

**Cần thay toàn bộ mô tả cũ** đang ghi khoảng 10.000 mẫu, 7.000 tin thật và 3.000 tin giả. Số đếm từ hai file nhãn là:

| Nguồn file | Nhãn 0 — tin thật | Nhãn 1 — tin giả | Tổng |
|---|---:|---:|---:|
| ReINTEL | 3.562 | 721 | 4.283 |
| VFND | 124 | 105 | 229 |
| Tổng trước làm sạch | 3.686 | 826 | 4.512 |

Sau bước làm sạch và loại bản ghi không còn hợp lệ, các file chia tập có tổng 4.505 mẫu. Tỷ lệ nhãn thực tế xấp xỉ 82% tin thật và 18% tin giả, không phải tỷ lệ 70%/30%.

**Nguồn kiểm tra:** `data/raw/vn_news_label_0_real.csv`, `data/raw/vn_news_label_1_fake.csv`, `data/splits/train.csv`, `val.csv`, `test.csv`.

### 2.2. Bảng 2.2 — phân chia Train/Validation/Test

**Cần thay toàn bộ số liệu cũ** 8.000/1.000/1.000 bằng số liệu đã chạy:

| Tập | Tỷ lệ | Tổng | Nhãn 0 | Nhãn 1 | Mục đích |
|---|---:|---:|---:|---:|---|
| Train | 80% | 3.604 | 2.944 | 660 | Trích xuất đặc trưng và huấn luyện |
| Validation | 10% | 450 | 368 | 82 | Chọn cấu hình SVM |
| Test | 10% | 451 | 368 | 83 | Đánh giá độc lập |
| Tổng | 100% | 4.505 | 3.680 | 825 | Toàn bộ dữ liệu sau làm sạch |

Giữ mô tả “stratified split” và kiểm tra lại mọi hình hoặc bảng khác có nhắc đến kích thước các tập.

### 2.3. Các đoạn trong mục 2.2.2–2.2.3 về tiền xử lý và dữ liệu

**Cần sửa:**

1. Ghi đúng thư viện tách từ đã thực thi là `pyvi` (ViTokenizer), không ghi RDRSegmenter/VnCoreNLP như công cụ đã chạy.
2. Giữ `max_length = 256` token nếu đoạn đó mô tả cấu hình thực thi.
3. Giữ tỷ lệ 80%/10%/10%, nhưng thay toàn bộ số lượng mẫu theo Bảng 2.2 mới.
4. Nêu rõ việc loại bản ghi trùng lặp/rỗng được thực hiện trước khi chia tập để tránh rò rỉ dữ liệu.
5. Nếu có ví dụ tiền xử lý dùng RDRSegmenter, thay ví dụ bằng đầu ra do `pyvi` tạo hoặc đánh dấu đó chỉ là mô tả phương án ban đầu.

### 2.4. Mục 2.4.2.1 — so sánh Linear và RBF

**Cần sửa cách kết luận:** mục này hiện mô tả Linear như lựa chọn tối ưu và có thể gắn với nhận định phân tách tuyến tính rõ rệt. Kết quả tuning trên Validation cho thấy cấu hình cuối được chọn là:

- kernel `rbf`;
- `C = 10.0`;
- `gamma = scale`;
- `class_weight = None`;
- seed `42`.

Việc chọn cấu hình phải được mô tả là dựa trên F1 lớp tin giả của Validation, không dựa trên Test. Bảng tuning có 12 cấu hình; cấu hình RBF, `C = 10.0`, `class_weight = None` đạt F1 tin giả Validation `0,7308`, trong khi cấu hình Linear, `C = 1.0`, `class_weight = None` đạt `0,6391`.

### 2.5. Mục 2.4.2.2 — nhận định về tính phân tách tuyến tính

**Cần viết lại các nhận định khẳng định Linear tương đương hoặc vượt RBF.** Kết quả Test của SVM Linear mặc định (`results_linear_default`) có F1 tin giả `0,4819`, còn cấu hình RBF cuối có F1 tin giả `0,6383`. Vì vậy chỉ nên mô tả Linear là cấu hình đối chứng, không kết luận Linear là lựa chọn tối ưu hoặc loại bỏ nguy cơ quá khớp.

### 2.6. Mục 2.4.2.3 — tinh chỉnh C

**Cần sửa:** ghi rõ Grid Search được đánh giá trên Validation. Giá trị `C = 10.0` là cấu hình cuối trong kết quả hiện tại; không được thay bằng `C = 1.0` nếu đang nói về mô hình PhoBERT-SVM dùng cho kết quả chính.

`C = 1.0`, kernel Linear chỉ được giữ ở vai trò cấu hình mặc định/đối chứng và được dùng riêng trong thí nghiệm ablation.

### 2.7. Mục 2.6.2 và Bảng 2.8 — thông số kỹ thuật cốt lõi

**Cần sửa Bảng 2.8 và mọi đoạn dẫn lại bảng này:**

| Thành phần | Giá trị cần cập nhật |
|---|---|
| Tổng dữ liệu sau làm sạch | 4.505 mẫu |
| Phân bố nhãn | 3.680 tin thật (0), 825 tin giả (1) |
| Chia tập | 3.604 Train / 450 Validation / 451 Test |
| Tách từ | `pyvi` |
| Mô hình đặc trưng | `vinai/phobert-base-v2` |
| Độ dài tối đa | 256 token |
| Chuẩn hóa | `StandardScaler` |
| SVM cuối | RBF, `C = 10.0`, `gamma = scale`, `probability=True` |
| Seed | 42 |

Không để đồng thời hai bộ thông số 10.000 mẫu/70-30 và 4.505 mẫu/82-18 trong các bảng mô tả cùng một thực nghiệm.

### 2.8. Các hình và chú thích ở Chương 2

Rà soát các hình mô tả pipeline, tách từ và cấu hình SVM. Nếu chú thích ghi RDRSegmenter, Linear là cấu hình cuối, hoặc ghi 8.000/1.000/1.000, phải sửa chú thích và đoạn giải thích kèm hình cho khớp mã nguồn.

Đặc biệt, Hình 2.7 nếu minh họa SVC cần ghi đây là minh họa API/thuật toán, không dùng chú thích đó để khẳng định kernel Linear là cấu hình cuối.

## 3. Kết quả cần dùng khi sửa phần thực nghiệm được dẫn từ Chương 2

Các giá trị dưới đây là mốc đối chiếu, lấy từ `results/tables/evaluation_metrics.csv`:

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0,8869 |
| Precision tin giả | 0,7759 |
| Recall tin giả | 0,5422 |
| F1 tin giả | 0,6383 |
| AUC-ROC | 0,9228 |
| Bootstrap CI 95% của F1 tin giả | [0,5333; 0,7260] |
| TN / FP / FN / TP | 355 / 13 / 38 / 45 |

Khi sửa các giả thuyết, không viết rằng mô hình luôn phát hiện tốt tin giả: Recall tin giả là khoảng 54%, nghĩa là còn 38 mẫu tin giả bị bỏ sót trong Test.

## 4. Các nội dung ablation cần sửa nếu đã nêu trong Chương 2

Thí nghiệm ablation dùng SVM Linear với `C = 1.0` cố định và chạy trên tập con giới hạn bởi `--max-samples 1500`, không phải quy trình chọn mô hình cuối. Không dùng cột `extraction_time_sec` để kết luận về hiệu năng.

| Biến thể | F1 tin giả |
|---|---:|
| A — không tiền xử lý | 0,552 |
| B — làm sạch URL/HTML | 0,548 |
| C — thêm hashtag và dấu câu lặp | 0,497 |
| D — thêm teencode, emoji và lặp ký tự | 0,519 |
| E — thêm `pyvi` và lowercase | 0,550 |
| F — không lowercase | 0,623 |

Kết luận H2 phải là **chưa có đủ bằng chứng ủng hộ**. So sánh E với A có `p = 0,5504`; E với F có `p = 0,3211`. F chỉ là xu hướng quan sát được, không phải cải thiện có ý nghĩa thống kê. Không viết rằng tiền xử lý chắc chắn cải thiện hiệu năng.

## 5. Các điểm cần sửa trong diễn giải giả thuyết và kết luận

- H1: chỉ giữ kết luận nếu đoạn văn có số liệu đánh giá mô hình PhoBERT-SVM làm căn cứ.
- H2: sửa thành chưa được ủng hộ bởi kết quả ablation và McNemar.
- H3: khi so sánh baseline, dẫn `results/tables/mcnemar_baselines.csv`; bốn p-value là 0,0003; 0,0018; 0,0006 và 0,0038. Sau Bonferroni, tất cả vẫn nhỏ hơn ngưỡng 0,0125. Mức độ chênh lệch cụ thể phải dẫn từ `model_comparison.csv`, không dùng cụm “vượt trội rõ rệt”.

## 6. Checklist rà soát toàn văn

- [ ] Tìm và sửa mọi biến thể của `10.000`, `10000`, `8.000`, `1.000`, `7.000`, `3.000`.
- [ ] Tìm và sửa mọi mô tả tỷ lệ nhãn `70/30`.
- [ ] Tìm `RDRSegmenter` và thay bằng `pyvi` ở nơi mô tả công cụ đã dùng.
- [ ] Tìm các câu ghi `1 = tin thật`; sửa thành `0 = tin thật`, `1 = tin giả`.
- [ ] Tìm kernel Linear được gọi là cấu hình cuối; sửa thành RBF, `C = 10.0` khi nói về mô hình chính.
- [ ] Cập nhật Bảng 2.2 và Bảng 2.8.
- [ ] Kiểm tra các chú thích hình có số mẫu, nhãn và công cụ tách từ.
- [ ] Loại các nhận định “tiền xử lý cải thiện hiệu năng” nếu không có kiểm định thống kê hỗ trợ.
- [ ] Bổ sung trích dẫn ReINTEL và VFND ở tài liệu tham khảo.
- [ ] Xin phép tác giả VFND theo thông tin liên hệ đã cung cấp: `thanh.hoquang@pm.me`.

## 7. Nguồn đối chiếu

- Báo cáo nền: `docs/Do_an/DACN.pdf`.
- Dữ liệu nhãn: `data/raw/vn_news_label_0_real.csv`, `data/raw/vn_news_label_1_fake.csv`.
- Các tập đã chia: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.
- Kết quả mô hình: `results/tables/evaluation_metrics.csv`, `svm_tuning.csv`, `svm_final_config.json`, `ablation_preprocessing.csv`, `mcnemar.csv`, `mcnemar_baselines.csv`.
- Mã tiền xử lý và chia tập: `src/preprocessing/`, `scripts/merge_datasets.py`, `scripts/build_input.py`.

## 8. File đã tạo/sửa

- Đã sửa: `SuaNoiDung.md`.
- Chưa sửa trực tiếp: `docs/Do_an/DACN.pdf`; người viết cần áp dụng checklist này vào bản Word/LaTeX nguồn rồi xuất lại PDF.
