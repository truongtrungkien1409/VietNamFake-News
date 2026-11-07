# CHƯƠNG 3. THỰC NGHIỆM VÀ XÂY DỰNG ỨNG DỤNG

Chương này trình bày thiết lập thực nghiệm, kết quả của mô hình PhoBERT-SVM, các phép đối sánh, nghiên cứu ablation và cách đóng gói mô hình thành dịch vụ Flask kết hợp tiện ích Chrome. Mục tiêu là kiểm tra khả năng nhận diện tin giả trên dữ liệu tiếng Việt trong điều kiện dữ liệu mất cân bằng, đồng thời xem xét hai giả thuyết H1 và H2 đã nêu ở các chương trước. H1 được đối chiếu bằng kết quả của mô hình PhoBERT-SVM trên tập kiểm tra độc lập; H2 được kiểm tra bằng các biến thể tiền xử lý và kiểm định McNemar. Các số liệu trong chương được lấy từ thư mục `results/`, `results_linear_default/` và các tập dữ liệu đã chia.

## 3.1. Môi trường thực nghiệm

### 3.1.1. Cấu hình phần cứng và phần mềm

Mục này xác định điều kiện chạy để kết quả có thể được tái lập. PhoBERT được dùng như bộ trích xuất đặc trưng cố định, sau đó véc-tơ 768 chiều được chuẩn hóa trước khi đưa vào SVM. Thông tin hệ thống được đọc từ `results/environment.json`.

**Bảng 3.1. Cấu hình phần cứng và phần mềm thực nghiệm**

| Thành phần | Giá trị |
|---|---|
| Hệ điều hành | Windows 10 |
| Python | 3.8.0 |
| PyTorch | 2.4.1+cpu |
| Transformers | 4.46.3 |
| scikit-learn | 1.3.2 |
| NumPy | 1.24.4 |
| pandas | 2.0.3 |
| CPU | AMD64 Family 23 Model 160 Stepping 0, 8 lõi |
| GPU | Không có; chạy CPU |

*Nguồn: `results/environment.json`.*

Các tham số pipeline được xác định từ mã nguồn và cấu hình kết quả như sau.

**Bảng 3.2. Tham số pipeline và mô hình**

| Tham số | Giá trị |
|---|---|
| Mô hình ngôn ngữ | `vinai/phobert-base-v2` |
| Độ dài chuỗi tối đa | 256 token |
| Kích thước batch khi trích xuất | 32 |
| Chuẩn hóa véc-tơ | `StandardScaler` |
| Kernel SVM cuối | RBF |
| `C` của SVM cuối | 10,0 |
| `gamma` | `scale` |
| `class_weight` | `None` |
| Seed | 42 |

*Nguồn: `src/features/extract_phobert.py`, `src/models/train_svm.py`, `app/app.py` và `results/tables/svm_final_config.json`.*

Thời gian trích xuất riêng cho ba tập Train, Validation và Test được ghi bởi
`src/features/extract_phobert.py` vào
`results/tables/phobert_extraction_timing.csv` sau khi chạy lệnh:
`python -m src.features.extract_phobert --force`.
Nếu file chưa có đủ ba dòng kết quả thực tế, chưa được điền số vào bảng này.

### 3.1.2. Dữ liệu thực nghiệm

Dữ liệu đầu vào kết hợp hai nguồn có đặc điểm khác nhau: ReINTEL chủ yếu là bài đăng mạng xã hội, còn VFND gồm các bài thuộc nhóm báo và Facebook. Hai file nhãn thô chứa lần lượt 3.686 mẫu nhãn 0 và 826 mẫu nhãn 1. Trong đó, phần VFND có 124 mẫu thật và 105 mẫu giả, tổng cộng 229 mẫu trong hai file nhãn; bộ VFND gốc được mô tả là 254 bài, nhưng các bản ghi có mặt trong dữ liệu đầu vào hiện tại được thống kê theo hai file CSV.

**Bảng 3.3. Phân bố dữ liệu theo nguồn và nhãn trong hai file đầu vào**

| Nguồn | Tin thật (0) | Tin giả (1) | Tổng |
|---|---:|---:|---:|
| ReINTEL | 3.562 | 721 | 4.283 |
| VFND | 124 | 105 | 229 |
| Tổng | 3.686 | 826 | 4.512 |

*Nguồn: đếm từ `data/raw/vn_news_label_0_real.csv` và `data/raw/vn_news_label_1_fake.csv`.*

Sau làm sạch và loại các bản ghi không còn hợp lệ, các file chia tập có tổng 4.505 mẫu. Việc chia được thực hiện phân tầng theo nhãn với tỷ lệ 80%/10%/10%.

**Bảng 3.4. Phân chia dữ liệu thực nghiệm**

| Tập | Tỷ lệ | Tổng số mẫu | Tin thật (0) | Tin giả (1) |
|---|---:|---:|---:|---:|
| Train | 80% | 3.604 | 2.944 | 660 |
| Validation | 10% | 450 | 368 | 82 |
| Test | 10% | 451 | 368 | 83 |
| Tổng | 100% | 4.505 | 3.680 | 825 |

*Nguồn: đếm từ `data/splits/train.csv`, `data/splits/val.csv` và `data/splits/test.csv`.*

Độ dài văn bản trong dữ liệu thô cũng khác theo nguồn. Với nhãn 0, ReINTEL có trung bình 726,97 ký tự và 157,65 từ, trong khi VFND có trung bình 2.389,44 ký tự và 526,43 từ. Với nhãn 1, các giá trị tương ứng là 873,84 ký tự và 194,26 từ đối với ReINTEL, còn VFND là 2.635,87 ký tự và 590,33 từ. Sự khác biệt này là một hạn chế về tính đồng nhất miền dữ liệu: mô hình phải xử lý cả văn bản mạng xã hội ngắn và bài viết dài hơn.

### 3.1.3. Tiền xử lý văn bản

Quy trình tiền xử lý được thực hiện trước khi chia tập nhằm hạn chế bản ghi trùng lặp xuất hiện ở nhiều tập. Các bước chính gồm chuẩn hóa văn bản, làm sạch URL/HTML, xử lý hashtag và dấu câu lặp, xử lý teencode/emoji/lặp ký tự, sau đó tách từ bằng `pyvi` và chuyển chữ thường ở cấu hình chuẩn. Các bản ghi rỗng hoặc trùng lặp được loại bỏ theo logic trong `src/preprocessing/preprocess.py`.

Ví dụ trước/sau được xuất từ dữ liệu thật bằng lệnh
`python -m experiments.export_preprocessing_examples --input data/raw/vn_news_label_0_real.csv data/raw/vn_news_label_1_fake.csv`.
Các mẫu được lưu tại `results/tables/preprocessing_examples.csv`; thống kê
lọc và trùng lặp được lưu tại `results/tables/preprocessing_summary.csv`.

**Bảng 3.5. Cấu trúc tệp ví dụ trước/sau tiền xử lý**

| Trường | Ý nghĩa |
|---|---|
| `id` | Mã mẫu trong tệp xuất |
| `label` | Nhãn gốc: 0 là tin thật, 1 là tin giả |
| `source` | Nguồn dữ liệu nếu cột này có trong CSV đầu vào |
| `original_text` | Văn bản trước tiền xử lý |
| `cleaned_text` | Văn bản sau cùng của pipeline tiền xử lý |
| `applied_transforms` | Các biến đổi quan sát được trên mẫu |

*Nguồn: `results/tables/preprocessing_examples.csv`. Tệp CSV chứa 10 mẫu thực tế; bảng này mô tả cấu trúc để giữ nguyên văn bản đầy đủ và không làm dài bản thảo.*

**Bảng 3.6. Tóm tắt định lượng bước lọc tiền xử lý**

| Chỉ số | Giá trị |
|---|---:|
| Mẫu đầu vào | 4.512 |
| Còn văn bản sau làm sạch rỗng | 4.510 |
| Loại do rỗng | 2 |
| Có ít nhất 3 từ | 4.509 |
| Loại do ngắn | 1 |
| Loại do xung đột nhãn | 2 |
| Số văn bản chuẩn hóa duy nhất sau lọc | 4.505 |
| Ứng viên trùng lặp sau lọc xung đột | 2 |

*Nguồn: `results/tables/preprocessing_summary.csv`.*

## 3.2. Kết quả thực nghiệm mô hình PhoBERT-SVM

### 3.2.1. Khảo sát siêu tham số SVM

Việc chọn cấu hình được thực hiện trên Validation, không dùng Test để chọn tham số. Bảng `svm_tuning.csv` ghi 12 cấu hình gồm hai kernel, ba giá trị `C` và hai lựa chọn `class_weight`. Cấu hình RBF với `C = 10,0`, `class_weight = None` đạt F1 lớp tin giả trên Validation là 0,7308 và được lưu trong `svm_final_config.json`.

**Bảng 3.7. Một số cấu hình SVM trong quá trình tuning trên Validation**

| Kernel | C | class_weight | Accuracy | Precision giả | Recall giả | F1 giả | F1 macro | AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Linear | 0,1 | None | 0,8711 | 0,6364 | 0,6829 | 0,6588 | 0,7897 | 0,8861 |
| Linear | 1,0 | None | 0,8644 | 0,6207 | 0,6585 | 0,6391 | 0,7778 | 0,8805 |
| Linear | 10,0 | None | 0,8467 | 0,5730 | 0,6220 | 0,5965 | 0,7509 | 0,8746 |
| RBF | 0,1 | None | 0,8244 | 0,8000 | 0,0488 | 0,0920 | 0,4974 | 0,9262 |
| RBF | 1,0 | None | 0,8956 | 0,7778 | 0,5976 | 0,6759 | 0,8068 | 0,9450 |
| RBF | 10,0 | None | 0,9067 | 0,7703 | 0,6951 | 0,7308 | 0,8372 | 0,9519 |

*Nguồn: `results/tables/svm_tuning.csv`.*

Hình nhiệt `svm_tuning_heatmap.png` cho phép quan sát sự thay đổi F1 theo kernel và `C`.

**Hình 3.1. Heatmap kết quả khảo sát siêu tham số SVM**

![Heatmap tuning SVM](results/figures/svm_tuning_heatmap.png)

*Nguồn: `results/figures/svm_tuning_heatmap.png`.*

Các cấu hình Linear có độ chính xác huấn luyện từ 0,9906 đến 0,9997 ở những dòng tương ứng trong file tuning, trong khi hiệu năng Validation không tăng theo độ chính xác huấn luyện. Đây là dấu hiệu cần thận trọng khi diễn giải khả năng tổng quát hóa của Linear, không phải bằng chứng để khẳng định một kernel luôn tốt hơn trong mọi điều kiện.

### 3.2.2. Quá trình huấn luyện và dự đoán

Mô hình cuối sử dụng RBF, `C = 10,0`, `gamma = scale` và có 1.826 support vector. Thời gian huấn luyện được ghi là 32,67 giây. Thời gian suy luận trung bình theo phép đo batch 100 mẫu là 0,7465669494 giây trên mỗi mẫu, trong đó PhoBERT chiếm 0,7430862761 giây và phần SVM chiếm 0,0034806734 giây.

**Bảng 3.8. Thời gian huấn luyện và suy luận**

| Đại lượng | Giá trị |
|---|---:|
| Thời gian huấn luyện SVM | 32,67 giây |
| Số support vector | 1.826 |
| Thời gian PhoBERT mỗi mẫu | 0,7430862761 giây |
| Thời gian SVM mỗi mẫu | 0,0034806734 giây |
| Tổng thời gian suy luận mỗi mẫu | 0,7465669494 giây |
| Phương pháp đo | `batch_100_samples` |

*Nguồn: `results/tables/svm_final_config.json` và `results/tables/inference_time.csv`.*

### 3.2.3. Kết quả đánh giá trên tập Test

Tập Test gồm 451 mẫu, trong đó có 83 mẫu tin giả. Do mất cân bằng lớp, F1 và Recall của lớp tin giả được ưu tiên khi diễn giải. Accuracy được báo cáo để tham khảo cùng với các chỉ số theo lớp, AUC và Average Precision.

**Bảng 3.9. Kết quả đánh giá định lượng trên tập Test**

| Chỉ số | Tin thật (0) | Tin giả (1) | Macro | Weighted |
|---|---:|---:|---:|---:|
| Precision | 0,9033 | 0,7759 | 0,8396 | 0,8799 |
| Recall | 0,9647 | 0,5422 | 0,7534 | 0,8869 |
| F1 | 0,9330 | 0,6383 | 0,7856 | 0,8788 |
| Support | 368 | 83 | 451 | 451 |

| Chỉ số tổng hợp | Giá trị |
|---|---:|
| Accuracy | 0,8869 |
| AUC-ROC | 0,9228 |
| Average Precision | 0,7465 |
| Bootstrap CI 95% của F1 tin giả | [0,5333; 0,7260] |

*Nguồn: `results/tables/evaluation_metrics.csv`.*

Khoảng tin cậy của F1 lớp tin giả tương đối rộng do lớp này chỉ có 83 mẫu trong Test. Vì vậy, kết quả cần được hiểu là ước lượng trên tập kiểm tra cụ thể, không phải khẳng định chắc chắn cho mọi miền dữ liệu.

**Bảng 3.10. Ma trận nhầm lẫn dạng số**

| | Dự đoán tin thật (0) | Dự đoán tin giả (1) |
|---|---:|---:|
| Thực tế tin thật (0) | TN = 355 | FP = 13 |
| Thực tế tin giả (1) | FN = 38 | TP = 45 |

*Nguồn: `results/tables/evaluation_metrics.csv`.*

**Hình 3.2. Ma trận nhầm lẫn trên tập Test**

![Confusion matrix](results/figures/confusion_matrix.png)

*Nguồn: `results/figures/confusion_matrix.png`.*

**Hình 3.3. Đường cong ROC**

![ROC curve](results/figures/roc_curve.png)

*Nguồn: `results/figures/roc_curve.png`.*

**Hình 3.4. Đường cong Precision–Recall**

![PR curve](results/figures/pr_curve.png)

*Nguồn: `results/figures/pr_curve.png`.*

### 3.2.4. So sánh kernel trên Test

Linear `C = 1,0` là cấu hình mặc định được lưu riêng trong `results_linear_default`, còn RBF `C = 10,0` là cấu hình được chọn theo F1 tin giả trên Validation. Khi đánh giá Test, Linear đạt F1 tin giả 0,4819 và Recall 0,4819; RBF đạt F1 tin giả 0,6383 và Recall 0,5422.

**Bảng 3.11. Đối chiếu SVM Linear mặc định và SVM RBF cuối**

| Cấu hình | Accuracy | Precision giả | Recall giả | F1 giả | AUC-ROC |
|---|---:|---:|---:|---:|---:|
| Linear, C = 1,0 | 0,8093 | 0,4819 | 0,4819 | 0,4819 | 0,8382 |
| RBF, C = 10,0 | 0,8869 | 0,7759 | 0,5422 | 0,6383 | 0,9228 |

*Nguồn: `results_linear_default/tables/evaluation_metrics.csv` và `results/tables/evaluation_metrics.csv`.*

## 3.3. So sánh với các mô hình khác

Mục này đối chiếu mô hình đề xuất với Naive Bayes, SVM dùng TF-IDF, TextLSTM và TextCNN trên cùng bảng tổng hợp. Vì lớp tin giả là lớp thiểu số, F1 và Recall tin giả được đặt cạnh Accuracy để tránh diễn giải chỉ dựa vào tỷ lệ dự đoán đúng chung.

**Bảng 3.12. So sánh các mô hình trên tập Test**

| Mô hình | Accuracy | Precision giả | Recall giả | F1 giả | AUC-ROC | Thời gian suy luận/mẫu (giây) |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes (TF-IDF) | 0,8293 | 0,8000 | 0,0964 | 0,1720 | 0,8281 | 0,000008 |
| SVM (TF-IDF) | 0,8359 | 0,5672 | 0,4578 | 0,5067 | 0,8275 | 0,004353 |
| TextLSTM | 0,8204 | 0,5114 | 0,5422 | 0,5263 | 0,8085 | 0,003726 |
| TextCNN | 0,8381 | 0,6667 | 0,2410 | 0,3540 | 0,8563 | 0,002160 |
| PhoBERT-SVM | 0,8869 | 0,7759 | 0,5422 | 0,6383 | 0,9228 | 0,7465669494 |

*Nguồn: `results/tables/model_comparison.csv`.*

**Hình 3.5. So sánh chỉ số giữa các mô hình**

![Model comparison](results/figures/model_comparison_bar.png)

*Nguồn: `results/figures/model_comparison_bar.png`.*

Các ma trận nhầm lẫn của các baseline được sinh tại `results/figures/cm_nb.png`, `cm_lstm.png`, `cm_cnn.png` và `svm_tfidf_confusion_matrix.png`.

**Bảng 3.13. Kiểm định McNemar giữa PhoBERT-SVM và các baseline**

| So sánh | Thống kê | p-value | Mẫu bất đồng |
|---|---:|---:|---:|
| PhoBERT-SVM – Naive Bayes (TF-IDF) | 12,5000 | 0,0003 | 50 |
| PhoBERT-SVM – SVM (TF-IDF) | 9,4902 | 0,0018 | 51 |
| PhoBERT-SVM – TextLSTM | 11,3649 | 0,0006 | 74 |
| PhoBERT-SVM – TextCNN | 8,1667 | 0,0038 | 54 |

*Nguồn: `results/tables/mcnemar_baselines.csv`.*

Bốn p-value đều nhỏ hơn 0,05 và vẫn nhỏ hơn ngưỡng Bonferroni 0,0125 khi hiệu chỉnh cho bốn phép so sánh. Kiểm định cho thấy dự đoán của PhoBERT-SVM khác có ý nghĩa thống kê so với từng baseline trong bộ kiểm tra này. Tuy nhiên, không nên chuyển trực tiếp kết quả kiểm định thành khẳng định “vượt trội rõ rệt”; cần đọc cùng F1, Recall, khoảng tin cậy và chi phí suy luận.

## 3.4. Nghiên cứu ablation về tiền xử lý

Thí nghiệm ablation dùng SVM Linear với `C = 1,0` cố định và giới hạn dữ liệu bằng tham số `--max-samples 1500`. Đây là thí nghiệm phân tích ảnh hưởng của các biến thể tiền xử lý, không phải quy trình chọn cấu hình RBF cuối cùng. Cột `extraction_time_sec` không được dùng để kết luận hiệu năng.

**Bảng 3.14. Kết quả ablation theo F1 lớp tin giả**

| Biến thể | Mô tả | Accuracy | Precision giả | Recall giả | F1 giả | AUC-ROC | F1 macro |
|---|---|---:|---:|---:|---:|---:|---:|
| A | Không tiền xử lý | 0,8271 | 0,5275 | 0,5783 | 0,5517 | 0,8237 | 0,7223 |
| B | Chỉ làm sạch URL/HTML | 0,8315 | 0,5412 | 0,5542 | 0,5476 | 0,8220 | 0,7220 |
| C | Bổ sung hashtag và dấu câu lặp | 0,8071 | 0,4778 | 0,5181 | 0,4971 | 0,8139 | 0,6889 |
| D | Bổ sung teencode, emoji và lặp ký tự | 0,8271 | 0,5316 | 0,5060 | 0,5185 | 0,8216 | 0,7066 |
| E | D + `pyvi` + lowercase | 0,8404 | 0,5714 | 0,5301 | 0,5500 | 0,8428 | 0,7265 |
| F | Như E nhưng không lowercase | 0,8603 | 0,6190 | 0,6265 | 0,6228 | 0,8635 | 0,7685 |

*Nguồn: `results/tables/ablation_preprocessing.csv`.*

**Hình 3.6. F1 lớp tin giả theo biến thể tiền xử lý**

![Ablation preprocessing](results/figures/ablation_preprocessing.png)

*Nguồn: `results/figures/ablation_preprocessing.png`.*

Biến thể E có F1 tin giả 0,5500, gần với A là 0,5517. Kiểm định McNemar giữa E và A cho `p = 0,5504`; các so sánh E với B, C, D và F lần lượt có p-value 0,7163; 0,0959; 0,5386 và 0,3211.

Theo đó, H2 **chưa được ủng hộ**: kết quả chưa cung cấp đủ bằng chứng rằng chuỗi tiền xử lý chuẩn làm tăng hiệu năng. F có F1 cao nhất trong bảng, nhưng chênh lệch E–F chưa có ý nghĩa thống kê. Việc F giữ lại chữ hoa có thể là một hướng giải thích cần kiểm chứng thêm, không phải kết luận nhân quả.

## 3.5. Phân tích lỗi và ngưỡng phân loại

### 3.5.1. Phân tích ngưỡng

Ngưỡng 0,5 được giữ làm kết quả chính. Các dòng khác trong bảng chỉ là phân tích độ nhạy trên cùng tập Test, dùng để quan sát đánh đổi Precision và Recall.

**Bảng 3.15. Phân tích độ nhạy theo ngưỡng dự đoán**

| Ngưỡng | Accuracy | Precision giả | Recall giả | F1 giả |
|---:|---:|---:|---:|---:|
| 0,3 | 0,8914 | 0,6977 | 0,7229 | 0,7101 |
| 0,4 | 0,8869 | 0,7286 | 0,6145 | 0,6667 |
| 0,5 | 0,8869 | 0,7759 | 0,5422 | 0,6383 |
| 0,6 | 0,8914 | 0,8400 | 0,5060 | 0,6316 |
| 0,7 | 0,8736 | 0,8250 | 0,3976 | 0,5366 |

*Nguồn: `results/tables/threshold_analysis.csv`.*

Giảm ngưỡng xuống 0,3 làm Recall tin giả tăng lên 0,7229 nhưng Precision giảm còn 0,6977. Tăng ngưỡng lên 0,6 làm Precision tăng lên 0,8400 nhưng Recall giảm còn 0,5060. Đây là đánh đổi vận hành, không phải căn cứ để thay đổi kết quả chính sau khi đã xem Test.

### 3.5.2. Phân tích FP và FN

Ở ngưỡng 0,5, hệ thống có 13 FP và 38 FN. Như vậy Recall tin giả chỉ đạt 0,5422; 38 trong 83 mẫu tin giả của Test bị bỏ sót. Các mẫu FP/FN đầy đủ được lưu trong `results/tables/error_analysis.csv` và dự đoán chi tiết trong `results/tables/predictions_test.csv`.

Một số mẫu trong `error_analysis.csv` cho thấy FP có thể mang văn phong cảnh báo, cảm xúc hoặc nội dung dịch bệnh nên bị đánh dấu là tin giả dù nhãn thật là 0. FN có thể có hình thức thông báo hành chính, mô tả sự kiện hoặc ngôn ngữ báo chí nên mô hình gán xác suất tin giả thấp. Đây là các hướng diễn giải định tính; không có nhãn nguyên nhân lỗi độc lập để khẳng định một nguyên nhân duy nhất.

Theo khung chi phí lỗi ở Chương 2, FP làm giảm độ tin cậy của cảnh báo đối với tin thật, còn FN làm bỏ sót tin giả. Vì FN đang nhiều hơn FP trong Test, việc điều chỉnh ngưỡng cho ứng dụng thực tế cần cân nhắc mục tiêu ưu tiên Recall, nhưng mọi thay đổi phải được đánh giá trên dữ liệu hiệu chỉnh riêng thay vì tối ưu trực tiếp trên Test.

## 3.6. Tích hợp mô hình vào Flask API và tiện ích Chrome

### 3.6.1. Triển khai REST API

`app/app.py` nạp PhoBERT tokenizer, PhoBERT model, `svm_model.pkl` và `scaler.pkl` từ thư mục kết quả. API cung cấp `GET /health`, `POST /predict` và `POST /predict_batch`. Với `/predict`, đầu vào là JSON có trường `text`; văn bản phải có độ dài từ 20 đến 20.000 ký tự. Kết quả gồm nhãn dự đoán, cờ `is_fake`, xác suất tin giả, điểm tin cậy, thời gian xử lý và phiên bản mô hình. Endpoint `/predict_batch` nhận danh sách từ 1 đến tối đa 32 văn bản.

**Bảng 3.16. Các endpoint chính của Flask API**

| Endpoint | Phương thức | Đầu vào | Kết quả chính |
|---|---|---|---|
| `/health` | GET | Không có | Trạng thái mô hình, thiết bị, phiên bản |
| `/predict` | POST | JSON `{ "text": "..." }` | Nhãn, xác suất giả, độ tin cậy, thời gian |
| `/predict_batch` | POST | JSON `{ "texts": [...] }` | Danh sách dự đoán và thời gian batch |

*Nguồn: `app/app.py`.*

### 3.6.2. Phát triển tiện ích mở rộng

Manifest của tiện ích khai báo quyền `activeTab`, `scripting`, `storage` và quyền truy cập localhost/127.0.0.1 cổng 5000. Popup cho phép nhập văn bản, lấy phần văn bản đang chọn trên trang, lưu địa chỉ API và gửi yêu cầu đến `/predict`. Kết quả được hiển thị bằng màu cảnh báo khác nhau cho tin giả và tin thật.

### 3.6.3. Quy trình hoạt động

Quy trình triển khai gồm: người dùng mở popup; nhập hoặc chọn văn bản; tiện ích gửi JSON đến Flask; API làm sạch văn bản, mã hóa bằng PhoBERT, chuẩn hóa bằng `StandardScaler`, chạy SVM; sau đó trả về xác suất và nhãn để popup hiển thị. Ứng dụng chạy cục bộ tại `http://127.0.0.1:5000`.

### 3.6.4. Đánh giá độ trễ

**Bảng 3.17. Thời gian suy luận và độ trễ API**

| Đại lượng | Giá trị |
|---|---:|
| PhoBERT mỗi mẫu | 0,7430862761 giây |
| SVM mỗi mẫu | 0,0034806734 giây |
| Tổng suy luận mỗi mẫu | 0,7465669494 giây |
| Phương pháp đo suy luận | `batch_100_samples` |
| Số request benchmark | 100 |
| Độ trễ trung bình API | 415,29 ms |
| Độ trễ trung vị API | 254,31 ms |
| P95 API | 920,32 ms |

*Nguồn: `results/tables/inference_time.csv` và `results/tables/api_latency.csv`.*

## 3.7. Thảo luận và hạn chế

### 3.7.1. Phân tích hiệu suất mô hình

Trên Test, mô hình đạt F1 tin giả 0,6383 và Recall tin giả 0,5422. AUC-ROC là 0,9228, nhưng AUC không thay thế cho yêu cầu kiểm soát FN ở ngưỡng 0,5. Khoảng tin cậy Bootstrap của F1 tin giả là [0,5333; 0,7260], cho thấy độ bất định đáng kể khi số mẫu tin giả chỉ là 83.

So sánh với các baseline cho thấy các chỉ số F1 và Recall có sự khác biệt, trong khi McNemar cho p-value nhỏ hơn ngưỡng Bonferroni 0,0125 ở cả bốn phép so sánh. Kết luận này chỉ áp dụng cho dữ liệu và quy trình thực nghiệm hiện tại; không dùng cụm “vượt trội rõ rệt” để khái quát ngoài phạm vi này.

### 3.7.2. Hạn chế

Thứ nhất, dữ liệu mất cân bằng với 825 mẫu tin giả trong 4.505 mẫu sau làm sạch, nên Accuracy không phản ánh đầy đủ khả năng bắt lớp thiểu số. Thứ hai, ReINTEL và VFND khác nhau về loại văn bản và độ dài. Thứ ba, PhoBERT được dùng ở chế độ đóng băng nên chưa đánh giá tác động của fine-tuning toàn bộ. Thứ tư, pipeline dùng `pyvi`, không phải RDRSegmenter. Thứ năm, toàn bộ phép đo hiện tại chạy trên CPU. Thứ sáu, ablation dùng tập con giới hạn và SVM Linear `C = 1,0`, nên không thể đồng nhất với kết quả mô hình RBF cuối. Cuối cùng, khoảng tin cậy F1 rộng vì số mẫu tin giả trong Test chỉ là 83.

## 3.8. Tóm tắt chương 3

Chương này đã mô tả môi trường, dữ liệu, quy trình tiền xử lý, lựa chọn SVM và kết quả đánh giá độc lập. Cấu hình chính là PhoBERT-SVM với kernel RBF và `C = 10,0`, được chọn theo F1 lớp tin giả trên Validation. Trên Test, F1 tin giả đạt 0,6383 và Recall đạt 0,5422; do đó khả năng bỏ sót tin giả vẫn là vấn đề cần tiếp tục cải thiện. Ablation chưa cung cấp đủ bằng chứng ủng hộ H2, còn các phép so sánh baseline được kiểm tra bằng McNemar và hiệu chỉnh Bonferroni. Sau cùng, mô hình đã được kết nối với Flask API và tiện ích Chrome để minh họa khả năng sử dụng trong ứng dụng.

## GHI CHÚ CHO TÔI

- Cần sửa quy ước nhãn tại mục 1.2.2.1 và 1.3.4.2 nếu còn ghi ngược nhãn.
- Cần sửa Bảng 2.2, Bảng 2.8 và mọi chỗ ghi 10.000 mẫu, 70/30 hoặc 8.000/1.000/1.000.
- Cần sửa mục 2.4.2 nếu đang chọn Linear làm kernel cuối; kết quả chính chọn RBF, `C = 10,0` theo Validation.
- Cần thay RDRSegmenter bằng `pyvi` ở nơi mô tả công cụ thực thi.
- Cần bỏ hoặc chú thích lại ý “dữ liệu cào tự động” nếu không có nguồn chứng minh trong repo.
- Cần xin phép tác giả VFND theo địa chỉ `thanh.hoquang@pm.me`.
- Cần bổ sung trích dẫn ReINTEL và VFND vào danh mục tài liệu tham khảo.

## File đã tạo/sửa

- Đã tạo: `Chuong3.md`.
