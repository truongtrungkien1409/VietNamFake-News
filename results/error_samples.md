# Mẫu dự đoán sai trên tập Test

Nguồn: `results/tables/error_analysis.csv`.
Với FP, mẫu được sắp theo xác suất tin giả giảm dần. Với FN, mẫu được sắp theo xác suất tin giả tăng dần để đưa ra các trường hợp bị bỏ sót với độ tin cậy cao nhất của nhãn tin thật.

- Tổng FP trong bảng nguồn: 13.
- Tổng FN trong bảng nguồn: 38.

## 1. False Positive (FP)

Các mẫu có nhãn thật là 0 nhưng bị dự đoán là 1.

### FP 1

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.958229`
- Văn bản (tối đa 300 ký tự): Thầy xin chia sẽ bài này đến các phật tử đọc trong lúc dịch đang có phần chộn rộn ở nước ta khi ca nhiễm thứ 21. 👉 Không phải tự dưng mà ở Vũ Hán TQ nhanh chóng trở thành thành phố ma đâu ạ, trong thời gian đầu, một phần bởi vì tâm lý sợ hãi tích trữ đồ ăn và đổ xô ra siêu thị, chạy trốn khắp nơi để...

### FP 2

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.925882`
- Văn bản (tối đa 300 ký tự): Tỷ phú Mỹ Bill Gates: rút tài trợ cho WHO là quyết định nguy hiểm! Bill Gates đã giành rất nhiều tâm huyết và tiền bạc chống dịch Covid-19 và hàng tỷ đô cho tiêm chủng toàn cầu hàng chục năm nay. Dù rất phẫn nộ với Tedros tổng giám đốc WHO trong việc che đậy dịch bệnh với Trung cộng, nhưng Trump nên...

### FP 3

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.904684`
- Văn bản (tối đa 300 ký tự): Đến giờ thì tôi đã biết, ác quỷ có hình hài như thế nào!  Những ngày thật dài, và nặng nề, cuộc sống của mấy người bọn tôi, về cơ bản, bị xáo trộn, khủng hoảng nghiêm trọng bởi sự ám ảnh là quá lớn.  Dù trăm ngàn lần không muốn tin, nhưng sự thật mà tôi biết, cho đến lúc này Minh chính là hung thủ s...

### FP 4

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.902137`
- Văn bản (tối đa 300 ký tự): NƯỚC Ý ĐÃ TAN TÁC HẾT CẢ RỒI! Nước Ý, kinh đô của Công giáo đã tan tác hết cả rồi.  Cho đến nay, con virus quái ác Corona đã cướp đi sinh mạng của hàng ngàn người Ý và đang đe dọa tính mạng của hàng chục ngàn người khác.... Hàng trăm chiếc quan tài của người đã ra đi được xếp cạnh nhau chen chúc la...

### FP 5

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.858302`
- Văn bản (tối đa 300 ký tự): Covid-19 lan ra hơn 200 nước mà TQ hiện đã xuất khẩu khẩu trang và vật tư y tế cho 74 nước, 72 nước khác cũng đang thương lượng để mua, thế là tổng cộng gần 150 nước rồi còn gì nữa. kèo thơm quá.

### FP 6

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.837068`
- Văn bản (tối đa 300 ký tự): Một lời tâm sự của người con dân Tỉnh Vĩnh phúc! Kính thưa toàn thể đồng bào Việt Nam ! Chúng tôi là cư dân thuộc địa bàn Thôn Ngọc Bảo-Xã Sơn Lôi- Huyện Bình Xuyên- Tỉnh Vĩnh Phúc ( tâm dịch thứ 2 của xã Sơn Lôi) Hiện nay trên địa bàn xã đang xảy ra nhiều trường hợp bị nhiễm virus Corona và đang kh...

### FP 7

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.798984`
- Văn bản (tối đa 300 ký tự): Bà con ở TP HCM chú ý nè Quán bar Buddha ở Thảo Điền, quận 2 và phòng khám FV Sài Gòn ở Võ Thị Sáu, quận 1. Đây là 2 địa điểm có ca nhiễm Covid đã lui tới.

### FP 8

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.670025`
- Văn bản (tối đa 300 ký tự): XIN HÃY LÊN TIẾNG BẢO VỆ ĐÀN KHỈ Ở NÚI SƠN TRÀ - ĐÀ NẴNG !!!  Có cầu ắt sẽ có cung. Những món đặc sản rợn người như Hầu Não ( Não khỉ ) hay cao khỉ của con người đã giúp những kẻ tham tiền đi săn, bẫy khỉ để mang bán kiếm lời.  Nhìn kỹ hình ảnh những chú khỉ bị lòi xương ra ngoài, cụt chân, cụt tay,...

### FP 9

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.595651`
- Văn bản (tối đa 300 ký tự): Chỉ trong nhiệm kỳ Lê văn Thành làm bí thư kiêm chủ tịch HĐND Tp Hải Phòng (2015 - 2020), tập đoàn Hoàng Huy đc cả triệu m2 đất vàng ở Hải Phòng theo hình thức thanh toán BT nhờ cải tạo mấy cái chung cư cũ !!!  Bài báo dưới đây mới chỉ đề cập đến 99 ha đất vàng mà Hải Phòng đang hợp thức hoá lấy cho...

### FP 10

- Nhãn thật: `0`
- Nhãn dự đoán: `1`
- Xác suất tin giả: `0.587845`
- Văn bản (tối đa 300 ký tự): Quá thần thánh, 10 ngày của mình chắc chỉ họp cãi nhau không đủ TIỀM LỰC CỦA TRUNG QUỐC Hai bệnh viện khổng lồ cách nhau một con sông và khoảng 40 phút di chuyển.... Tuy nhiên, có một điều cần làm rõ ở đây là không có một tổng công ty hay doanh nghiệp lớn, hoặc chủ đầu tư nào đã một mình đứng ra xây...

## 2. False Negative (FN)

Các mẫu có nhãn thật là 1 nhưng bị dự đoán là 0.

### FN 1

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.016698`
- Văn bản (tối đa 300 ký tự): THÔNG TIN TỪ C.A TỈNH GIA LAI Theo thông báo của Bộ Công an, qua kiểm tra các thiết bị điện tử do các nước ngoài trao tặng (cho lực lượng công an và ngoài ngành công an) đã phát hiện một số yếu tố gây mất an ninh, an toàn thông tin, đặc biệt là một số điện thoại di động có nguồn gốc từ Trung Quốc do...

### FN 2

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.026961`
- Văn bản (tối đa 300 ký tự): Thông điệp của bức tranh mang tên là: “Khải Hoàn” có nghĩa là chiến thắng trở về ! Bức tranh nói về một ngày mai tươi sáng phía trước, khi Việt Nam hoàn toàn chiến thắng đại dịch COVID -19. Lúc ấy, những người bác sĩ được cất đi bộ áo “anh hùng”, tạm bỏ đi những gánh nặng đè nén suốt mấy tháng trời...

### FN 3

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.030109`
- Văn bản (tối đa 300 ký tự): Dự án 7.615 tỏi. Chỉ có 1 nhóm liên doanh nhà thầu tư nhân, 3 nhóm còn lại đều có ông kẹ Nhà nước.

### FN 4

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.034316`
- Văn bản (tối đa 300 ký tự): Dưới đây là danh sách 168 người VĨNH PHÚC đi khám và chữa bệnh tại Bệnh viện Bạch Mai, Hà Nội từ ngày 10/3-24/3/2020.  Bộ Y tế xác định bệnh viện Bạch Mai có hiện tượng lây nhiễm chéo Covid 19 và hiện được coi là "Ổ Dịch" Mọi người cố gắng đọc, ch.i.a s.ẻ rộng rãi để tỉnh nhà có biện pháp khoanh vùn...

### FN 5

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.045449`
- Văn bản (tối đa 300 ký tự): Theo ông Dũng, chỉ cần nhỏ 1 giọt tinh dầu tràm lên khẩu trang, khả năng ngăn ngừa dịch cúm sẽ được nâng cao. Sắp tới, ngành y tế sẽ yêu cầu các nhà cung cấp tăng cường bán loại tinh dầu này ở siêu thị, cửa hàng tiện lợi. Việc sử dụng tinh dầu tràm để tăng tính kháng khuẩn đã có nhiều nghiên cứu chứ...

### FN 6

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.062618`
- Văn bản (tối đa 300 ký tự): Chỉ là một bệnh nhân bình phục thì đưa họ về nước. Chỉ cần bác sĩ điều trị khám và quyết định thôi chứ làm gì mà HỘI CHẨN QUỐC GIA kinh rứa.  Các anh làm quá!  [<URL>

### FN 7

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.077194`
- Văn bản (tối đa 300 ký tự): Trung Quốc thay đổi số liệu để làm giảm số người nhiễm virus 2019-nCoV Ủy ban Y tế Quốc gia Trung Quốc đã thay đổi định nghĩa về xác nhận nhiễm virus 2019-nCoV trong hướng dẫn mới nhất của họ ngày 7/2. Bệnh nhân được xét nghiệm dương tính với virus nhưng không có triệu chứng sẽ không còn được coi là...

### FN 8

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.113714`
- Văn bản (tối đa 300 ký tự): Cần phải truy tìm và xử nghiêm những cán bộ y tế  tha hóa biến chất này. Điêu này rất nguy hại gây mất niềm tin dối với một bệnh viện lớn như Chợ Rẫy - nơi đang là niềm tự hào của ngành y tế khi đã cứu sống thành công phi công người Anh mắc COVD-19 ...

### FN 9

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.130827`
- Văn bản (tối đa 300 ký tự): Biểu quyết 17 người mà cái người đưa ra ý kiến biểu quyết to như vậy 16 người kia phải hoặc sẽ phụ thuộc thì sao họ chẳng đồng ý , có vẻ công bằng nhưng chỉ là phù phiếm

### FN 10

- Nhãn thật: `1`
- Nhãn dự đoán: `0`
- Xác suất tin giả: `0.169416`
- Văn bản (tối đa 300 ký tự): Quả thật đã đến lúc Chánh Án Nguyễn Hòa Bình phải “bạo tay” để sống còn. Không còn cách nào khác, vì sinh mạng chính trị của ông, gia đình ông, vì những cơ ngơi đồ sộ mà ông đã dùng biết bao “xương máu kẻ khác”mới xây dựng được như hôm nay.
