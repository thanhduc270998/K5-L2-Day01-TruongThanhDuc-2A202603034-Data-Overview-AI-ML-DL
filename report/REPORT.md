# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
[
  {
    "taxonomy_name": "ImageNet-1K",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }
]

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về một vật thể riêng bởi thì tôi thấy trong danh sách record có dữ liệu `sample_id = traffic` tức là bức ảnh liên quan tới hình ảnh giao thông, 
trong đó "rank": 1 là mức độ ưu tiên phân tích đối tượng trong bức ảnh, "class_id" và "class_name" nói về đối tượng được quan tâm còn "score" chính là mức độ điểm hiển thị của đối tượng đó trong bức ảnh

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp không phải do model tự nghĩ ra mà được xác định từ bộ dữ liệu huấn luyện hoặc cấu hình đi kèm model


4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Vì thực tế mỗi tên lớp (class_name) thì có thể có nhiều class_id và nhiều taxonomy_name, một tên lớp có thể sẽ không biểu hiện rõ được nó thuộc phân loại nào, id nào trong phân loại đó
-> từ đó sẽ khiến model train nhầm lẫn sai dữ liệu, theo cấp độ thì taxonomy_name sẽ là thuộc tính cha, class_id sẽ là thuộc tính con và class_name là tên biểu thị của class_id để cho người đọc vào có thể hiểu class đó là gì

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- guideline cần phải nói rõ chủ thể chính trong bức ảnh là gì, chọn theo tiêu chí như thế nào và nếu không chọn được cần phải làm gì

6. Vì sao không được lấy `score` của model làm ground truth?
- Vì score chỉ là điểm chấm biểu hiện mức độ tự tin của AI đưa ra dự đoán của nó chứ chưa chắc đáp án đó là đúng

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
[
	{
		"class_name": "cup",
		"score": 0.381542,
		"bbox_xyxy": [
		  119.95,
		  272.2,
		  141.78,
		  304.16
		],
		"bbox_width": 21.84,
		"bbox_height": 31.96
	}
]

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Tọa độ góc trái: 119.95 , góc trên: 272.2, góc phải: 141.78, góc dưới: 304.16 , rộng: 21.84 pixel, cao: 31.96 pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- chọn hai ngưỡng 0.20 và 0.60 thì ngưỡng 0.20 có 1 prediction , ngưỡng 0.60 có 0 prediction do 0.20 < 0.381542 < 0.60

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng, model giữ lại nhiều bounding box hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Vì vậy reviewer phải làm thêm công việc kiểm tra và xác nhận các prediction này:
	+ Kiểm tra xem prediction này có đúng không
	+ Kiểm tra Bounding box
	+ Phân biệt vật thể dễ nhầm
	+ Xử lí prediction trùng lặp
	+ Xác nhận Ground Truth
	
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Người gán nhãn phải vẽ Bounding Box bao quanh toàn bộ phần vật thể nhìn thấy trong ảnh, đảm bảo khung hình sát với ranh giới vật thể nhất có thể, không bao gồm quá nhiều vùng nền xung quanh.
Mỗi vật thể thuộc danh sách lớp phải được gán một Bounding Box riêng và một class_name tương ứng. Không gán nhãn cho vật thể không thể xác định rõ lớp; đối với vật thể bị che khuất, chỉ vẽ quanh
phần nhìn thấy được. Tất cả Bounding Box phải tuân thủ cùng một quy tắc thống nhất, không tự ý thay đổi theo từng ảnh

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật bị cắt mép ảnh là Người (person 0.61), người này bị cắt chỉ nhìn thấy mỗi cái tay không thể xác định rõ hình dáng
- Tôi sẽ viết thành quy tắc trong guideline vì đây là tình huống thường gặp trong dữ liệu Object Detection, ta có thể xây dựng quy tắc chung để tất cả các annotator áp dụng thống nhất mà không cần hỏi Lab Coach mỗi lần gặp.
- Quy tắc đề xuất đưa vào guideline: Đối với vật thể bị cắt bởi mép ảnh, người gán nhãn chỉ vẽ Bounding Box bao quanh phần vật thể còn nhìn thấy trong ảnh. Không tự suy đoán hoặc vẽ thêm phần vật thể nằm ngoài khung hình.
Nếu phần còn nhìn thấy không đủ để xác định chính xác class, hãy đánh dấu cần review hoặc escalation theo quy trình.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
    "instance_id": "kitchen-011",
    "class_name": "spoon",
    "score": 0.359147,
    "polygon_point_count": 37,
    "polygon_xy": [
      [
        529.0,
        66.0
      ],
      [
        529.0,
        72.0
      ],
      [
        528.0,
        73.0
      ],
      [
        528.0,
        74.0
      ],
      [
        527.0,
        75.0
      ]
    ]
}

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp thêm thông tin như là hình dáng, đường biên, diện tích, mức độ che khuất
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- "instance_id" dùng để định danh duy nhất một đối tượng cụ thể trong một ảnh hoặc một mẫu dữ liệu, giúp hệ thống phân biệt các vật thể khác nhau, kể cả khi chúng có cùng class_name
- "instance_id" **không** phải class ID và **không** phải tracking ID tức là "instance_id" chỉ dùng để nhận diện một đối tượng cụ thể trong dữ liệu, không dùng để xác định loại đối tượng "class ID"
và cũng không dùng để theo dõi đối tượng qua thời gian "tracking ID"

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Người gán nhãn phải vẽ đường biên Mask bám sát đường viền thực tế của vật thể, chỉ bao phủ phần pixel thuộc về vật thể, không lấy thêm vùng nền hoặc vật thể bên cạnh. Đối với vật thể có đường viền cong hoặc hình dạng phức tạp,
cần vẽ đủ chi tiết để giữ đúng hình dạng. Nếu vật thể bị che khuất hoặc cắt mép ảnh, chỉ gán nhãn phần có thể quan sát được, không tự suy đoán phần bị khuất hoặc nằm ngoài ảnh. Mỗi vật thể phải có một Mask riêng tương ứng với instance_id

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Chỗ biên mờ / hai vật chạm nhau / bị che là : dining table 0.58 và person 0.43
- Dùng guideline vì đây là tình huống thường gặp trong bài toán Instance Segmentation, có thể chuẩn hóa thành quy tắc chung

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một ảnh tương ứng một class_name nếu bài toán yêu cầu một nhãn duy nhất.| Ảnh kitchen có nhiều chủ đề: người, bát, thìa, bàn, lò nướng. Khó xác định chủ thể chính để chọn một lớp. | Áp dụng guideline để chọn chủ thể chính. Nếu không xác định được, dùng ambiguous hoặc other theo quy định. | Kiểm tra lớp được chọn có phù hợp nội dung ảnh và guideline không. Kiểm tra tính nhất quán giữa các annotator. |
| Phát hiện vật thể | Mỗi vật thể là một bộ nhãn gồm class_name và bbox_xyxy (bounding box). | JSON cup có score = 0.381542, bbox rộng 21.84, cao 31.96. PNG có nhiều box chồng lấn, ví dụ bowl 0.46, bowl 0.70, oven 0.69. Người person 0.43 bị bàn che khuất. | Xác định đúng class, vẽ box sát vật thể, xử lý vật thể bị che hoặc cắt mép theo guideline. Không lấy score làm Ground Truth. | Kiểm tra class, vị trí box, độ bao phủ, box trùng lặp và vật thể bị bỏ sót. |
| Instance segmentation | Mỗi vật thể có một class_name, instance_id và một Mask/Polygon riêng. | JSON spoon có score = 0.359147, polygon_point_count = 37. PNG có vùng người và bàn tiếp giáp, khó xác định ranh giới Mask. | Vẽ Mask riêng cho từng instance, bám sát biên vật thể, không lấy pixel của nền hoặc vật thể khác. | Kiểm tra hình dạng Mask, đường biên, pixel bị gán nhầm, vùng bị thiếu và sự phân biệt giữa các instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng ảnh và JSON trong phạm vi bài lab được phép, không chia sẻ, phát tán hoặc tiết lộ dữ liệu, thông tin nhạy cảm và không tự ý tải dữ liệu lên các nền tảng bên ngoài
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Tôi sẽ dừng xử lí, không chỉnh sửa hoặc chia sẻ dữ liệu và báo ngay cho Lab Coach hoặc người phụ trách dự án để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
