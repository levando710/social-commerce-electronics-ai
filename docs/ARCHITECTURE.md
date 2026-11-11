# Kiến trúc và ranh giới module

## Hai thành phần nghiệp vụ

`rag/` nghiên cứu tư vấn hội thoại và gợi ý: chuẩn hóa catalog → embedding chữ/ảnh → đồ thị và chỉ mục → hiểu truy vấn → truy hồi/kiểm ràng buộc → Co-Guiding → câu trả lời và danh sách gợi ý. Lịch sử visitor được lưu riêng dữ liệu benchmark; người dùng công khai trong tập nghiên cứu không được gán thành người đang truy cập web.

`pricing/` định giá tin điện thoại: chuẩn hóa mô tả → đặc trưng → mô hình dự báo giá/khoảng giá → độ tin cậy. Module có đường dẫn và môi trường riêng. Chưa có một hợp đồng API hợp nhất triển khai hai module thành dịch vụ web sản xuất.

## Tổ chức mã và tài sản

| Nhóm | Đường dẫn | Quy tắc |
|---|---|---|
| Mã nghiên cứu | `rag/src/phone_assistant`, `rag/scripts`, `rag/tests` | Package theo chức năng; `rag/src/run.py` chuyển tiếp tên lệnh và tham số |
| Mã định giá | `pricing/src/phone_pricing`, `pricing/tests` | Package dữ liệu, đặc trưng, mô hình, training, LLM và ứng dụng; điểm vào `pricing/src/run.py` |
| Công cụ repository | `tools`, `tests` | Kiểm tra cấu trúc và số liệu báo cáo, chỉ dùng thư viện chuẩn |
| Tài liệu | `docs`, `rag/docs`, `pricing/docs` | README gốc là điểm vào |
| Dữ liệu tải về | `rag/data/external`, `pricing/data` | Không nằm trong Git |
| Feature và encoder | `rag/artifacts/cache` | Nhiều cache là hiện vật thực nghiệm cần thiết, không xóa hàng loạt |
| Checkpoint | `rag/artifacts/models/recommendation`, `rag/artifacts/models/joint`, `pricing/artifacts/models` | Giữ đối chiếu manifest; không nằm trong Git |
| Kết quả đã lưu | `rag/artifacts/evaluation`, `pricing/artifacts/reports` | Kết quả nghiên cứu có nguồn; không thay bằng lần chạy mới |

Registry trong `config/python_modules.json` và `config/pricing_modules.json` ghi vị trí từng module; tên lệnh và tham số được chuyển qua dispatcher. Mỗi thư mục kết quả chứa bản sao mã tại thời điểm chạy để kiểm tra nguồn gốc của số liệu.
