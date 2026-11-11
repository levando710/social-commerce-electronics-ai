# RAG và gợi ý hội thoại

## Thành phần

- `src/phone_assistant/`: các package theo chức năng; xem [sơ đồ package](docs/PACKAGES.md).
- `src/run.py`: điểm vào chung cho các lệnh Python.
- `scripts/`: launcher PowerShell.
- `tests/`: kiểm thử thuật toán và tích hợp.
- `requirements/`: dependency; cài theo nhóm từ `requirements/rag.txt` ở gốc repository.
- `docs/`: tài liệu phương pháp, hình gốc và hướng dẫn.
- `data/`: nguồn công khai, các bộ thu thập và fixture kiểm thử.
- `artifacts/`: checkpoint, vector/ảnh/encoder, runtime và kết quả (git bỏ qua).

Đường dẫn dùng chung được mô tả trong `config/layout.json` ở gốc dự án.

## Chạy từ thư mục gốc dự án

```powershell
powershell -NoProfile -File rag/scripts/run_handset_offline.ps1
.venvs/rag/Scripts/python.exe -B -m unittest discover -s rag/tests
```

Các lệnh demo cần tài sản cục bộ đã tạo. Mặc định không gọi Gemini. Thêm `-Neural` khi cache encoder sẵn có. API sinh chỉ được bật rõ khi thực hiện đợt đánh giá có cấu hình và ngân sách riêng.

## Dữ liệu và thực nghiệm

[Tái lập kết quả](../docs/REPRODUCIBILITY.md) liệt kê chuỗi chuẩn bị, feature, train, IR, Co-Guiding và joint.

Mỗi thư mục kết quả có manifest và snapshot mã tại thời điểm chạy; thực nghiệm mới ghi vào thư mục mới.
