# Gợi ý và tư vấn hội thoại đa phương thức cho thiết bị điện tử

Đồ án nghiên cứu hệ thống tư vấn và gợi ý sản phẩm điện tử cho mạng xã hội thương mại, kết hợp bốn thành phần:

- **Multimodal RAG:** truy hồi lai (BM25, vector văn bản MiniLM, vector ảnh CLIP, hợp nhất theo hạng), kiểm ràng buộc thông số, sinh câu trả lời có viện dẫn và từ chối khi thiếu bằng chứng.
- **Hệ gợi ý trên đồ thị:** đồ thị dị thể người dùng–sản phẩm–nội dung–thông số, so với Popularity, LightGCN, VBPR, BM3.
- **Học tương phản đồ thị (GCL)** để tăng độ bền khi thiếu tương tác.
- **Co-Guiding:** cơ chế để kết quả truy hồi và gợi ý hướng dẫn lẫn nhau, cùng các biến thể học kết hợp.

Ngoài ra có một mô-đun định giá tin đăng điện thoại (CatBoost, khoảng giá và độ tin cậy) tách riêng trong `pricing/`.

Kết quả, phương pháp và phân tích giới hạn được trình bày trong [báo cáo](docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md) (chương 1–2); đề cương ở [docs/requirements](docs/requirements/De_cuong_Mang_xa_hoi_thuong_mai_do_dien_tu.md).

## Dữ liệu thực nghiệm

| Bộ dữ liệu | Vai trò |
|---|---|
| Amazon Reviews 2023, Electronics (kèm ảnh) | Thực nghiệm chính: truy hồi, gợi ý, GCL, Co-Guiding, joint |
| Amazon Reviews 2023, Baby_Products | Bộ gợi ý thứ hai |
| MIND-small | Bộ gợi ý thứ ba, chỉ văn bản |
| SWAP, VOGUE | Truy vấn hội thoại và truy hồi ngoài miền |
| Điện thoại | Kiểm tra ngành hàng và định giá |

Dữ liệu không nằm trong repository; xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) và cách tải lại ở [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Cấu trúc thư mục

```text
DATN/
├── docs/
│   ├── reports/                # Báo cáo (Markdown và Word)
│   ├── requirements/           # Đề cương
│   ├── ARCHITECTURE.md        # Ranh giới module
│   └── REPRODUCIBILITY.md     # Cách chạy lại từng thực nghiệm
├── rag/
│   ├── src/phone_assistant/    # Mã nghiên cứu theo package chức năng
│   ├── src/run.py              # Điểm vào chung cho các lệnh
│   ├── scripts/                # Launcher PowerShell cho từng thực nghiệm
│   ├── tests/                  # Kiểm thử
│   ├── docs/                   # Mô tả phương pháp, hình gốc (SVG, draw.io), tài liệu tham khảo
│   ├── data/fixtures/          # Tập nhỏ phục vụ kiểm thử
│   └── requirements/
├── pricing/                    # Mô-đun định giá điện thoại
├── tools/, tests/              # Kiểm tra cấu trúc và số liệu báo cáo
├── config/                     # Đăng ký module và ánh xạ đường dẫn
└── requirements/               # Nhóm cài đặt chung
```

Dữ liệu tải về, bộ nhớ đệm, trọng số và kết quả thực nghiệm được tạo cục bộ trong `rag/data`, `rag/artifacts`, `pricing/data` và `pricing/artifacts`, và không nằm trong Git.

## Cài đặt

Python 3.10, chạy từ thư mục gốc trong PowerShell:

```powershell
py -3.10 -m venv .venvs/rag
.venvs/rag/Scripts/python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venvs/rag/Scripts/python.exe -m pip install -r requirements/rag.txt
py -3.10 -m venv .venvs/pricing
.venvs/pricing/Scripts/python.exe -m pip install -r requirements/pricing.txt
```

Các lockfile Windows trong `rag/requirements` và `pricing/requirements` ghi môi trường đã dùng; chúng không bảo đảm cài nguyên trạng trên mọi nền tảng.

## Kiểm tra

```powershell
tools/Verify-Project.ps1
```

Lệnh chạy toàn bộ kiểm thử, kiểm tra cấu trúc và đối chiếu số liệu trong báo cáo với tệp kết quả. Không cần dữ liệu, mạng hay khóa API.

## Chạy ứng dụng

```powershell
# Tư vấn hội thoại (cần dữ liệu, chỉ mục và checkpoint đã chuẩn bị); mặc định không gọi API
powershell -NoProfile -File rag/scripts/run_handset_offline.ps1
# Định giá
powershell -NoProfile -File tools/Run-Pricing.ps1
```

Chat mặc định ở `http://127.0.0.1:7860`. Khóa API (OpenAI, Gemini, Hugging Face) là tùy chọn và đặt trong `rag/.env` (xem `.env.example`); không đưa khóa vào Git.

## Tài liệu

- [Báo cáo chương 1–2](docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md)
- [Tái lập kết quả](docs/REPRODUCIBILITY.md)
- [Kiến trúc](docs/ARCHITECTURE.md)
- [Cấu trúc package RAG](rag/docs/PACKAGES.md) và [định giá](pricing/docs/PACKAGES.md)
