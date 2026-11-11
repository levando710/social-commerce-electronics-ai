# Tổ chức mã nguồn Python

Mã nghiên cứu thuộc package `phone_assistant`. Các thành phần dùng import đầy đủ theo package; `__init__.py` không nạp mô hình hoặc chạy tác vụ khi import.

```text
rag/
├── src/
│   ├── run.py
│   └── phone_assistant/
│       ├── __main__.py
│       ├── apps/
│       ├── common/
│       ├── data/
│       │   ├── collection/
│       │   └── preparation/
│       ├── features/
│       ├── retrieval/
│       │   ├── search/
│       │   ├── indexing/
│       │   ├── queries/
│       │   └── serving/
│       ├── recommendation/
│       ├── training/
│       ├── conversation/
│       ├── llm/
│       ├── evaluation/
│       │   ├── audits/
│       │   ├── metrics/
│       │   ├── analysis/
│       │   ├── benchmarks/
│       │   └── reviews/
│       └── reporting/
│           ├── figures/
│           └── chapters/
├── scripts/
└── tests/
```

| Package | Trách nhiệm |
|---|---|
| `apps` | Máy chủ chatbot và giao diện đánh giá thủ công |
| `common` | Schema catalog, đường dẫn dùng chung, nhận diện MIME ảnh |
| `data` | Thu thập, làm sạch, chuẩn bị corpus, chia tập và lọc điện thoại |
| `features` | Chuẩn bị vector văn bản/ảnh, phân tích ảnh và ràng buộc feature với catalog |
| `retrieval` | BM25, dense/hybrid search, hiểu truy vấn, ràng buộc sản phẩm và runtime truy hồi |
| `recommendation` | Dữ liệu đồ thị, mô hình gợi ý, GCL, joint ranking và dịch vụ gợi ý |
| `training` | Vòng train, checkpoint, resume, kiểm tra thiết bị và điều phối thực nghiệm |
| `conversation` | Trạng thái hội thoại, hồ sơ tương tác, bằng chứng review và Co-Guiding |
| `llm` | Tích hợp Gemini và xử lý yêu cầu gọi mô hình |
| `evaluation` | Chỉ số, kiểm toán dữ liệu/kết quả, benchmark, smoke và công cụ chấm |
| `reporting` | Hình minh họa và xuất báo cáo |

Trong `evaluation`, `audits` kiểm tra dữ liệu và tính toàn vẹn thực nghiệm; `metrics` tính các chỉ số; `analysis` tổng hợp/phân tích kết quả; `benchmarks` đo thời gian và smoke test; `reviews` tạo biểu mẫu, chuẩn bị đợt chấm và kiểm tra nhãn thủ công.

`data` tách thu thập nguồn (`collection`) khỏi chuẩn bị corpus/split (`preparation`); bộ lọc phạm vi điện thoại vẫn ở `data.phone_scope`. `retrieval` tách thuật toán tìm kiếm (`search`), xây chỉ mục (`indexing`), hiểu truy vấn và ràng buộc (`queries`), runtime/dịch vụ (`serving`). `reporting` tách tạo hình (`figures`) khỏi tổng hợp và xuất chương (`chapters`). Các package còn lại có số file ít và trách nhiệm rõ nên giữ nguyên.

## Điểm vào

Từ gốc repository, dùng Python của môi trường RAG:

```powershell
.venvs/rag/Scripts/python.exe -B rag/src/run.py --list
.venvs/rag/Scripts/python.exe -B rag/src/run.py chat_app --help
.venvs/rag/Scripts/python.exe -B rag/src/run.py chat_app --handsets --offline --method legacy
.venvs/rag/Scripts/python.exe -B rag/src/run.py train_recsys_portable --help
powershell -NoProfile -File rag/scripts/run_handset_offline.ps1
```

`--list` liệt kê các module đã đăng ký, gồm cả thư viện dùng nội bộ. Chỉ các module có chương trình dòng lệnh mới thực hiện tác vụ khi chạy trực tiếp qua dispatcher. Tham số sau tên lệnh được chuyển nguyên vẹn tới chương trình đó, bao gồm `--help`.

Khi thư mục `rag/src` có trong Python import path, có thể dùng `python -m phone_assistant ...` hoặc `python -m phone_assistant.apps.chat_app ...`. Dispatcher `run.py` giúp chạy từ gốc repository mà không phải cài editable hoặc tự thêm đường dẫn.

## Import và bổ sung chức năng

Ví dụ import:

```python
from phone_assistant.retrieval.search.bm25 import tokenize
from phone_assistant.recommendation.recsys_data import InteractionData
```

Đặt module mới vào package có trách nhiệm phù hợp. Dùng import đầy đủ, không thêm các package con vào `sys.path` để giữ import phẳng. Cập nhật `config/python_modules.json` khi cần đăng ký tên lệnh hoặc đối chiếu basename source trong manifest. `common.project_layout.source_path()` tìm source hiện hành theo registry; việc đọc dữ liệu vẫn dựa vào gốc mô-đun, không phụ thuộc thư mục đang chạy lệnh.

Các launcher PowerShell ở `rag/scripts/` gọi qua `src/run.py`.

## Bảo toàn thực nghiệm

Dữ liệu, trọng số và manifest kết quả không được sửa để khớp mã mới. Kiểm tra run lịch sử đối chiếu mã nguồn băm trong snapshot của thư mục kết quả; thực nghiệm mới ghi vào thư mục output mới.

## Phân loại nhóm hàng

`phone_assistant.retrieval.queries.category_types.category_type(product)` trả về nhóm hàng (computers, camera, tv_video, headphones...) và vai trò (device, accessory, component, service) từ đường dẫn danh mục Amazon, không suy từ tiêu đề; sản phẩm không có danh mục giữ `unassigned`. Bộ nhận diện theo quy tắc cũ (`product_types`) được giữ nguyên để không đổi kết quả frozen. Đề xuất bộ thuộc tính theo nhóm hàng suy từ dữ liệu: `rag/data/discovery/electronics_family_schema_v1.json`.
