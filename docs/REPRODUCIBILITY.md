# Tái lập kết quả

Tài liệu này liệt kê cách chạy lại từng thực nghiệm trong báo cáo (`docs/reports/BAO_CAO_NGHIEN_CUU_GNN_GCL_MULTIMODAL_RAG.md`). Mọi lệnh chạy từ thư mục gốc repository; trên Windows dùng `.venvs/rag/Scripts/python.exe` hoặc truyền `-Python` cho launcher PowerShell.

## 1. Môi trường

- Python 3.10; PyTorch CPU 2.14.1, NumPy 2.2.6. Cài từ `requirements/rag.txt` (RAG) hoặc `requirements/pricing.txt` (định giá). `rag/requirements/requirements-dense-lock.txt` và `rag/requirements/requirements-training.txt` là phiên bản đã dùng, trên Windows.
- Công cụ dựng báo cáo (Word, công thức) dùng `requirements/reports.txt`.
- Không sao chép thư mục môi trường ảo sang máy khác; tạo lại từ requirements. Không tiếp tục (resume) một run CPU trên CUDA như cùng một run.
- Encoder dùng ở đúng revision và chạy offline sau khi đã tải về cache:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.
  - `sentence-transformers/clip-ViT-B-32`, revision `327ab6726d33c0e22f920c83f2ff9e4bd38ca37f`.
- API chỉ cần cho bốn việc: sinh câu trả lời, chấm judge, bóc tách thông số và (tuỳ chọn) Gemini trong ứng dụng. Khóa đặt trong `rag/.env` (xem `.env.example`). Mọi lệnh gọi API chạy ở chế độ thử (dry run) nếu không có `--live`, và yêu cầu `--budget-usd` và `--max-requests`.

## 2. Nguyên tắc chung

- Mỗi thực nghiệm ghi cấu hình, seed, mã nguồn băm và nguồn dữ liệu vào thư mục kết quả của nó. Thực nghiệm mới dùng thư mục đầu ra mới; không ghi đè kết quả đã hoàn thành.
- Chọn checkpoint và siêu tham số bằng validation. Test chỉ chấm một lần theo giao thức đã cố định.
- Run chỉ được coi là hoàn tất khi có `experiment.json` và bước kiểm tra tái dựng checkpoint (`verify_recsys_checkpoints`) khớp.
- Kết quả chạy dài có thể tiếp tục: launcher huấn luyện tự thêm `--resume` khi đã có `run_plan.json`; thực nghiệm API dùng `--resume-after-stop` và nhật ký request, không gọi lại request đã bắt đầu.

## 3. Các thực nghiệm

| Thực nghiệm | Mục trong báo cáo | Lệnh | Kết quả |
|---|---|---|---|
| Gợi ý, truy hồi, Co-Guiding và joint trên Electronics | 2.6.3.1–2.6.3.3 | `rag/scripts/run_electronics_research.ps1 -Stage {Prepare,Features,Train,Retrieval,Coupling,Joint}` | `results_electronics_*_v1` |
| Độ thưa tương tác, joint giữ nền ngữ nghĩa, đối chứng lịch sử | 2.6.3.3, 2.6.4.1 | `run_electronics_robustness`, `train_anchored_joint --resume`, rồi `analyze_electronics_robustness`, `evaluate_anchored_joint`, `evaluate_history_guidance` | `results_electronics_robustness_*`, `results_electronics_anchored_*`, `results_electronics_history_guidance_v1` |
| Truy hồi ngoài miền (VOGUE) | 2.6.3.1 | `evaluate_vogue_retrieval --phase {text,clip,evaluate}` | `results_vogue_ir_v1` |
| Gợi ý thứ hai: Amazon Baby_Products | 2.6.3.2 | `rag/scripts/run_amazon_category_research.ps1 -Stage All` | `results_amazon2023_baby_v1`, `..._analysis_v1` |
| Gợi ý thứ ba: MIND-small (chỉ văn bản) | 2.6.3.2 | `rag/scripts/run_mind_research.ps1 -Stage All` (cần `HF_TOKEN` trong `rag/.env`) | `results_mind_small_text_v1`, `..._analysis_v1` |
| Sinh câu trả lời có viện dẫn và judge (104 câu) | 2.6.3.1 | `prepare_generation_expansion`, `run_generation_expansion --provider openai --live --budget-usd <ngân sách> --max-requests <trần>`, `analyze_generation_expansion` | `results_rag_generation_expansion_openai_*` |
| Answer Relevance (phương pháp câu hỏi ngược) | 2.6.3.1 | `evaluate_answer_relevance --live --budget-usd <ngân sách> --max-requests <trần>`, rồi `evaluate_answer_relevance --analyze` | `results_answer_relevance_openai_v1` |
| Bóc tách thông số đa nguồn | 2.6.4.1 | `evaluate_spec_extraction --split dev --live ...`, sau đó `--split test --live ...` một lần, rồi `--analyze` | `results_spec_extraction_{dev,test}_v1` |
| Độ trễ CPU | 2.6.6 | `benchmark_electronics_latency --phase {search,text,clip,anchored}` (từng pha) | `results_electronics_latency_v1` |
| Định giá điện thoại | pricing | [pricing/docs/TRAINING_GUIDE.md](../pricing/docs/TRAINING_GUIDE.md) | `pricing/artifacts` |

Tên lệnh được gọi qua `rag/src/run.py <tên>` (ví dụ `rag/src/run.py evaluate_vogue_retrieval --phase text`); `--help` liệt kê tham số. Danh sách lệnh đăng ký ở `config/python_modules.json`.

### 3.1. Chuỗi Electronics

```powershell
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Prepare
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Features
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Train
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Retrieval
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Coupling
powershell -NoProfile -File rag/scripts/run_electronics_research.ps1 -Stage Joint
```

Prepare tải bản 5-core Electronics chính thức, tách người dùng hội thoại khỏi tương tác bổ sung, rồi tạo split. Features tải ảnh, mã hóa MiniLM và CLIP theo pha, và gắn feature vào đúng split sau khi kiểm tra ID và metadata. Train gồm 18 run (6 phương pháp × 3 seed; 100 epoch, chiều 16, batch 1024). Joint gồm 9 run riêng.

Nếu chưa có nguồn SWAP, VOGUE và split nền:

```powershell
.venvs/rag/Scripts/python.exe -B rag/src/run.py fetch_dataset_candidates --all-vogue-images --sim-revision 0a6b4d65370916b9e70b4590eb39c897376e0d83 --vogue-revision 68edb8e1ac43f3718844e00a0f40f4f10173b24b
.venvs/rag/Scripts/python.exe -B rag/src/run.py prepare_sim_benchmark --domain Electronics --output rag/data/external/benchmarks/electronics_v1
```

### 3.2. Thiếu phản hồi (giữ một phần cạnh huấn luyện)

```powershell
.venvs/rag/Scripts/python.exe -B rag/src/run.py prepare_sparse_electronics
.venvs/rag/Scripts/python.exe -B rag/src/run.py rebind_catalog_features --source-split rag/data/external/benchmarks/connected_electronics_v2 --source-features rag/artifacts/cache/electronics_features_v1 --target-split rag/data/external/benchmarks/electronics_sparse50_v1 --output rag/artifacts/cache/electronics_sparse50_features_v1
& ./rag/scripts/run_portable_training.ps1 -Split rag/data/external/benchmarks/electronics_sparse50_v1 -Features rag/artifacts/cache/electronics_sparse50_features_v1 -RunName electronics_sparse50_v1 -Methods @('graph_multimodal', 'graph_multimodal_gcl') -Seeds @(42, 43, 44) -Epochs 100 -Dim 16 -BatchSize 1024 -Device cpu
.venvs/rag/Scripts/python.exe -B rag/src/run.py verify_recsys_checkpoints --split rag/data/external/benchmarks/electronics_sparse50_v1 --features rag/artifacts/cache/electronics_sparse50_features_v1 --models rag/artifacts/models/recommendation/electronics_sparse50_v1 --experiment rag/artifacts/evaluation/results_electronics_sparse50_v1/experiment.json
```

Mặt nạ huấn luyện dùng seed 20261108 và giữ `ceil(0,5 × số cạnh của mỗi người dùng)`, ít nhất một cạnh, tức 5.813 trên 10.080 cạnh (57,67%) do làm tròn lên theo người dùng. Validation và test không đổi. Dùng mảng PowerShell thật cho `-Methods` và `-Seeds` khi gọi qua `powershell -File`.

Phân tích (mỗi analysis chỉ tạo sau khi các run tương ứng hoàn tất):

```powershell
.venvs/rag/Scripts/python.exe -B rag/src/run.py analyze_connected_results --recsys rag/artifacts/evaluation/results_electronics_multimodal_v1/experiment.json --ir rag/artifacts/evaluation/results_electronics_ir_v1/experiment.json --output rag/artifacts/evaluation/results_electronics_analysis_v1
.venvs/rag/Scripts/python.exe -B rag/src/run.py analyze_electronics_sparsity
.venvs/rag/Scripts/python.exe -B rag/src/run.py audit_electronics_joint
```

### 3.3. Bóc tách thông số

```powershell
.venvs/rag/Scripts/python.exe -B rag/src/run.py evaluate_spec_extraction --split dev --live --budget-usd 3 --max-requests 400
.venvs/rag/Scripts/python.exe -B rag/src/run.py evaluate_spec_extraction --split test --live --budget-usd 3 --max-requests 1400 [--resume-after-stop]
.venvs/rag/Scripts/python.exe -B rag/src/run.py evaluate_spec_extraction --split test --analyze
```

Bộ chấm và lời nhắc được cố định bằng tập dev trước khi chạy test; không chỉnh chúng sau khi đã thấy kết quả test. Thông số gốc của người bán chỉ dùng làm đáp án, không đưa vào đầu vào của mô hình.

## 4. Đánh giá bởi người

Các thực nghiệm sinh câu trả lời xuất tệp CSV để người chấm điền nhãn (`human_*`). Công cụ `audit_generation_human` không tạo điểm chất lượng khi cột này còn trống. Nhãn do mô hình AI gán được lưu ở tệp và thư mục riêng, không điền vào cột `human_*` và không được trình bày như đánh giá của người thật.

## 5. Dựng báo cáo

```powershell
.venvs/rag/Scripts/python.exe -B tools/renumber_report_tables.py --apply
.venvs/rag/Scripts/python.exe -B tools/check_report_numbers.py
.venvs/rag/Scripts/python.exe -B rag/src/run.py export_chapters_docx
```

`renumber_report_tables.py` đánh số lại bảng và hình tuần tự và cập nhật mọi tham chiếu. `check_report_numbers.py` kiểm mọi số thập phân trong các bảng kết quả có xuất hiện trong các tệp kết quả được trích dẫn; nó xác nhận giá trị tồn tại, không xác nhận số nằm đúng hàng hoặc cột. Xuất Word cần `python-docx`, `lxml`, `latex2mathml==3.81.1` và `MML2OMML.XSL` của Office (hoặc truyền `--math-xsl`). Công thức xuất ra là OMML chỉnh sửa được; hình SVG và draw.io, cùng `rag/docs/references.bib`, là nguồn gốc để chỉnh sửa.

## 6. Kiểm tra nhanh (không cần dữ liệu, mạng hay khóa API)

```powershell
tools/Verify-Project.ps1
```

Lệnh chạy bộ kiểm thử RAG và công cụ, kiểm tra cấu trúc và băm các nguồn kết quả đóng băng, và đối chiếu số trong báo cáo với tệp kết quả. Một số ca tích hợp cần dữ liệu cục bộ và tự bỏ qua khi thiếu.

## 7. Dữ liệu không nằm trong repository

Dữ liệu tải về, bộ nhớ đệm, trọng số và kết quả thực nghiệm (`rag/data/external`, `rag/artifacts`, `pricing/data`, `pricing/artifacts`) được git bỏ qua. Các bộ dữ liệu (Amazon Reviews 2023, MIND, SWAP, VOGUE) có điều khoản riêng của chủ sở hữu và được tải lại bằng các lệnh ở trên; xem `THIRD_PARTY_NOTICES.md`. Fixture nhỏ `rag/data/fixtures/recommendation` phục vụ kiểm thử đối chiếu baseline Popularity.
