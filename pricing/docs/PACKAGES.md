# Package định giá điện thoại

```text
pricing/
├── src/
│   ├── run.py
│   └── phone_pricing/
│       ├── __main__.py
│       ├── common/       # Đường dẫn, cấu hình thư mục và chuẩn hóa output
│       ├── data/         # Thu thập và chuẩn hóa tập tin đăng
│       ├── features/     # Đặc trưng văn bản và sản phẩm
│       ├── models/       # Baseline và độ tin cậy dự đoán
│       ├── training/     # Train CatBoost, so sánh, Optuna
│       ├── llm/          # Trích xuất đặc trưng bằng LLM tùy chọn
│       └── apps/         # Giao diện Streamlit
├── tests/
├── requirements/
├── docs/
├── data/
└── artifacts/
```

Các file thuộc `phone_pricing` sử dụng import đầy đủ theo package. `common.project_paths` lấy gốc mô-đun từ vị trí file, vì vậy dữ liệu và mô hình được tìm đúng khi lệnh chạy từ gốc repository hoặc thư mục khác. Các file `__init__.py` không khởi động giao diện, thu thập dữ liệu hoặc train.

## Lệnh từ gốc repository

```powershell
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py --list
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py phonetrain --help
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py getdata --help
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py data_preprocessing
powershell -NoProfile -File tools/Run-Pricing.ps1
.venvs/pricing/Scripts/python.exe -B -m unittest discover -s pricing/tests
```

`data_preprocessing` ghi dữ liệu đã chuẩn hóa; các lệnh `phonetrain`, `compare_model` và `tune_phone_model` thực hiện train/đánh giá khi được chạy thực tế. Dispatcher chỉ chuyển tiếp tham số; `--help` không khởi chạy train. Danh sách lệnh nằm trong `config/pricing_modules.json` ở gốc repository. Khi `pricing/src` thuộc Python import path, có thể gọi `python -m phone_pricing ...`.

Giao diện được khởi động qua launcher chung để Streamlit chạy với thư mục làm việc `pricing/` và đọc cấu hình riêng đúng vị trí. File ứng dụng thực tế là `src/phone_pricing/apps/app_demo.py`.

## Bảo toàn mô hình

Việc chia package giữ nguyên thuật toán xử lý, danh sách đặc trưng, dữ liệu và trọng số. Đối chiếu bốn mô hình CatBoost trên tám mẫu kiểm tra cho kết quả dự đoán giống hệt trước khi di chuyển; dữ liệu và mô hình được kiểm SHA256. Bản source trước thay đổi nằm trong `artifacts/provenance/pre_functional/` để đối chiếu cục bộ.

## Đo lại sai số theo giao thức không rò rỉ

```powershell
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py evaluate_holdout_protocol
```

`phonetrain` dùng tập test làm `eval_set` cho early stopping, nên số cây được chọn bằng chính dữ liệu test và MAPE lưu trong metadata là lạc quan. Lệnh trên giữ nguyên các dòng test (kiểm bằng cách mô hình frozen phải tái hiện đúng MAPE đã lưu), chỉ tách validation từ phần train, và ghi `artifacts/reports/holdout_protocol_v1/result.json`. Không ghi đè mô hình, metadata hay dữ liệu.

Kết quả đã lưu (3 seed): MAPE 32,54 ± 0,13% (mô hình frozen: 31,39%); RMSE 2,90 triệu đồng (frozen 2,79 triệu). Baseline trung vị theo model + dung lượng có MAPE 58,6%, RMSE 2,92 triệu và median APE 12,9% so với 18,2% của CatBoost, tức là lợi thế của CatBoost không rõ ràng theo RMSE và median APE. Giới hạn: tham số lấy từ lần tinh chỉnh từng thấy tập này, nhãn là giá tin đăng, và mới chỉ khử trùng theo `ad_id`.
