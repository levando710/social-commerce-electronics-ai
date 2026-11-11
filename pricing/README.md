# Định giá điện thoại

Module dự đoán giá tham khảo của tin đăng bằng đặc trưng văn bản và mô hình CatBoost, có baseline median sản phẩm tương tự và ước lượng khoảng giá. Nhãn là giá tin đăng quan sát, không được xem là giá giao dịch đã xác nhận.

## Thành phần

- `src/phone_pricing/features/phone_features.py`: chuẩn hóa dữ liệu, đặc trưng sản phẩm/tình trạng được ghi trong văn bản.
- `src/phone_pricing/models/price_baselines.py`: median theo các mức tương đồng và fallback.
- `src/phone_pricing/models/confidence_utils.py`: mô tả mức hỗ trợ của dữ liệu cho dự đoán.
- `src/phone_pricing/llm/llm_feature_extractor.py`: trích xuất bằng LLM tùy chọn.
- `src/phone_pricing/common/project_paths.py`: đường dẫn tương đối theo module, không phụ thuộc ổ đĩa.
- `src/run.py`: điểm vào cho thu thập, chuẩn hóa, so sánh, tối ưu và train.
- `tests/`: kiểm thử đường dẫn package và lệnh chạy.
- `src/phone_pricing/apps/app_demo.py`: giao diện Streamlit.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
py -3.10 -m venv .venvs/pricing
.venvs/pricing/Scripts/python.exe -m pip install -r requirements/pricing.txt
powershell -NoProfile -File tools/Run-Pricing.ps1
```

Ứng dụng cần `pricing/artifacts/models/phone_price_model.cbm`. Hai mô hình `phone_price_model_lower.cbm` và `phone_price_model_upper.cbm` cung cấp khoảng giá khi có. Dữ liệu `pricing/data/processed/phone_clean.csv` bổ sung catalog lựa chọn và ước lượng độ tin cậy; dữ liệu thiếu không được thay bằng nhãn nghiên cứu giả.

Môi trường thực nghiệm Windows/Python 3.10 được ghi trong `requirements/requirements-windows-lock.txt`; nhóm dependency tối thiểu ở `requirements/requirements.txt`. Các script so sánh và Optuna cần thêm `requirements/requirements-experiments.txt`.

## Train và đánh giá

```powershell
.venvs/pricing/Scripts/python.exe -m pip install -r pricing/requirements/requirements-experiments.txt
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py data_preprocessing
.venvs/pricing/Scripts/python.exe -B pricing/src/run.py compare_model
```

Các lệnh này ghi dữ liệu/kết quả và có thể train lâu. Quy trình chi tiết, bao gồm quantile và tuning: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md). Mô hình, dữ liệu gốc và metadata hiện tại được giữ trên máy làm việc; repository không kèm trọng số hoặc dữ liệu tin đăng.

## API tùy chọn

Các tên cấu hình được mô tả ở [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example). Chạy với khóa do người sử dụng tự cung cấp bằng biến môi trường hoặc Streamlit secrets. Không cần API khi chỉ dùng bộ trích xuất quy tắc và mô hình CatBoost đã có.

Xem [cấu trúc package](docs/PACKAGES.md) để biết trách nhiệm từng nhóm và cách bổ sung lệnh.
