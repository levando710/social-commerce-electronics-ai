# Quy trình train AI định giá điện thoại

Các lệnh dưới đây chạy trong thư mục `pricing/`. Để chạy từ gốc repository, dùng các lệnh trong [README](../README.md).

## 1. Chuẩn hóa dữ liệu

Chạy sau khi cào xong dữ liệu thô:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py data_preprocessing
```

Kết quả chính:

- `data\processed\phone_clean.csv`

## 2. So sánh thuật toán

Lệnh này có thể tốn thời gian vì train nhiều model:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py compare_model
```

Kết quả chính:

- `artifacts/reports\model_comparison_results.csv`
- `artifacts/reports\model_comparison_chart.png`
- `artifacts/reports\high_confidence_predictions.csv`

So sánh hiện gồm:

- Median Similar Products baseline
- Ridge Regression
- Random Forest
- XGBoost
- CatBoost mặc định
- CatBoost tối ưu

Bảng kết quả có thêm `HighConf MAPE (%)`, là sai số trên nhóm mẫu đáng tin cậy hơn:

- hãng phổ biến
- model đủ mẫu
- có thông tin quốc tế/lock
- tình trạng chưa sửa
- bảo hành rõ hơn
- không có dấu hiệu hỏng nặng/iCloud/Face ID

## 3. Tối ưu CatBoost bằng Optuna

Lệnh ngắn cho demo:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py tune_phone_model --trials 30
```

Lệnh nghiêm túc hơn sau khi có dataset cuối:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py tune_phone_model --trials 100
```

Study được lưu vào `artifacts/models\phone_optuna.db`, nên có thể chạy tiếp nhiều lần mà không mất kết quả cũ.

## 4. Train model cuối

Train model trung tâm:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py phonetrain
```

Train thêm quantile models để app có khoảng giá học từ dữ liệu:

```powershell
..\.venvs\pricing\Scripts\python.exe src\run.py phonetrain --with-quantiles
```

Kết quả chính:

- `artifacts/models\phone_price_model.cbm`
- `artifacts/models\phone_model_metadata.json`
- `artifacts/models\phone_price_model_lower.cbm` nếu dùng `--with-quantiles`
- `artifacts/models\phone_price_model_median.cbm` nếu dùng `--with-quantiles`
- `artifacts/models\phone_price_model_upper.cbm` nếu dùng `--with-quantiles`

## 5. Chạy demo

```powershell
..\.venvs\pricing\Scripts\streamlit.exe run src\phone_pricing\apps\app_demo.py
```

Nếu có quantile models, app dùng khoảng giá học từ dữ liệu. Nếu chưa có, app dùng biên độ MAPE.
