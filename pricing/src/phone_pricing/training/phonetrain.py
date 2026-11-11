import argparse
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from catboost import CatBoostRegressor

from phone_pricing.features.phone_features import CAT_FEATURES, FEATURES
from phone_pricing.common.project_paths import (
    CLEAN_PHONE_DATA,
    MODEL_DIR,
    PHONE_MODEL,
    PHONE_MODEL_METADATA,
    configure_utf8_output,
    ensure_project_dirs,
)
configure_utf8_output()
import warnings
warnings.filterwarnings('ignore') # Tắt các cảnh báo không cần thiết


BEST_PARAMS = {
    "iterations": 1985,
    "learning_rate": 0.035734208684345625,
    "depth": 9,
    "l2_leaf_reg": 0.08109643173245676,
    "random_strength": 2.609659436746568,
    "bagging_temperature": 0.6228098550921024,
}


def build_catboost_model(cat_features, *, loss_function="RMSE", eval_metric="RMSE"):
    return CatBoostRegressor(
        **BEST_PARAMS,
        cat_features=cat_features,
        verbose=False,
        random_seed=42,
        loss_function=loss_function,
        eval_metric=eval_metric,
    )

def train_price_prediction_model(data_path=CLEAN_PHONE_DATA, model_save_path=PHONE_MODEL, train_quantiles=False):
    ensure_project_dirs()
    # 1. Đọc dữ liệu
    print("Loading data...")
    df = pd.read_csv(data_path)

    # 2. Tiền xử lý cơ bản (Lọc outlier)
    df = df[df['price_outlier'] == 0] 
    
    # 3. Định nghĩa các đặc trưng (Features) & Biến mục tiêu (Target)
    cat_features = CAT_FEATURES
    features = FEATURES
    target = 'log_price'

    # --- BƯỚC QUAN TRỌNG: CHUẨN HÓA CATEGORICAL (FIX LỖI TEXT) ---
    # Điền chữ 'unknown' cho các ô trống
    df[cat_features] = df[cat_features].fillna('unknown')
    
    # Vòng lặp ép toàn bộ dữ liệu chữ về chữ thường và xóa khoảng trắng thừa 2 đầu
    for col in cat_features:
        df[col] = df[col].astype(str).str.lower().str.strip()
    
    X = df[features]
    y = df[target]

    # 4. Chia tập Train / Test (Tỉ lệ 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 5. Khởi tạo và Huấn luyện mô hình CatBoost (Với thông số Optuna Tối ưu)
    print("Training model...")
    model = build_catboost_model(cat_features, loss_function='RMSE', eval_metric='RMSE')

    # Huấn luyện với cơ chế Early Stopping
    model.fit(
        X_train, y_train, 
        eval_set=(X_test, y_test), 
        early_stopping_rounds=50
    )

    # 6. Đánh giá mô hình trên tập Test
    y_pred_log = model.predict(X_test)

    # Đưa giá trị Logarit về lại đơn vị VNĐ (Dùng expm1)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    rmse = root_mean_squared_error(y_test_real, y_pred_real)
    
    print("\n" + "="*45)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print(f"MAPE (Sai số phần trăm trung bình) : {mape * 100:.2f}%")
    print(f"RMSE (Sai số trung bình bằng VNĐ)  : {rmse:,.0f} VNĐ")
    print("="*45)

    # 7. Phân tích Feature Importance (Mức độ quan trọng của biến)
    importance = model.get_feature_importance()
    feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)
    print("\nMức độ ảnh hưởng của các yếu tố tới Giá bán:")
    print(feat_imp)

    # 8. Lưu mô hình (.cbm)
    model.save_model(str(model_save_path))
    print(f"\n[Done] Đã lưu mô hình chuẩn tại: {model_save_path}")

    metadata = {
        "model_path": str(model_save_path),
        "features": features,
        "cat_features": cat_features,
        "target": target,
        "mape": float(mape),
        "mape_percent": float(mape * 100),
        "rmse_vnd": float(rmse),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "params": BEST_PARAMS,
        "quantile_models": {},
    }

    if train_quantiles:
        print("\nTraining quantile models for learned price interval...")
        quantiles = {
            "lower": 0.15,
            "median": 0.50,
            "upper": 0.85,
        }
        for name, alpha in quantiles.items():
            quantile_path = MODEL_DIR / f"phone_price_model_{name}.cbm"
            quantile_model = build_catboost_model(
                cat_features,
                loss_function=f"Quantile:alpha={alpha}",
                eval_metric=f"Quantile:alpha={alpha}",
            )
            quantile_model.fit(
                X_train,
                y_train,
                eval_set=(X_test, y_test),
                early_stopping_rounds=50,
            )
            quantile_model.save_model(str(quantile_path))
            metadata["quantile_models"][name] = {
                "alpha": alpha,
                "path": str(quantile_path),
            }
            print(f"[Done] Đã lưu quantile model {name}: {quantile_path}")

    with open(PHONE_MODEL_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[Done] Đã lưu metadata mô hình vào: {PHONE_MODEL_METADATA}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost model định giá điện thoại.")
    parser.add_argument("--data", default=str(CLEAN_PHONE_DATA), help="File dữ liệu đã chuẩn hóa.")
    parser.add_argument("--model", default=str(PHONE_MODEL), help="Đường dẫn lưu model trung tâm.")
    parser.add_argument("--with-quantiles", action="store_true", help="Train thêm lower/median/upper quantile models để dự đoán khoảng giá.")
    args = parser.parse_args()
    train_price_prediction_model(args.data, args.model, train_quantiles=args.with_quantiles)
