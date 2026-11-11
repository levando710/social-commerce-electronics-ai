import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from phone_pricing.features.phone_features import CAT_FEATURES, FEATURES
from phone_pricing.models.price_baselines import SimilarProductMedianRegressor
from phone_pricing.training.phonetrain import BEST_PARAMS
from phone_pricing.models.confidence_utils import build_confidence_profile, high_confidence_mask
from phone_pricing.common.project_paths import (
    CLEAN_PHONE_DATA,
    HIGH_CONFIDENCE_PREDICTIONS,
    MODEL_COMPARISON_CHART,
    MODEL_COMPARISON_RESULTS,
    configure_utf8_output,
    ensure_project_dirs,
)
configure_utf8_output()
import warnings
warnings.filterwarnings('ignore')

def run_model_comparison(data_path=CLEAN_PHONE_DATA):
    ensure_project_dirs()
    print("1. Đang nạp và chuẩn bị dữ liệu...")
    df = pd.read_csv(data_path)
    df = df[df['price_outlier'] == 0]
    
    cat_features = CAT_FEATURES
    df[cat_features] = df[cat_features].fillna('Unknown')
    
    X = df[FEATURES]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )

    # 2. Định nghĩa các mô hình (Bao gồm cả CatBoost Tối ưu Optuna)
    models = {
        "1. Median Similar Products": SimilarProductMedianRegressor(),
        "2. Ridge Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))
        ]),
        "3. Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        "4. XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1))
        ]),
        "5. CatBoost (Mặc định)": CatBoostRegressor(
            iterations=1000, 
            learning_rate=0.05, 
            depth=6, 
            cat_features=cat_features, 
            verbose=False, 
            random_seed=42
        ),
        "6. CatBoost (Tối ưu)": CatBoostRegressor(
            **BEST_PARAMS,
            cat_features=cat_features, 
            verbose=False, 
            random_seed=42
        )
    }

    confidence_profile = build_confidence_profile(pd.concat([X_train, y_train], axis=1))
    high_conf_mask = high_confidence_mask(X_test, confidence_profile)
    print(f"High-confidence test samples: {int(high_conf_mask.sum())}/{len(X_test)}")

    results = []
    predictions = {}
    print("\n2. Bắt đầu huấn luyện và đánh giá các mô hình...\n")
    
    for name, model in models.items():
        print(f"Đang chạy {name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_test)
        
        y_pred_real = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test)
        predictions[name] = y_pred_real
        
        mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100
        rmse = root_mean_squared_error(y_test_real, y_pred_real)
        if high_conf_mask.any():
            high_conf_mape = mean_absolute_percentage_error(
                y_test_real[high_conf_mask],
                y_pred_real[high_conf_mask],
            ) * 100
            high_conf_rmse = root_mean_squared_error(
                y_test_real[high_conf_mask],
                y_pred_real[high_conf_mask],
            )
        else:
            high_conf_mape = np.nan
            high_conf_rmse = np.nan
        train_time = time.time() - start_time
        
        results.append({
            "Mô hình": name,
            "MAPE_raw": mape, # Giữ biến số nguyên bản để vẽ biểu đồ
            "MAPE (%)": f"{mape:.2f}%",
            "RMSE (VNĐ)": f"{rmse:,.0f}",
            "HighConf_MAPE_raw": high_conf_mape,
            "HighConf MAPE (%)": "N/A" if np.isnan(high_conf_mape) else f"{high_conf_mape:.2f}%",
            "HighConf RMSE (VNĐ)": "N/A" if np.isnan(high_conf_rmse) else f"{high_conf_rmse:,.0f}",
            "Thời gian (s)": f"{train_time:.2f}s"
        })

    # 3. In báo cáo ra Terminal
    results_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("BẢNG SO SÁNH HIỆU NĂNG CÁC THUẬT TOÁN ĐỊNH GIÁ ĐIỆN THOẠI CŨ")
    print("="*70)
    result_cols = ['Mô hình', 'MAPE (%)', 'RMSE (VNĐ)', 'HighConf MAPE (%)', 'HighConf RMSE (VNĐ)', 'Thời gian (s)']
    print(results_df[result_cols].to_string(index=False))
    print("="*70)
    results_df[result_cols].to_csv(MODEL_COMPARISON_RESULTS, index=False, encoding="utf-8-sig")
    print(f"[Done] Đã lưu bảng so sánh vào: {MODEL_COMPARISON_RESULTS}")

    best_name = results_df.sort_values("MAPE_raw").iloc[0]["Mô hình"]
    best_pred = predictions[best_name]
    segment_df = X_test.copy()
    segment_df["real_price"] = np.expm1(y_test).values
    segment_df["pred_price"] = best_pred
    segment_df["ape"] = (segment_df["pred_price"] - segment_df["real_price"]).abs() / segment_df["real_price"]

    print(f"\nMô hình tốt nhất: {best_name}")
    if high_conf_mask.any():
        high_conf_segment = segment_df[high_conf_mask].copy()
        print("\nHigh-confidence evaluation:")
        print(f"- Số mẫu: {len(high_conf_segment)}/{len(segment_df)}")
        print(f"- MAPE: {high_conf_segment['ape'].mean() * 100:.2f}%")
        print(f"- Median price: {high_conf_segment['real_price'].median():,.0f} VNĐ")
        high_conf_segment.to_csv(HIGH_CONFIDENCE_PREDICTIONS, index=False, encoding="utf-8-sig")
        print(f"[Done] Đã lưu mẫu high-confidence vào: {HIGH_CONFIDENCE_PREDICTIONS}")

    for col in ["brand", "lock_status", "condition_score", "warranty_type"]:
        if col not in segment_df.columns:
            continue
        summary = (
            segment_df.groupby(col)
            .agg(n=("ape", "size"), mape=("ape", "mean"), median_price=("real_price", "median"))
            .sort_values("n", ascending=False)
            .head(12)
        )
        summary["mape"] = summary["mape"] * 100
        print(f"\nSai số theo {col}:")
        print(summary.to_string())

    # 4. Sinh và lưu biểu đồ
    print("\n3. Đang xuất biểu đồ trực quan...")
    plot_df = results_df.sort_values('MAPE_raw', ascending=True)
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Đổi màu xanh lá cho mô hình chiến thắng
    colors = ['#2ca02c' if 'Tối ưu' in model else '#1f77b4' for model in plot_df['Mô hình']]
    ax = sns.barplot(x='MAPE_raw', y='Mô hình', data=plot_df, palette=colors)

    plt.title('So sanh Sai so MAPE giua cac Mo hinh AI Dinh gia', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Sai so MAPE (%) - Cang thap cang tot', fontsize=12)
    plt.ylabel('Mo hinh huan luyen', fontsize=12)

    for i, v in enumerate(plot_df['MAPE_raw']):
        ax.text(v + 0.3, i, f"{v:.2f}%", color='black', va='center', fontweight='bold', fontsize=11)

    plt.xlim(0, plot_df['MAPE_raw'].max() + 5)
    plt.tight_layout()

    plt.savefig(MODEL_COMPARISON_CHART, dpi=300)
    print(f"[Done] Đã lưu biểu đồ thành công vào: {MODEL_COMPARISON_CHART}\n")

if __name__ == "__main__":
    run_model_comparison()
