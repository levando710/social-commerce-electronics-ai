import argparse

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor


from phone_pricing.features.phone_features import CAT_FEATURES, FEATURES
from phone_pricing.common.project_paths import CLEAN_PHONE_DATA, PHONE_OPTUNA_DB, configure_utf8_output, ensure_project_dirs

configure_utf8_output()
import warnings
warnings.filterwarnings('ignore') # Tắt cảnh báo để log cho đẹp

DATA_PATH = CLEAN_PHONE_DATA

# 1. Hàm Objective (Mục tiêu tối ưu hóa của Optuna)
def objective(trial):
    # Đọc dữ liệu
    df = pd.read_csv(DATA_PATH)
    df = df[df['price_outlier'] == 0] 
    
    cat_features = CAT_FEATURES
    features = FEATURES
    
    df[cat_features] = df[cat_features].fillna('Unknown')
    X = df[features]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Định nghĩa không gian tìm kiếm siêu tham số (Hyperparameter Space)
    param = {
        'iterations': trial.suggest_int('iterations', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'cat_features': cat_features,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': False, # Tắt log nội bộ của CatBoost để giao diện gọn gàng
        'random_seed': 42
    }

    # 3. Huấn luyện thử nghiệm
    model = CatBoostRegressor(**param)
    model.fit(
        X_train, y_train, 
        eval_set=(X_test, y_test), 
        early_stopping_rounds=50
    )

    # 4. Tính toán hàm mất mát (Lấy MAPE làm thước đo chính)
    y_pred_log = model.predict(X_test)
    
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    
    # Optuna sẽ cố gắng thu nhỏ giá trị trả về của hàm này (Minimize)
    return mape

# Khởi chạy quá trình dò tìm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tối ưu hyperparameter CatBoost bằng Optuna.")
    parser.add_argument("--data", default=str(CLEAN_PHONE_DATA), help="File dữ liệu sạch.")
    parser.add_argument("--trials", type=int, default=30, help="Số trial Optuna.")
    parser.add_argument("--storage", default=f"sqlite:///{PHONE_OPTUNA_DB.as_posix()}", help="Nơi lưu study để resume.")
    parser.add_argument("--study-name", default="phone_price_catboost", help="Tên study Optuna.")
    args = parser.parse_args()

    ensure_project_dirs()
    DATA_PATH = args.data

    print("Bắt đầu quá trình dò tìm siêu tham số với Optuna...")
    print(f"Quá trình này sẽ chạy thử {args.trials} mô hình khác nhau.\n")
    
    # Tạo study (nhiệm vụ) hướng tới việc tối thiểu hóa (minimize) MAPE
    study = optuna.create_study(
        direction="minimize",
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)

    print("\n" + "="*40)
    print("HOÀN THÀNH TỐI ƯU HÓA!")
    print(f"MAPE tốt nhất đạt được: {study.best_value * 100:.2f}%")
    print("Bộ cấu hình tốt nhất (Best Params):")
    for key, value in study.best_params.items():
        print(f"    '{key}': {value},")
    print("="*40)
    
    print("\nBạn hãy copy bộ Best Params này vào BEST_PARAMS trong file phonetrain.py để train lại mô hình cuối cùng nhé!")
