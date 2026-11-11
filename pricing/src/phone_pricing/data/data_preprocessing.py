
import pandas as pd


from phone_pricing.features.phone_features import FINAL_COLUMNS, add_phone_features, is_non_phone_listing
from phone_pricing.common.project_paths import CLEAN_PHONE_DATA, RAW_PHONE_DATA, configure_utf8_output, ensure_project_dirs

configure_utf8_output()


def process_raw_data(input_file=RAW_PHONE_DATA, output_file=CLEAN_PHONE_DATA):
    ensure_project_dirs()
    print(f"Đang đọc dữ liệu thô từ {input_file}...")
    df = pd.read_csv(input_file)

    before_rows = len(df)
    if "ad_id" in df.columns:
        df = df.drop_duplicates("ad_id", keep="first")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].notna() & (df["price"] > 0)].copy()
    df["price"] = df["price"].astype(int)

    non_phone_mask = df.apply(is_non_phone_listing, axis=1)
    non_phone_count = int(non_phone_mask.sum())
    if non_phone_count:
        df = df[~non_phone_mask].copy()

    print("Đang chuẩn hóa dữ liệu và bóc tách đặc trưng điện thoại...")
    df = add_phone_features(df)

    df_clean = df[FINAL_COLUMNS]
    df_clean.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n[Done] Đã lưu dữ liệu sạch vào: {output_file}")
    print(f"- Tổng số dòng thô: {before_rows}")
    print(f"- Tổng số dòng sau bỏ trùng: {len(df_clean)}")
    print(f"- Số dòng không phải điện thoại di động đã loại: {non_phone_count}")
    print(f"- Outliers phát hiện: {df_clean['price_outlier'].sum()}")
    print(f"- Số máy có pin được bóc tách: {df_clean['has_battery_info'].sum()}")
    print(f"- Số máy hỏng/xác/icloud bắt được: {df_clean['is_broken'].sum()}")
    print(f"- Số máy có bảo hành: {df_clean['has_warranty'].sum()}")
    print(f"- Số dòng dung lượng >= 1TB: {(df_clean['capacity_gb'] >= 1024).sum()}")


if __name__ == "__main__":
    process_raw_data()
