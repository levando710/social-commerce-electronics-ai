import pandas as pd


POPULAR_BRANDS = {"apple", "samsung", "xiaomi", "oppo", "vivo"}
UNKNOWN_MODELS = {"unknown", "dòng khác", "khác", "nan", ""}


def normalize_value(value):
    if value is None or pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def build_confidence_profile(df):
    clean = df.copy()
    clean["brand"] = clean["brand"].apply(normalize_value)
    clean["model"] = clean["model"].apply(normalize_value)
    if "price_outlier" in clean.columns:
        clean = clean[clean["price_outlier"].eq(0)]
    return {
        "brand_counts": clean["brand"].value_counts().to_dict(),
        "model_counts": clean["model"].value_counts().to_dict(),
        "brand_model_counts": clean.groupby(["brand", "model"]).size().to_dict(),
        "total_rows": int(len(clean)),
    }


def high_confidence_mask(X, profile, *, min_model_count=30, min_brand_model_count=10):
    frame = X.copy()
    brand = frame["brand"].apply(normalize_value)
    model = frame["model"].apply(normalize_value)

    model_count = model.map(lambda value: profile["model_counts"].get(value, 0))
    brand_model_count = [
        profile["brand_model_counts"].get((brand_value, model_value), 0)
        for brand_value, model_value in zip(brand, model)
    ]
    brand_model_count = pd.Series(brand_model_count, index=frame.index)

    return (
        brand.isin(POPULAR_BRANDS)
        & ~model.isin(UNKNOWN_MODELS)
        & (model_count >= min_model_count)
        & (brand_model_count >= min_brand_model_count)
        & frame["lock_status"].apply(normalize_value).ne("unknown")
        & frame["condition_score"].ne(0)
        & ~frame["warranty_type"].apply(normalize_value).isin({"official_policy", "none"})
        & frame["is_broken"].eq(0)
        & frame["has_icloud_issue"].eq(0)
        & frame["has_face_id_issue"].eq(0)
    )


def score_prediction_confidence(row, profile):
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    brand = normalize_value(row.get("brand"))
    model = normalize_value(row.get("model"))
    model_count = profile["model_counts"].get(model, 0)
    brand_model_count = profile["brand_model_counts"].get((brand, model), 0)

    score = 0
    reasons = []

    if brand in POPULAR_BRANDS:
        score += 2
        reasons.append("hãng phổ biến")
    else:
        reasons.append("hãng ít dữ liệu")

    if model not in UNKNOWN_MODELS and model_count >= 30:
        score += 3
        reasons.append(f"model có {model_count} mẫu")
    elif model not in UNKNOWN_MODELS and model_count >= 10:
        score += 1
        reasons.append(f"model chỉ có {model_count} mẫu")
    else:
        reasons.append("model ít dữ liệu hoặc chưa rõ")

    if brand_model_count >= 10:
        score += 1
    else:
        reasons.append("ít mẫu cùng hãng và model")

    if normalize_value(row.get("lock_status")) != "unknown":
        score += 1
        reasons.append("có thông tin quốc tế/lock")
    else:
        reasons.append("thiếu thông tin quốc tế/lock")

    if normalize_value(row.get("warranty_type")) not in {"official_policy", "none"}:
        score += 1
        reasons.append("bảo hành rõ hơn")
    else:
        reasons.append("bảo hành mơ hồ hoặc không có")

    risk_flags = []
    if int(row.get("condition_score", 1)) == 0:
        risk_flags.append("đã qua sửa chữa")
    if int(row.get("is_broken", 0)) == 1:
        risk_flags.append("có dấu hiệu hỏng/xác")
    if int(row.get("has_icloud_issue", 0)) == 1:
        risk_flags.append("có dấu hiệu iCloud")
    if int(row.get("has_face_id_issue", 0)) == 1:
        risk_flags.append("có lỗi Face ID")

    if risk_flags:
        score -= 3
        reasons.extend(risk_flags)
    else:
        score += 1
        reasons.append("không thấy lỗi nặng trong mô tả")

    if score >= 7:
        label = "Cao"
    elif score >= 4:
        label = "Trung bình"
    else:
        label = "Thấp"

    return {
        "label": label,
        "score": int(score),
        "model_count": int(model_count),
        "brand_model_count": int(brand_model_count),
        "reasons": reasons,
    }
