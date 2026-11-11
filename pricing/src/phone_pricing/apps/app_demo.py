import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from catboost import CatBoostRegressor


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from phone_pricing.features.phone_features import FEATURES, preprocess_raw_input
from phone_pricing.models.confidence_utils import build_confidence_profile, score_prediction_confidence
from phone_pricing.llm.llm_feature_extractor import (
    apply_llm_features,
    extract_listing_features_with_gemini,
    extract_listing_features_with_llm,
)
from phone_pricing.common.project_paths import (
    CLEAN_PHONE_DATA,
    PHONE_MODEL,
    PHONE_MODEL_LOWER,
    PHONE_MODEL_UPPER,
    RAW_PHONE_DATA,
)

# --- 1. GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="AI Định giá Điện thoại", layout="wide")
st.title("🤖 Demo AI Định giá Điện thoại Cũ")
st.markdown("Nhập thông tin rao bán y như cách người dùng viết trên Chợ Tốt.")

# Nạp mô hình (Cache để không nạp lại nhiều lần)
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model(str(PHONE_MODEL))
    return model


@st.cache_resource
def load_quantile_models():
    paths = {
        "lower": PHONE_MODEL_LOWER,
        "upper": PHONE_MODEL_UPPER,
    }
    if not all(path.exists() for path in paths.values()):
        return None

    models = {}
    for name, path in paths.items():
        quantile_model = CatBoostRegressor()
        quantile_model.load_model(str(path))
        models[name] = quantile_model
    return models

try:
    model = load_model()
    quantile_models = load_quantile_models()
except Exception as e:
    st.error(f"Không tìm thấy file mô hình '{PHONE_MODEL}'. Vui lòng kiểm tra lại. Lỗi: {e}")
    st.stop()


@st.cache_data
def load_model_catalog():
    source_path = None
    for candidate in [CLEAN_PHONE_DATA, RAW_PHONE_DATA]:
        try:
            df = pd.read_csv(candidate)
            source_path = candidate
            break
        except Exception:
            continue
    else:
        return {"models": {}, "source": None}

    df = df[["brand", "model"]].dropna()
    df["brand_norm"] = df["brand"].astype(str).str.strip().str.lower()
    df["model_norm"] = df["model"].astype(str).str.strip()
    invalid_models = {"", "nan", "unknown", "none", "dòng khác", "dong khac", "khác", "khac"}
    catalog = {}
    for brand_norm, group in df.groupby("brand_norm"):
        valid_models = group["model_norm"].loc[
            lambda s: ~s.str.lower().isin(invalid_models)
        ]
        model_counts = valid_models.value_counts()
        models = sorted(model_counts.index.tolist(), key=lambda name: (-model_counts[name], name.lower()))
        catalog[brand_norm] = models
    return {"models": catalog, "source": source_path}


@st.cache_data
def load_confidence_profile():
    try:
        df = pd.read_csv(CLEAN_PHONE_DATA)
    except Exception:
        return None
    return build_confidence_profile(df)


def get_api_key(key_name):
    try:
        value = st.secrets.get(key_name)
    except Exception:
        value = os.getenv(key_name)
    return value


def get_llm_provider_name():
    if get_api_key("GEMINI_API_KEY"):
        return "Gemini"
    return "OpenAI"


model_catalog_data = load_model_catalog()
model_catalog = model_catalog_data["models"]
confidence_profile = load_confidence_profile()

# Layout chia 2 cột
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin cơ bản (Form nhập liệu)")
    brand = st.selectbox("Hãng", ["Apple", "Samsung", "Xiaomi", "Oppo", "Vivo", "Honor", "Huawei", "Google", "Realme", "Sony", "Khác"])
    brand_key = brand.lower()
    model_options = model_catalog.get(brand_key, [])
    model_choice = st.selectbox("Dòng máy", model_options + ["Khác"], index=0 if model_options else 0)
    st.caption(f"{len(model_options)} dòng máy trong dữ liệu ({model_catalog_data['source'] or 'không có file catalog'}).")
    if model_choice == "Khác":
        custom_model_name = st.text_input("Tên dòng máy khác", "")
        model_name = custom_model_name.strip() or "Khác"
    else:
        model_name = model_choice

    capacity = st.selectbox("Dung lượng", ["< 8GB", "8GB", "16GB", "32GB", "64GB", "128GB", "256GB", "512GB", "1TB", "2TB"], index=6)
    color = st.selectbox("Màu sắc", ["Đen", "Trắng", "Xanh dương", "Xanh lá", "Vàng", "Xám", "Bạc", "Tím", "Hồng", "Đỏ", "Cam", "Màu khác"], index=0)
    origin = st.selectbox("Xuất xứ", ["Việt Nam", "Mỹ", "Trung Quốc", "Hàn Quốc", "Nhật Bản", "Nước khác", "Unknown"])
    lock_status = st.selectbox("Phiên bản", ["Quốc tế", "Lock", "Không rõ"])
    condition = st.selectbox("Tình trạng", ["Đã sử dụng (chưa sửa chữa)", "Đã sử dụng (qua sửa chữa)", "Mới"])
    warranty = st.selectbox("Bảo hành", ["Hết bảo hành", "Còn bảo hành", "1 tháng", "2 tháng", "3 tháng", "4-6 tháng", "7-12 tháng", ">12 tháng", "Bảo hành hãng"])

with col2:
    st.subheader("Nội dung bài đăng (Tự do)")
    subject = st.text_input("Tiêu đề bài đăng", "Bán xác iPhone 13 Pro Max hỏng màn")
    body = st.text_area("Mô tả chi tiết", "Máy trần, rơi vỡ hỏng màn hình, mất face ID, pin zin 85%. Ai mua xác về lấy linh kiện thì nhắn.")
    use_llm = st.checkbox("Dùng LLM đọc mô tả", value=False)
    if use_llm:
        st.caption(f"Provider ưu tiên: {get_llm_provider_name()}")
    
    predict_btn = st.button("Dự đoán giá bán 🚀", type="primary", use_container_width=True)

# Xử lý dự đoán
if predict_btn:
    raw_input = {
        'brand': brand, 'model': model_name, 'capacity': capacity, 
        'color': color, 'origin': origin, 'condition': condition, 
        'lock_status': lock_status, 'warranty': warranty, 'subject': subject, 'body': body
    }
    
    # Tiền xử lý
    processed_df = preprocess_raw_input(raw_input)
    llm_features = None
    if use_llm:
        gemini_api_key = get_api_key("GEMINI_API_KEY")
        if gemini_api_key:
            llm_features, llm_error = extract_listing_features_with_gemini(
                subject,
                body,
                api_key=gemini_api_key,
            )
        else:
            llm_features, llm_error = extract_listing_features_with_llm(
                subject,
                body,
                api_key=get_api_key("OPENAI_API_KEY"),
            )
        if llm_error:
            st.warning(llm_error)
        else:
            processed_df = apply_llm_features(processed_df, llm_features)
    
    # Đảm bảo đúng thứ tự cột như lúc train
    X_pred = processed_df[FEATURES]
    
    # Dự đoán
    pred_log = model.predict(X_pred)
    pred_price = np.expm1(pred_log)[0]
    
    st.divider()
    st.subheader("Kết quả định giá:")
    
    if quantile_models:
        lower_bound = np.expm1(quantile_models["lower"].predict(X_pred))[0]
        upper_bound = np.expm1(quantile_models["upper"].predict(X_pred))[0]
        lower_bound = min(lower_bound, pred_price)
        upper_bound = max(upper_bound, pred_price)
        interval_note = "Khoảng giá học trực tiếp từ quantile models."
    else:
        # Fallback khi chưa train quantile models: dùng biên độ dựa trên MAPE 24.55%.
        mape_margin = 0.2455
        lower_bound = pred_price * (1 - mape_margin)
        upper_bound = pred_price * (1 + mape_margin)
        interval_note = "Khoảng giá dùng biên độ MAPE vì chưa có quantile models."
    
    st.success(f"### Giá dự đoán trung tâm: {pred_price:,.0f} VNĐ")
    st.info(f"**Khoảng giá giao dịch hợp lý:** {lower_bound:,.0f} VNĐ  --- đến ---  {upper_bound:,.0f} VNĐ")
    st.caption(interval_note)

    if confidence_profile:
        confidence = score_prediction_confidence(X_pred, confidence_profile)
        st.write(f"**Độ tin cậy:** {confidence['label']}")
        st.caption(
            f"Model có {confidence['model_count']} mẫu trong dữ liệu sạch; "
            f"cùng hãng-model có {confidence['brand_model_count']} mẫu."
        )
        st.write("Lý do:", ", ".join(confidence["reasons"]))
    
    st.write("---")
    st.write("Dữ liệu AI đã bóc tách được (sau tiền xử lý):")
    st.dataframe(X_pred)
    if llm_features:
        st.write("LLM đã đọc được:")
        st.json(llm_features)
