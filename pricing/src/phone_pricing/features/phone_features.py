import re
import unicodedata

import numpy as np
import pandas as pd


CAT_FEATURES = ["brand", "model", "color", "origin", "lock_status", "warranty_type"]
NUM_FEATURES = [
    "capacity_gb",
    "condition_score",
    "warranty_months",
    "has_warranty",
    "battery_health",
    "has_battery_info",
    "is_zin",
    "is_broken",
    "is_fullbox",
    "is_repaired",
    "has_screen_issue",
    "has_battery_replaced",
    "has_icloud_issue",
    "has_face_id_issue",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
FINAL_COLUMNS = [
    "ad_id",
    "price",
    "log_price",
    "price_outlier",
    "brand",
    "model",
    "capacity_gb",
    "color",
    "origin",
    "lock_status",
    "condition_score",
    "warranty_months",
    "has_warranty",
    "warranty_type",
    "battery_health",
    "has_battery_info",
    "is_zin",
    "is_broken",
    "is_fullbox",
    "is_repaired",
    "has_screen_issue",
    "has_battery_replaced",
    "has_icloud_issue",
    "has_face_id_issue",
]

VALID_CAPACITIES_GB = {2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}


def strip_accents(value):
    text = unicodedata.normalize("NFD", str(value))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_text(value, default="unknown"):
    if value is None or pd.isna(value):
        return default
    text = " ".join(str(value).strip().split())
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return default
    return text


def normalize_category(value):
    return normalize_text(value).lower()


def parse_capacity(value):
    text = normalize_text(value, default="").lower().replace(",", ".")
    if not text:
        return np.nan
    if "<" in text and "8" in text:
        return 4.0

    match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|t|gb|g)?", text)
    if not match:
        return np.nan

    number = float(match.group(1))
    unit = match.group(2) or "gb"
    capacity = number * 1024 if unit in {"tb", "t"} else number
    return capacity if 0 < capacity <= 2048 else np.nan


def parse_capacity_from_text(value):
    text = normalize_text(value, default="").lower().replace(",", ".")
    if not text:
        return np.nan

    candidates = []

    for match in re.finditer(r"\b\d{1,2}\s*[/\\-]\s*(\d{2,4})\s*(?:gb|g|tb)?\b", text):
        number = float(match.group(1))
        if number in VALID_CAPACITIES_GB:
            candidates.append(number)

    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(tb|gb|g)\b", text):
        number = float(match.group(1))
        unit = match.group(2)
        capacity = number * 1024 if unit == "tb" else number
        if capacity in VALID_CAPACITIES_GB:
            candidates.append(capacity)

    return max(candidates) if candidates else np.nan


def parse_condition(value):
    text = normalize_category(value)
    if "mới" in text or "moi" in strip_accents(text):
        return 2
    if "qua sửa" in text or "qua sua" in strip_accents(text):
        return 0
    return 1


def parse_warranty(value, context_text=""):
    warranty_text = normalize_category(value)
    ascii_text = strip_accents(warranty_text)
    context = strip_accents(normalize_text(context_text, default="").lower())

    if "hết" in warranty_text or "het" in ascii_text:
        return 0, 0, "none"
    if "bảo hành hãng" in warranty_text or "bao hanh hang" in ascii_text:
        active_patterns = [
            r"con bh",
            r"con bao hanh",
            r"bh chinh hang",
            r"bao hanh chinh hang",
            r"bao hanh.*(?:den|toi|t\d{1,2}|20\d{2}|\d+\s*th)",
        ]
        if any(re.search(pattern, context) for pattern in active_patterns):
            return 1, 1, "official_active"
        return 0, 0, "official_policy"
    if ">12" in warranty_text or "trên 12" in warranty_text or "tren 12" in ascii_text:
        return 13, 1, "long"

    range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*th", ascii_text)
    if range_match:
        return int(range_match.group(2)), 1, "shop"

    match = re.search(r"(\d+)\s*th", ascii_text)
    if match:
        return int(match.group(1)), 1, "shop"

    if "còn bảo hành" in warranty_text or "con bao hanh" in ascii_text:
        return 1, 1, "unknown_remaining"

    return 0, 0, "unknown"


def parse_battery(text):
    normalized = normalize_text(text, default="").lower()
    if not normalized:
        return 0, 0

    pin_match = re.search(r"pin\s*[:\-\s]?\s*(\d{2,3})\s*%", normalized)
    if pin_match:
        return min(int(pin_match.group(1)), 100), 1

    pin_x_match = re.search(r"pin\s*[:\-\s]?\s*(\d)\s*x", normalized)
    if pin_x_match:
        return int(pin_x_match.group(1)) * 10 + 5, 1

    pin_number_match = re.search(r"pin\s*[:\-\s]?\s*(\d{2,3})(?!\s*(?:mah|w))", normalized)
    if pin_number_match:
        value = int(pin_number_match.group(1))
        if 50 <= value <= 100:
            return value, 1

    return 0, 0


def parse_lock_status(value, text=""):
    explicit = normalize_category(value)
    combined = f"{explicit} {normalize_text(text, default='').lower()}"
    ascii_text = strip_accents(combined)

    if "quốc tế" in combined or "quoc te" in ascii_text or "qt" in ascii_text:
        return "international"
    if re.search(r"\block\b", ascii_text) or "khóa mạng" in combined or "khoa mang" in ascii_text:
        return "locked"
    return "unknown"


def has_any_pattern(text, patterns):
    return int(any(re.search(pattern, text) for pattern in patterns))


def build_text_combo(row):
    return f"{normalize_text(row.get('subject', ''), default='')} {normalize_text(row.get('body', ''), default='')}".lower()


def is_non_phone_listing(row):
    text = strip_accents(build_text_combo(row))
    brand = strip_accents(normalize_category(row.get("brand", "")))
    model = strip_accents(normalize_category(row.get("model", "")))

    landline_patterns = [
        r"dien thoai ban",
        r"\bpanasonic\b",
        r"\bgrandstream\b",
        r"\bgxp\d+",
    ]
    if any(re.search(pattern, text) for pattern in landline_patterns):
        return True
    if brand in {"panasonic", "grandstream"} or model in {"panasonic", "grandstream"}:
        return True
    return False


def is_unknown_model(value):
    text = normalize_category(value)
    ascii_text = strip_accents(text)
    return text in {"unknown", "dòng khác", "khác"} or ascii_text in {"dong khac", "khac", "unknown"}


def infer_common_model_from_text(text):
    normalized = strip_accents(normalize_text(text, default="").lower())
    patterns = [
        r"\biphone\s*(\d{1,2})\s*(pro max|pro|plus|mini)?\b",
        r"\bgalaxy\s*s\s*(\d{1,2})\s*(ultra|plus|fe)?\b",
        r"\bgalaxy\s*note\s*(\d{1,2})\s*(ultra|plus)?\b",
        r"\b(?:galaxy\s*)?z\s*fold\s*(\d{1,2})\b",
        r"\b(?:galaxy\s*)?z\s*flip\s*(\d{1,2})\b",
        r"\bredmi\s*note\s*(\d{1,2})\s*(pro max|pro|plus)?\b",
        r"\bredmi\s*k\s*(\d{2,3})\s*(pro)?\b",
        r"\bpoco\s*(x|f|m)\s*(\d{1,2})\s*(pro)?\b",
        r"\bpixel\s*(\d{1,2})\s*(pro xl|pro|a)?\b",
        r"\bfind\s*x\s*(\d{1,2})\s*(pro|ultra)?\b",
        r"\breno\s*(\d{1,2})\s*(pro|f)?\b",
        r"\bvivo\s*v\s*(\d{1,2})\s*(pro)?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        words = [part for part in match.group(0).split() if part]
        return " ".join(words)
    return None


def infer_models_from_text(df, min_count=10):
    if "model" not in df.columns:
        return df

    model_counts = df["model"].apply(normalize_text).value_counts()
    candidates = []
    for model, count in model_counts.items():
        normalized_model = strip_accents(str(model).lower()).strip()
        if count < min_count or is_unknown_model(model):
            continue
        # Numeric-only model labels like "12" or "15" are valid for some brands
        # when provided by the API, but they are too ambiguous to infer from text.
        if re.fullmatch(r"\d{1,3}", normalized_model):
            continue
        candidates.append(model)
    candidates = sorted(candidates, key=len, reverse=True)
    normalized_candidates = [(model, strip_accents(model.lower())) for model in candidates]

    def infer(row):
        current = row.get("model", "unknown")
        if not is_unknown_model(current):
            return current
        text = strip_accents(build_text_combo(row))
        for model, normalized_model in normalized_candidates:
            if normalized_model and re.search(rf"(?<!\w){re.escape(normalized_model)}(?!\w)", text):
                return model
        return infer_common_model_from_text(text) or current

    df = df.copy()
    df["model"] = df.apply(infer, axis=1)
    return df


def add_phone_features(df):
    df = df.copy()

    text_cols = ["subject", "body", "capacity", "color", "brand", "model", "origin", "condition", "warranty", "lock_status"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda value: normalize_text(value, default="unknown"))

    df = infer_models_from_text(df)

    for col in ["brand", "model", "color", "origin"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].apply(normalize_category)

    capacity_source = df.get("capacity", pd.Series(index=df.index, dtype=object)).apply(parse_capacity)
    capacity_from_text = df.apply(build_text_combo, axis=1).apply(parse_capacity_from_text)
    df["capacity_gb"] = capacity_source.combine_first(capacity_from_text)
    median_capacity = df["capacity_gb"].dropna().median()
    df["capacity_gb"] = df["capacity_gb"].fillna(128.0 if pd.isna(median_capacity) else median_capacity)

    text_combo = df.apply(build_text_combo, axis=1)
    df["lock_status"] = [
        parse_lock_status(value, text)
        for value, text in zip(df.get("lock_status", pd.Series(index=df.index, dtype=object)), text_combo)
    ]

    df["condition_score"] = df.get("condition", pd.Series(index=df.index, dtype=object)).apply(parse_condition)

    warranty_values = [
        parse_warranty(value, text)
        for value, text in zip(df.get("warranty", pd.Series(index=df.index, dtype=object)), text_combo)
    ]
    warranty_values = pd.Series(warranty_values, index=df.index)
    df["warranty_months"] = warranty_values.apply(lambda item: item[0]).astype(int)
    df["has_warranty"] = warranty_values.apply(lambda item: item[1]).astype(int)
    df["warranty_type"] = warranty_values.apply(lambda item: item[2]).astype(str)

    battery_values = text_combo.apply(parse_battery)
    df["battery_health"] = battery_values.apply(lambda item: item[0]).astype(float)
    df["has_battery_info"] = battery_values.apply(lambda item: item[1]).astype(int)

    zin_patterns = [r"\bzin\b", r"chưa bung", r"chua bung", r"chưa sửa", r"chua sua", r"nguyên bản", r"nguyen ban"]
    broken_patterns = [
        r"hỏng",
        r"\bhong\b",
        r"vỡ",
        r"\bvo\b",
        r"vo man",
        r"nứt",
        r"\bnut\b",
        r"liệt",
        r"\bliet\b",
        r"sọc",
        r"\bsoc\b",
        r"chết nguồn",
        r"chet nguon",
        r"mất face",
        r"mat face",
        r"ám màn",
        r"am man",
        r"\bxác\b",
        r"icloud",
        r"icould",
    ]
    box_patterns = [r"fullbox", r"full box", r"còn hộp", r"con hop", r"đầy đủ phụ kiện", r"day du phu kien"]
    screen_issue_patterns = [
        r"hỏng màn",
        r"hong man",
        r"màn hỏng",
        r"man hong",
        r"vỡ màn",
        r"vo man",
        r"màn vỡ",
        r"man vo",
        r"bể màn",
        r"be man",
        r"màn bể",
        r"man be",
        r"nứt màn",
        r"nut man",
        r"màn nứt",
        r"man nut",
        r"sọc màn",
        r"soc man",
        r"màn sọc",
        r"man soc",
        r"ám màn",
        r"am man",
        r"màn ám",
        r"man am",
        r"chảy mực",
        r"chay muc",
        r"loang mực",
        r"loang muc",
        r"burn\s*in",
        r"lưu ảnh",
        r"luu anh",
        r"hở sáng",
        r"ho sang",
        r"đen màn",
        r"den man",
        r"màn đen",
        r"man den",
        r"liệt cảm ứng",
        r"liet cam ung",
    ]
    battery_replaced_patterns = [r"thay pin", r"pin thay", r"pin linh kiện", r"pin linh kien"]
    repaired_patterns = [r"thay pin", r"pin thay", r"ép kính", r"ep kinh", r"thay kính", r"thay man", r"thay màn", r"qua sửa", r"qua sua", r"main sửa", r"main sua"]
    face_id_patterns = [r"face\s*id", r"faceid", r"mất face", r"mat face"]

    ascii_text_combo = text_combo.apply(strip_accents)
    combined_text = text_combo + " " + ascii_text_combo
    df["is_zin"] = combined_text.apply(lambda text: has_any_pattern(text, zin_patterns)).astype(int)
    df["is_broken"] = combined_text.apply(lambda text: has_any_pattern(text, broken_patterns)).astype(int)
    df["is_fullbox"] = combined_text.apply(lambda text: has_any_pattern(text, box_patterns)).astype(int)
    df["is_repaired"] = combined_text.apply(lambda text: has_any_pattern(text, repaired_patterns)).astype(int)
    df["has_screen_issue"] = combined_text.apply(lambda text: has_any_pattern(text, screen_issue_patterns)).astype(int)
    df["has_battery_replaced"] = combined_text.apply(lambda text: has_any_pattern(text, battery_replaced_patterns)).astype(int)
    df["has_icloud_issue"] = combined_text.str.contains(r"icloud|icould", regex=True).astype(int)
    df["has_face_id_issue"] = combined_text.apply(lambda text: has_any_pattern(text, face_id_patterns)).astype(int)

    df["log_price"] = np.log1p(df["price"])
    lower_bound = df["price"].quantile(0.005)
    upper_bound = df["price"].quantile(0.995)
    df["price_outlier"] = ((df["price"] < lower_bound) | (df["price"] > upper_bound)).astype(int)

    return df


def preprocess_raw_input(raw_data):
    row = {
        "ad_id": 0,
        "price": 0,
        "subject": raw_data.get("subject", ""),
        "body": raw_data.get("body", ""),
        "capacity": raw_data.get("capacity", ""),
        "color": raw_data.get("color", ""),
        "brand": raw_data.get("brand", ""),
        "model": raw_data.get("model", ""),
        "origin": raw_data.get("origin", ""),
        "lock_status": raw_data.get("lock_status", ""),
        "condition": raw_data.get("condition", ""),
        "warranty": raw_data.get("warranty", ""),
    }
    return add_phone_features(pd.DataFrame([row]))[FEATURES]
