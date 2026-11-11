"""Query understanding for the Multimodal RAG demo.

The module turns a noisy natural-language request into a compact retrieval plan.
It has a deterministic fallback for offline evaluation and an optional Gemini
path for the chat demo when an API key is available.
"""
from __future__ import annotations

import json
import re
import unicodedata

from phone_assistant.llm.gemini_client import call_gemini


BRAND_ALIASES = {
    "apple": ["apple", "iphone"],
    "samsung": ["samsung", "galaxy"],
    "google": ["google", "pixel"],
    "sony": ["sony", "xperia"],
    "motorola": ["motorola", "moto"],
    "oneplus": ["oneplus"],
    "nokia": ["nokia"],
    "xiaomi": ["xiaomi", "redmi"],
    "oppo": ["oppo"],
    "vivo": ["vivo"],
    "realme": ["realme"],
    "huawei": ["huawei"],
    "lg": ["lg"],
    "blackberry": ["blackberry"],
    "blu": ["blu"],
}


INTENT_RULES = [
    {
        "tag": "battery_good",
        "terms": ["pin", "battery", "pin trau", "lau het pin", "dung ca ngay", "long lasting"],
        "keywords": ["battery", "long lasting", "charger"],
    },
    {
        "tag": "camera_good",
        "terms": ["camera", "chup anh", "chup dep", "photo", "picture", "anh dep"],
        "keywords": ["camera", "photo", "mp", "rear", "front"],
    },
    {
        "tag": "unlocked",
        "terms": ["unlocked", "quoc te", "mo khoa", "gsm", "khong khoa mang"],
        "keywords": ["unlocked", "international", "gsm"],
    },
    {
        "tag": "locked",
        "terms": ["locked", "lock", "khoa mang", "carrier", "verizon", "at&t", "t-mobile"],
        "keywords": ["locked", "carrier", "verizon", "at&t", "t-mobile"],
    },
    {
        "tag": "review_rich",
        "terms": ["review", "danh gia", "rating", "nguoi dung", "nhieu danh gia"],
        "keywords": ["review", "rating", "verified purchase"],
    },
    {
        "tag": "has_image",
        "terms": ["anh", "hinh", "image", "photo", "picture", "anh san pham"],
        "keywords": ["image", "photo", "visual evidence"],
    },
    {
        "tag": "cheap",
        "terms": ["gia re", "re", "cheap", "budget", "tiet kiem"],
        "keywords": ["cheap", "budget", "basic"],
    },
    {
        "tag": "basic_phone",
        "terms": ["nghe goi", "co ban", "basic", "ban phim", "qwerty", "flip"],
        "keywords": ["basic", "keypad", "qwerty", "flip phone"],
    },
    {
        "tag": "good_quality",
        "terms": ["tot", "on", "muot", "ben", "excellent", "great", "good"],
        "keywords": ["good", "excellent", "great", "renewed"],
    },
    {
        "tag": "storage",
        "terms": ["bo nho", "dung luong", "storage", "gb", "memory"],
        "keywords": ["storage", "gb", "memory"],
    },
]

NOISE_TERMS = [
    "toi", "minh", "can", "muon", "tim", "mua", "cho", "dung", "may",
    "dien thoai", "smartphone", "phone", "giup", "voi", "nhe", "nha",
    "me", "bo", "chau", "hoc sinh", "sinh vien",
]


def strip_accents(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def dedupe(items):
    seen = set()
    output = []
    for item in items:
        clean = str(item or "").strip().lower()
        if clean and clean not in seen:
            output.append(clean)
            seen.add(clean)
    return output


def extract_brands(normalized_query: str):
    brands = []
    for brand, aliases in BRAND_ALIASES.items():
        if any(alias in normalized_query for alias in aliases):
            brands.append(brand)
    return brands


def extract_storage(normalized_query: str):
    values = []
    for match in re.finditer(r"\b(8|16|32|64|128|256|512)\s?gb\b|\b1\s?tb\b", normalized_query):
        values.append(match.group(0).replace(" ", ""))
    return values


def deterministic_query_plan(query: str) -> dict:
    normalized = strip_accents(query)
    tags = []
    keywords = []
    reasons = []

    for rule in INTENT_RULES:
        if any(term in normalized for term in rule["terms"]):
            tags.append(rule["tag"])
            keywords.extend(rule["keywords"])
            reasons.append(f"phat hien y dinh {rule['tag']}")

    brands = extract_brands(normalized)
    if brands:
        keywords.extend(brands)
        for brand in brands:
            keywords.extend(BRAND_ALIASES.get(brand, []))
        reasons.append("phat hien hang/model trong truy van")

    storage_values = extract_storage(normalized)
    if storage_values:
        keywords.extend(storage_values)
        tags.append("storage")
        reasons.append("phat hien dung luong bo nho")

    compact = normalized
    for noise in NOISE_TERMS:
        compact = re.sub(rf"\b{re.escape(noise)}\b", " ", compact)
    compact = re.sub(r"\s+", " ", compact).strip()

    if not keywords:
        keywords = compact.split()[:8]

    must_have = []
    nice_to_have = []
    avoid = []
    filters = {}

    if brands:
        filters["brand_preference"] = brands
        must_have.extend(brands)
    if "unlocked" in tags:
        filters["network"] = "unlocked"
        must_have.append("unlocked")
    if "locked" in tags:
        filters["network"] = "locked"
        must_have.append("locked")
    if "review_rich" in tags:
        filters["require_review"] = True
    if "has_image" in tags:
        filters["require_product_image"] = True
    if "cheap" in tags:
        nice_to_have.extend(["cheap", "budget"])
    if "battery_good" in tags:
        nice_to_have.extend(["battery", "long lasting"])
    if "camera_good" in tags:
        nice_to_have.extend(["camera", "photo"])
    if "basic_phone" in tags:
        nice_to_have.extend(["basic", "keypad", "flip"])
        avoid.extend(["case", "cover", "protector"])

    rewritten = " ".join(dedupe([compact] + keywords + must_have + nice_to_have))
    return {
        "source": "rules",
        "original_query": query,
        "normalized_query": normalized,
        "query_rewrite": rewritten,
        "keywords": dedupe(keywords),
        "must_have": dedupe(must_have),
        "nice_to_have": dedupe(nice_to_have),
        "avoid": dedupe(avoid),
        "tags": dedupe(tags),
        "filters": filters,
        "reasoning": dedupe(reasons),
    }


def _parse_json_object(text: str):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(match.group(0) if match else text)
    except json.JSONDecodeError:
        return None


def gemini_query_plan(query: str, timeout=18) -> dict:
    prompt = f"""
Bạn là bộ phận query understanding cho hệ thống Multimodal RAG tư vấn điện thoại.
Nhiệm vụ: loại bỏ nhiễu trong câu hỏi, trích xuất ý định tìm kiếm cốt lõi và tạo query ngắn để retrieval.

Câu hỏi người dùng:
{query}

Trả về JSON hợp lệ, không thêm giải thích ngoài JSON:
{{
  "intent": "product_search|comparison|question_answering",
  "query_rewrite": "câu truy vấn ngắn bằng tiếng Anh/không dấu, giữ từ khóa quan trọng",
  "keywords": ["keyword quan trọng"],
  "must_have": ["điều kiện bắt buộc"],
  "nice_to_have": ["ưu tiên mềm"],
  "avoid": ["điều nên tránh"],
  "tags": ["battery_good|camera_good|unlocked|review_rich|has_image|cheap|basic_phone|good_quality|storage"],
  "filters": {{
    "brand_preference": ["brand nếu có"],
    "require_review": false,
    "require_product_image": false,
    "network": "unlocked|locked|any"
  }},
  "reasoning": ["lý do ngắn gọn"]
}}

Quy tắc:
- Bỏ qua các từ xã giao hoặc bối cảnh không giúp truy xuất.
- Không tự bịa model/brand nếu người dùng không nói.
- Nếu người dùng hỏi pin trâu/dùng cả ngày, thêm tag battery_good.
- Nếu người dùng hỏi chụp đẹp/camera, thêm tag camera_good.
- Nếu người dùng cần quốc tế/mở khóa, thêm tag unlocked.
""".strip()
    result = call_gemini([{"text": prompt}], temperature=0.1, max_output_tokens=700, timeout=timeout, retries=1)
    parsed = _parse_json_object(result.get("text"))
    if not parsed:
        fallback = deterministic_query_plan(query)
        fallback["source"] = "rules_fallback"
        fallback["llm_error"] = result.get("error")
        return fallback

    fallback = deterministic_query_plan(query)
    merged = {
        **fallback,
        **{key: parsed.get(key, fallback.get(key)) for key in (
            "intent", "query_rewrite", "keywords", "must_have", "nice_to_have",
            "avoid", "tags", "filters", "reasoning",
        )},
        "source": "gemini",
        "original_query": query,
    }
    for key in ("keywords", "must_have", "nice_to_have", "avoid", "tags", "reasoning"):
        merged[key] = dedupe(merged.get(key) or [])
    if not isinstance(merged.get("filters"), dict):
        merged["filters"] = fallback["filters"]
    return merged


def understand_query(query: str, use_llm: bool = False, timeout=18) -> dict:
    if use_llm:
        return gemini_query_plan(query, timeout=timeout)
    return deterministic_query_plan(query)

