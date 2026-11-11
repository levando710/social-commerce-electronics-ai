"""Gemini client for RAG answer generation.

The key is read at runtime from environment variables or local secret files.
The module never prints the key.
"""
from phone_assistant.common.project_layout import MODULE_ROOT
from phone_assistant.common.project_layout import rebind_paths
import base64
import json
from phone_assistant.common.image_mime import image_mime
import os
from pathlib import Path
import re
import time

import requests


ROOT = MODULE_ROOT
DEFAULT_SECRET_FILES = (
    ROOT / ".env",
    ROOT / ".streamlit" / "secrets.toml",
    ROOT.parent / "pricing" / ".streamlit" / "secrets.toml",
)
GENERATION_THINKING_BUDGET = 0
GENERATED_ANSWER_TOKEN_LIMIT = 1024
DEFAULT_GENERATION_RETRIES = 0


def parse_key_value_file(path):
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def get_secret(*names):
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for path in DEFAULT_SECRET_FILES:
        values = parse_key_value_file(path)
        for name in names:
            if values.get(name):
                return values[name]
    return None


def clean_text(value):
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def image_part(path):
    image_path = Path(rebind_paths(str(path)))
    if not image_path.exists() or not image_path.is_file():
        return None
    mime_type = image_mime(image_path)
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": mime_type, "data": data}}


def gemini_model_name():
    return (
        os.environ.get("GEMINI_MODEL")
        or get_secret("GEMINI_MODEL", "gemini_model")
        or "gemini-2.5-flash"
    )


def call_gemini(parts, temperature=0.2, max_output_tokens=900, timeout=35, retries=3,
                thinking_budget=None, response_mime_type=None):
    api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY", "google_api_key", "gemini_api_key")
    if not api_key:
        return {
            "enabled": False,
            "text": None,
            "error": "Chưa tìm thấy GEMINI_API_KEY hoặc GOOGLE_API_KEY.",
        }

    model = gemini_model_name()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if thinking_budget is not None and model.startswith("gemini-2.5"):
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    if response_mime_type is not None:
        payload["generationConfig"]["responseMimeType"] = response_mime_type
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, params={"key": api_key}, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc).replace(api_key, "[REDACTED]")
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
                continue
            return {"enabled": True, "text": None, "error": last_error, "model": model}
        if response.status_code >= 400:
            last_error = f"Gemini API {response.status_code}: {response.text[:500]}".replace(api_key, "[REDACTED]")
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 8)
                time.sleep(delay)
                continue
            return {
                "enabled": True,
                "text": None,
                "error": last_error,
                "model": model,
            }
        try:
            data = response.json()
            text_parts = []
            for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
                if part.get("text"):
                    text_parts.append(part["text"])
            text = "\n".join(text_parts).strip()
            return {
                "enabled": True,
                "text": text or None,
                "error": None if text else "Gemini không trả về nội dung text.",
                "model": model,
                "usage": data.get("usageMetadata", {}),
                "model_version": data.get("modelVersion"),
            }
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            return {"enabled": True, "text": None, "error": str(exc), "model": model}
    return {"enabled": True, "text": None, "error": last_error, "model": model}


def build_prompt(question, results, history=None):
    lines = [
        "Bạn là trợ lý RAG cho demo đồ án Multimodal RAG về điện thoại.",
        "Chỉ trả lời dựa trên CONTEXT được cung cấp. Nếu thiếu dữ kiện, nói rõ là dữ liệu chưa đủ.",
        "Trả lời bằng tiếng Việt, ngắn gọn, có nhận xét thực tế, không bịa giá hoặc thông số ngoài context.",
        "Sau câu trả lời, thêm mục 'Nguồn evidence' liệt kê 2-3 sản phẩm đã dùng.",
        "Viện dẫn [doc_id] cho mỗi nhận định về sản phẩm. Không suy ra pin/camera/tình trạng từ ảnh catalog.",
        "Lịch sử chỉ giúp hiểu câu hỏi; mọi thông số phải được xác nhận từ CONTEXT hiện tại.",
        "",
        f"Câu hỏi: {question}",
        "",
    ]
    lines.append("Fields marked null or listed in fact_warnings are NOT verified. Do not present suspicious RAM values from raw source quotes as confirmed hardware; say the specification needs checking. Numeric source references refer only to the ordered contexts below.")
    if history:
        lines.append("HISTORY (conversation context only):")
        for turn in history[-6:]:
            lines.append(f"User: {str(turn.get('query') or '')[:1500]}")
            lines.append(f"Assistant: {str(turn.get('answer') or '')[:1500]}")
    lines.append("CONTEXT (authoritative evidence for this answer):")
    for idx, item in enumerate(results[:5], 1):
        lines.extend([
            f"[{idx}] [{item.get('doc_id')}] {item.get('title')}",
            f"Brand: {item.get('brand')}; Rating: {item.get('average_rating')}; Review count: {item.get('review_count')}",
            f"Evidence: {clean_text(item.get('evidence'))}",
            f"Verified catalog fields: {json.dumps(item.get('catalog_facts') or {}, ensure_ascii=False)}",
            "",
        ])
    return "\n".join(lines)


def generate_answer(question, results, include_images=True, timeout=35, history=None, retries=DEFAULT_GENERATION_RETRIES):
    parts = [{"text": build_prompt(question, results, history=history)}]
    if include_images:
        added = 0
        for item in results[:3]:
            for path in item.get("product_images") or []:
                part = image_part(path)
                if part:
                    parts.append({"text": f"Ảnh catalog [{item.get('doc_id')}]: {item.get('title')}"})
                    parts.append(part)
                    added += 1
                    break
            if added >= 2:
                break
    result = call_gemini(parts, temperature=0.25, max_output_tokens=GENERATED_ANSWER_TOKEN_LIMIT,
                         timeout=timeout, retries=retries, thinking_budget=GENERATION_THINKING_BUDGET)
    raw_answer = result.get("text")
    answer, citation_audit = normalize_answer_citations(raw_answer, results)
    return {
        "enabled": result.get("enabled"),
        "answer": answer,
        "raw_answer": raw_answer,
        "citation_audit": citation_audit,
        "error": result.get("error"),
        "model": result.get("model"),
        "usage": result.get("usage"),
    }


def normalize_answer_citations(answer, results):
    """Resolve provider numeric references to the exact ordered contexts it received."""
    if answer is None:
        return None, {"normalized": 0, "unresolved_numeric": []}
    normalized, unresolved = 0, set()
    def replace(match):
        nonlocal normalized
        index = int(match.group(1))
        if 1 <= index <= min(5, len(results)):
            normalized += 1
            return "[" + results[index - 1]["doc_id"] + "]"
        unresolved.add(index)
        return match.group(0)
    text = re.sub(r"\[(\d{1,2})\]", replace, answer)
    return text, {"normalized": normalized, "unresolved_numeric": sorted(unresolved)}


def generate_image_analysis(image_path, product_title="", product_brand="", timeout=20, retries=2):
    prompt = f"""
Bạn là mô hình phân tích ảnh sản phẩm điện thoại cho hệ thống Multimodal RAG.
Hãy chỉ dựa trên ảnh được cung cấp và thông tin tiêu đề nếu có.

Tiêu đề sản phẩm: {product_title}
Hãng/brand: {product_brand}

Trả về JSON hợp lệ, không thêm giải thích ngoài JSON, theo schema:
{{
  "caption_vi": "mô tả ngắn bằng tiếng Việt về ảnh",
  "caption_en": "short English caption",
  "visible_product_type": "phone|phone_accessory|unknown",
  "visible_brand_or_model": "nếu nhìn thấy hoặc suy ra từ tiêu đề, ngược lại null",
  "visual_tags": ["tag ngắn liên quan đến ảnh"],
  "condition_tags": ["new_like", "used", "scratched", "cracked_screen", "unknown_condition"],
  "risk_tags": ["possible_accessory_only", "not_enough_visual_evidence", "screen_issue", "body_damage"],
  "evidence": ["chi tiết nhìn thấy trong ảnh hoặc tiêu đề"]
}}

Quy tắc:
- Không khẳng định lỗi/hỏng nếu ảnh không cho thấy rõ.
- Nếu ảnh chỉ là phụ kiện, đặt visible_product_type là "phone_accessory".
- Nếu không đủ bằng chứng, dùng "unknown" hoặc "not_enough_visual_evidence".
""".strip()
    part = image_part(image_path)
    if not part:
        return {
            "enabled": False,
            "analysis": None,
            "raw_text": None,
            "error": f"Không đọc được ảnh: {image_path}",
        }
    result = call_gemini(
        [{"text": prompt}, part],
        temperature=0.1,
        max_output_tokens=700,
        timeout=timeout,
        retries=retries,
    )
    raw_text = result.get("text")
    analysis = None
    error = result.get("error")
    if raw_text:
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        try:
            analysis = json.loads(match.group(0) if match else raw_text)
        except json.JSONDecodeError as exc:
            error = f"Không parse được JSON từ Gemini: {exc}"
    return {
        "enabled": result.get("enabled"),
        "analysis": analysis,
        "raw_text": raw_text,
        "error": error,
        "model": result.get("model"),
    }
