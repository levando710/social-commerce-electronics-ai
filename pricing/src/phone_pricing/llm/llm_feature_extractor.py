import json
import os
import urllib.error
import urllib.request


LLM_BOOLEAN_FEATURES = [
    "is_zin",
    "is_broken",
    "is_fullbox",
    "is_repaired",
    "has_screen_issue",
    "has_battery_replaced",
    "has_icloud_issue",
    "has_face_id_issue",
]


FEATURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "battery_health": {
            "type": ["integer", "null"],
            "minimum": 50,
            "maximum": 100,
            "description": "Battery health percent if explicitly mentioned, otherwise null.",
        },
        "is_zin": {"type": ["boolean", "null"]},
        "is_broken": {"type": ["boolean", "null"]},
        "is_fullbox": {"type": ["boolean", "null"]},
        "is_repaired": {"type": ["boolean", "null"]},
        "has_screen_issue": {"type": ["boolean", "null"]},
        "has_battery_replaced": {"type": ["boolean", "null"]},
        "has_icloud_issue": {"type": ["boolean", "null"]},
        "has_face_id_issue": {"type": ["boolean", "null"]},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short Vietnamese phrases from the listing that support the extracted fields.",
        },
    },
    "required": [
        "battery_health",
        "is_zin",
        "is_broken",
        "is_fullbox",
        "is_repaired",
        "has_screen_issue",
        "has_battery_replaced",
        "has_icloud_issue",
        "has_face_id_issue",
        "evidence",
    ],
}


def _extract_response_text(payload):
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    chunks = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def _extract_gemini_text(payload):
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""

    parts = candidates[0].get("content", {}).get("parts", [])
    chunks = [part.get("text", "") for part in parts if isinstance(part.get("text"), str)]
    return "\n".join(chunks).strip()


def _json_extraction_prompt(subject, body):
    return f"""
Bạn trích xuất thuộc tính máy điện thoại cũ từ bài rao tiếng Việt.
Với các lỗi hoặc sửa chữa, chỉ đánh true khi có bằng chứng rõ trong text.
Nếu bài rao không nói gì về việc máy hỏng, lỗi, đã sửa, đã thay linh kiện hoặc đã bung máy, hãy xem máy là zin/nguyên bản và trả is_zin = true.
Nếu có bằng chứng máy hỏng, lỗi, đã sửa, đã thay linh kiện hoặc đã bung máy, trả is_zin = false.
Nếu thông tin mâu thuẫn hoặc không đủ rõ, trả null cho trường chưa chắc.
Không suy diễn từ giá, hãng hoặc dòng máy.

Trả về đúng một JSON object có các khóa:
- battery_health: số nguyên 50-100 nếu có phần trăm pin rõ ràng, nếu không thì null
- is_zin: true nếu máy được mô tả zin/nguyên bản/chưa bung/chưa sửa, hoặc không thấy mô tả hỏng/sửa/thay linh kiện; false nếu có dấu hiệu hỏng/sửa/thay linh kiện/bung máy
- is_broken: máy hỏng, vỡ, bán xác, lỗi nặng
- is_fullbox: có fullbox/hộp/phụ kiện đầy đủ
- is_repaired: đã sửa, thay màn, thay kính, ép kính, thay main, qua sửa chữa
- has_screen_issue: lỗi màn hình/cảm ứng như màn vỡ, hỏng màn, sọc, ám, chảy mực, burn in, hở sáng
- has_battery_replaced: đã thay pin/pin linh kiện
- has_icloud_issue: dính iCloud/mất iCloud/ẩn iCloud
- has_face_id_issue: mất/hỏng Face ID
- evidence: mảng các cụm từ ngắn trong bài rao làm bằng chứng

Tiêu đề: {subject}
Mô tả: {body}
""".strip()


def extract_listing_features_with_gemini(subject, body, api_key=None, model=None, timeout=20):
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "Chưa có GEMINI_API_KEY."

    model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _json_extraction_prompt(subject, body)}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"Lỗi Gemini API {exc.code}: {detail[:300]}"
    except Exception as exc:
        return None, f"Lỗi gọi Gemini: {exc}"

    text = _extract_gemini_text(payload)
    if not text:
        return None, "Gemini không trả về JSON."

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"Không parse được JSON từ Gemini: {exc}"


def extract_listing_features_with_llm(subject, body, api_key=None, model=None, timeout=20):
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return extract_listing_features_with_gemini(
            subject,
            body,
            api_key=gemini_key,
            model=os.getenv("GEMINI_MODEL"),
            timeout=timeout,
        )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "Chưa có OPENAI_API_KEY."

    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    listing_text = f"Tiêu đề: {subject}\nMô tả: {body}".strip()
    request_body = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "Bạn trích xuất thuộc tính máy điện thoại cũ từ bài rao tiếng Việt. "
                    "Với các lỗi hoặc sửa chữa, chỉ đánh true khi có bằng chứng rõ trong text. "
                    "Nếu bài rao không nói gì về việc máy hỏng, lỗi, đã sửa, đã thay linh kiện hoặc đã bung máy, "
                    "hãy xem máy là zin/nguyên bản và trả is_zin = true. "
                    "Nếu có bằng chứng máy hỏng, lỗi, đã sửa, đã thay linh kiện hoặc đã bung máy, trả is_zin = false. "
                    "Nếu thông tin mâu thuẫn hoặc không đủ rõ, trả null cho trường chưa chắc. "
                    "Không suy diễn từ giá, hãng hoặc dòng máy."
                ),
            },
            {"role": "user", "content": listing_text},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "phone_listing_features",
                "schema": FEATURE_SCHEMA,
                "strict": True,
            }
        },
        "temperature": 0,
    }

    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return None, f"Lỗi OpenAI API {exc.code}: {detail[:300]}"
    except Exception as exc:
        return None, f"Lỗi gọi LLM: {exc}"

    text = _extract_response_text(payload)
    if not text:
        return None, "LLM không trả về JSON."

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"Không parse được JSON từ LLM: {exc}"


def apply_llm_features(processed_df, llm_features):
    df = processed_df.copy()
    if not llm_features:
        return df

    battery_health = llm_features.get("battery_health")
    if isinstance(battery_health, int) and 50 <= battery_health <= 100:
        df.loc[:, "battery_health"] = float(battery_health)
        df.loc[:, "has_battery_info"] = 1

    for feature in LLM_BOOLEAN_FEATURES:
        value = llm_features.get(feature)
        if isinstance(value, bool):
            df.loc[:, feature] = int(value)

    negative_condition_cols = [
        "is_broken",
        "is_repaired",
        "has_screen_issue",
        "has_battery_replaced",
        "has_icloud_issue",
        "has_face_id_issue",
    ]
    has_negative_condition = any(int(df.iloc[0].get(col, 0)) == 1 for col in negative_condition_cols)
    if not has_negative_condition:
        df.loc[:, "is_zin"] = 1

    return df
