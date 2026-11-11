"""OpenAI chat-completions adapter with the same call shape and result fields as ``call_gemini``.

It exists so the generation-expansion runner can drive another provider without changing its
journal, budget and stop rules.  Usage is mapped to the Gemini field names the runner already
prices (prompt / candidates / thoughts).  There are no automatic retries: every request that is
sent is journaled once by the caller.  The API key is read at call time and never printed.
"""
from __future__ import annotations

import os

import requests

from phone_assistant.llm.gemini_client import get_secret

CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4.1-mini"
SECRET_NAMES = ("OPENAI_API_KEY", "openai_api_key")
ERROR_PREFIX = "OpenAI API"


def openai_model_name():
    return os.environ.get("OPENAI_MODEL") or get_secret("OPENAI_MODEL", "openai_model") or DEFAULT_MODEL


def to_message_content(parts):
    """Translate Gemini-style parts (text / inline_data) into chat-completions content blocks."""
    content = []
    for part in parts:
        if "text" in part:
            content.append({"type": "text", "text": part["text"]})
        elif "inline_data" in part:
            inline = part["inline_data"]
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{inline['mime_type']};base64,{inline['data']}"}})
        else:
            raise ValueError("Unsupported part type")
    return content


def map_usage(usage):
    """Chat-completions usage -> the Gemini usage field names used for pricing."""
    usage = usage or {}
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {"promptTokenCount": int(usage.get("prompt_tokens") or 0),
            "candidatesTokenCount": max(0, completion - reasoning), "thoughtsTokenCount": reasoning,
            "totalTokenCount": int(usage.get("total_tokens") or 0)}


def call_openai(parts, temperature=0.2, max_output_tokens=900, timeout=60, retries=0,
                thinking_budget=None, response_mime_type=None, model=None):
    """Single request; ``retries`` and ``thinking_budget`` exist only for signature compatibility."""
    api_key = get_secret(*SECRET_NAMES)
    if not api_key:
        return {"enabled": False, "text": None, "error": "Chưa tìm thấy OPENAI_API_KEY."}
    model = model or openai_model_name()
    payload = {"model": model, "temperature": temperature, "max_completion_tokens": max_output_tokens,
               "messages": [{"role": "user", "content": to_message_content(parts)}]}
    if response_mime_type == "application/json":
        payload["response_format"] = {"type": "json_object"}
    try:
        response = requests.post(CHAT_COMPLETIONS_URL, headers={"Authorization": f"Bearer {api_key}"},
                                 json=payload, timeout=timeout)
    except requests.RequestException as error:
        return {"enabled": True, "text": None, "model": model,
                "error": f"{ERROR_PREFIX} request failed: {error}".replace(api_key, "[REDACTED]")}
    if response.status_code >= 400:
        return {"enabled": True, "text": None, "model": model,
                "error": f"{ERROR_PREFIX} {response.status_code}: {response.text[:500]}".replace(api_key, "[REDACTED]")}
    try:
        data = response.json()
        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        return {"enabled": True, "text": text or None, "model": model, "model_version": data.get("model"),
                "finish_reason": choice.get("finish_reason"), "usage": map_usage(data.get("usage")),
                "error": None if text else "OpenAI không trả về nội dung text."}
    except (KeyError, IndexError, ValueError) as error:
        return {"enabled": True, "text": None, "model": model, "error": f"{ERROR_PREFIX} bad response: {error}"}
