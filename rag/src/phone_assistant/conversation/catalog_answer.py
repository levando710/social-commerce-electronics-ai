"""Deterministic catalog summaries for offline use; not LLM generation or quality scores."""
import math


def number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        return None
    return f"{value:g}"


def catalog_answer(results, plan=None):
    if not results:
        return "Chưa có đủ dữ liệu đã xác minh để tìm điện thoại đáp ứng các điều kiện này. Bạn có thể bỏ bớt điều kiện hoặc bắt đầu tìm mới."
    lines = ["Theo dữ liệu sản phẩm hiện có:"]
    for row in results[:3]:
        facts = row.get("catalog_facts") or {}
        parts = []
        for key, name, unit in (("storage_gb", "Bộ nhớ trong", "GB"), ("ram_gb", "RAM", "GB"), ("battery_mah", "Pin", "mAh")):
            if key == "ram_gb" and "ram_capacity_requires_review" in facts.get("fact_warnings", []):
                continue
            value = number(facts.get(key))
            if value:
                parts.append(f"{name}: {value} {unit}")
        if facts.get("network") == "unlocked":
            parts.append("Nguồn ghi không khóa mạng")
        elif facts.get("network") == "locked":
            parts.append("Nguồn ghi khóa mạng")
        if "ram_capacity_requires_review" in facts.get("fact_warnings", []):
            parts.append("RAM chưa được xác minh, thông số trong nguồn cần kiểm tra lại")
        title = str(row.get("title") or "Điện thoại").replace("\n", " ")[:240]
        citation = f"[{row['doc_id']}]" if row.get("doc_id") else ""
        lines.append(f"- {title} {citation}" + (" — " + "; ".join(parts) if parts else ""))
    if "gaming" in (plan or {}).get("preferences", []):
        lines.append("Chưa có phép đo hiệu năng để xác nhận mức phù hợp khi chơi game.")
    lines.append("Thông tin trên là mô tả từ nguồn catalog, chưa xác nhận giá bán hiện tại hoặc tình trạng của một máy cụ thể.")
    return "\n".join(lines)


def answer_policy(query, results, *, allow_llm=False, requested=True, catalog_enabled=True, plan=None, history=None):
    if allow_llm and requested and results:
        from phone_assistant.llm.gemini_client import generate_answer
        response = generate_answer(query, results, history=history, timeout=25, retries=0)
        if response.get("answer"):
            return {**response, "mode": "llm", "attempted": True}
        if not catalog_enabled:
            return {**response, "mode": "generation_failed", "attempted": True}
        return {"answer": catalog_answer(results, plan), "enabled": False, "mode": "catalog_template_fallback",
                "attempted": True, "error": response.get("error"), "model": response.get("model")}
    return {"answer": catalog_answer(results, plan) if catalog_enabled else None,
            "enabled": False, "attempted": False, "mode": "catalog_template" if catalog_enabled else "disabled"}
