"""Phone dialogue state and catalog facts with exact metadata provenance."""
import json
import re

from phone_assistant.conversation.conversation_state import contains, normalize
from phone_assistant.data.phone_scope import handset_decision
from phone_assistant.retrieval.queries.product_constraints import ProductQueryPlanner, connected_facts, connected_verdict
from phone_assistant.retrieval.queries.query_understanding import BRAND_ALIASES
from phone_assistant.recommendation.recsys_data import read_jsonl, sha256


def capacity_gb(value):
    matches = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(TB|GB|MB)\b", str(value), re.I)
    values = {float(n) * {"TB": 1024, "GB": 1, "MB": 1 / 1024}[u.upper()] for n, u in matches}
    return next(iter(values)) if len(values) == 1 else None


class HandsetFactIndex:
    def __init__(self, split):
        path = split / "products.jsonl"
        manifest = json.loads((split / "manifest.json").read_text(encoding="utf-8"))
        if sha256(path) != manifest["artifacts_sha256"]["products.jsonl"]:
            raise ValueError("Handset metadata changed")
        self.products = {p["parent_asin"]: p for p in read_jsonl(path)}

    def __call__(self, doc):
        return handset_facts(doc, self.products[doc["doc_id"]])


def handset_facts(doc, product):
    facts = connected_facts(doc)
    evidence = {}
    details = product.get("details") or {}
    for field, keys in {"storage_gb": ("Memory Storage Capacity", "Storage Capacity", "Digital Storage Capacity"),
                        "ram_gb": ("RAM", "RAM Memory Installed Size", "Memory RAM")}.items():
        observed = [(key, capacity_gb(details[key])) for key in keys if details.get(key)]
        if observed:
            values = {value for _, value in observed if value is not None}
            facts[field] = next(iter(values)) if len(values) == 1 and all(v is not None for _, v in observed) else None
            evidence[field] = [{"field": f"details.{key}", "quote": str(details[key])} for key, _ in observed]
    if details.get("Battery Capacity"):
        raw = str(details["Battery Capacity"])
        match = re.fullmatch(r"\s*([\d,]+(?:\.\d+)?)\s*(?:mAh|Milliamp Hours?)\s*", raw, re.I)
        facts["battery_mah"] = float(match.group(1).replace(",", "")) if match else None
        evidence["battery_mah"] = [{"field": "details.Battery Capacity", "quote": raw}]
    warnings = []
    ram, storage = facts.get("ram_gb"), facts.get("storage_gb")
    if ram is not None and (ram > 32 or (storage is not None and ram >= storage)):
        warnings.append("ram_capacity_requires_review")
        facts["ram_gb"] = None  # Quarantine a suspicious source value; never guess the correct RAM.
    cpus = []
    for key in ("CPU", "Processor", "Processor Brand", "Processor Speed"):
        if details.get(key):
            cpus.append({"field": f"details.{key}", "quote": str(details[key])})
    # Extract explicit chip names from catalog statements, never from reviews or model-name guesses.
    pattern = r"\b(?:Snapdragon\s+(?:[468]\s+Gen\s+\d|\d{3}[A-Za-z+]*|8\d{2})|Exynos\s+\d{3,5}|Helio\s+[A-Z]\d{1,3}|Apple\s+A\d{1,2}(?:\s+Bionic)?)\b"
    for key in ("features", "description"):
        values = product.get(key) or []
        if isinstance(values, str):
            values = [values]
        for index, value in enumerate(values):
            for match in re.finditer(pattern, str(value), re.I):
                cpus.append({"field": f"{key}[{index}]", "quote": match.group(0)})
    facts["processor_claims"] = list(dict.fromkeys(c["quote"] for c in cpus))
    if cpus:
        evidence["processor_claims"] = cpus
    if handset_decision(doc)[0]:
        facts["product_types"] = ["phone"]
    facts["fact_evidence"] = evidence
    facts["fact_warnings"] = warnings
    facts["source"] = "explicit catalog metadata/title; exact field quotes; no review inference; ambiguous capacities unknown"
    return facts


class HandsetQueryPlanner(ProductQueryPlanner):
    def __call__(self, query, previous=None):
        plan = super().__call__(query, previous)
        constraints = plan["constraints"]
        old = (previous or {}).get("constraints", {}) if not plan["reset"] else {}
        avoided = set(constraints.get("avoid_brands", []))
        for brand, aliases in BRAND_ALIASES.items():
            for alias in aliases:
                if re.search(r"\b(?:khong (?:chon|mua|dung|lay|phai)|avoid|not|exclude)\s+" + re.escape(normalize(alias)) + r"\b", normalize(query)):
                    avoided.add(brand)
        constraints["avoid_brands"] = sorted(avoided)
        constraints["brands"] = [b for b in constraints.get("brands", []) if b not in avoided]
        if avoided & set(old.get("brands", [])) and not constraints["brands"]:
            constraints.pop("model_terms", None)
        # A platform brand already has its own constraint; duplicated maker constraints
        # must not survive a negation or a change of brand.
        maker = constraints.get("manufacturer_names", [])
        if any(any(contains(m, alias) for aliases in BRAND_ALIASES.values() for alias in aliases) for m in maker):
            constraints.pop("manufacturer_names", None)
        constraints.setdefault("product_type", "phone")
        preferences = [] if plan["reset"] else list((previous or {}).get("preferences", []))
        terms = {"gaming": ("choi game", "gaming", "play games"), "camera": ("chup anh", "camera", "photography"),
                 "battery": ("pin trau", "dung ca ngay", "long lasting battery")}
        for name, values in terms.items():
            if any(contains(query, word) for word in values) and name not in preferences:
                preferences.append(name)
        if re.search(r"\b(?:khong (?:choi|can) game|not for gaming)\b", normalize(query)):
            preferences = [p for p in preferences if p != "gaming"]
        extras = [constraints["product_type"]] + constraints.get("brands", []) + constraints.get("manufacturer_names", [])
        if constraints.get("model_terms"):
            extras.append(constraints["model_terms"])
        for field, suffix in (("storage_gb", "GB"), ("ram_gb", "GB RAM"), ("battery_mah", "mAh")):
            if constraints.get(field) is not None:
                extras.append(f"{constraints[field]} {suffix}")
        if constraints.get("network"):
            extras.append(constraints["network"])
        extras.extend({"gaming": "gaming performance", "camera": "camera photography", "battery": "long lasting battery"}[p] for p in preferences)
        plan.update(source="handset_dialogue_v1", constraints=constraints, filters=constraints,
            preferences=preferences, query_rewrite=" ".join(dict.fromkeys([query] + extras)))
        return plan


def dialogue_options(runtime, documents):
    return dict(planner=HandsetQueryPlanner(documents), fact_extractor=HandsetFactIndex(runtime.split), verdict=connected_verdict)
