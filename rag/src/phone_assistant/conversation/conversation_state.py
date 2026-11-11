"""Bounded conversational state and explicit, conservative catalog constraints."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import re
import secrets
import threading
import time
import unicodedata

from phone_assistant.retrieval.queries.query_understanding import BRAND_ALIASES, INTENT_RULES


def normalize(text):
    value = unicodedata.normalize("NFKD", str(text).lower()).replace("đ", "d")
    return "".join(char for char in value if not unicodedata.combining(char))


def contains(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", normalize(text)) is not None


def parse_turn(query):
    text = normalize(query)
    brands, avoided = [], []
    for brand, aliases in BRAND_ALIASES.items():
        if any(contains(text, alias) for alias in aliases):
            if any(re.search(r"(?:khong (?:chon|dung|lay|mua)|bo|avoid|not)\s+" + re.escape(alias) + r"\b", text)
                   for alias in aliases):
                avoided.append(brand)
            else:
                brands.append(brand)
    updates = {}
    if brands:
        updates["brands"] = brands
    if avoided:
        updates["avoid_brands"] = avoided
    model = re.search(r"\b(?:iphone\s*\d{1,2}(?:\s+(?:pro|max|mini|plus)){0,2}|galaxy\s+(?:[sa]\d{1,2}|note\s*\d{1,2})|pixel\s*\d[a-z]?)\b", text)
    if model:
        updates["model_terms"] = model.group(0)
    ram = re.search(r"\bram\s*(\d{1,3})\s*gb\b|\b(\d{1,3})\s*gb\s*ram\b", text)
    storage_text = text
    if ram:
        updates["ram_gb"] = int(ram.group(1) or ram.group(2))
        storage_text = text[:ram.start()] + text[ram.end():]
    storage = re.search(r"\b(8|16|32|64|128|256|512|1024)\s*gb\b|\b([12])\s*tb\b", storage_text)
    if storage:
        updates["storage_gb"] = int(storage.group(1)) if storage.group(1) else int(storage.group(2)) * 1024
    battery = re.search(r"\b(\d{3,5})\s*mah\b", text)
    if battery:
        updates["battery_mah"] = int(battery.group(1))
    budget = re.search(r"(?:duoi|toi da|under|budget)\s*(\d+(?:[.,]\d+)?)\s*(trieu|tr|vnd|usd|dollars?)\b", text)
    if budget:
        multiplier = 1000000 if budget.group(2) in ("trieu", "tr") else 1
        updates["max_price"] = float(budget.group(1).replace(",", ".")) * multiplier
        updates["price_currency"] = "VND" if budget.group(2) in ("trieu", "tr", "vnd") else "USD"
    if any(contains(text, term) for term in ("unlocked", "quoc te", "mo khoa", "khong khoa mang")):
        updates["network"] = "unlocked"
    elif any(contains(text, term) for term in ("locked", "lock", "khoa mang")):
        updates["network"] = "locked"
    if any(contains(text, term) for term in ("co anh", "co hinh", "has image", "with image")):
        updates["require_image"] = True
    if any(contains(text, term) for term in ("khong can anh", "khong can hinh", "image optional")):
        updates["require_image"] = False
    tags = [rule["tag"] for rule in INTENT_RULES
            if any(contains(text, term) for term in rule["terms"])]
    if updates.get("network") == "unlocked":
        tags = [tag for tag in tags if tag != "locked"]
    return updates, tags


def next_plan(query, previous=None):
    previous = previous or {}
    reset = any(contains(query, phrase) for phrase in ("tim moi", "bat dau lai", "new search", "reset"))
    constraints = {} if reset else dict(previous.get("constraints") or {})
    updates, tags = parse_turn(query)
    if updates.get("brands") and set(updates["brands"]) != set(constraints.get("brands", [])):
        constraints.pop("model_terms", None)
    constraints.update(updates)
    if updates.get("brands"):
        constraints["avoid_brands"] = [brand for brand in constraints.get("avoid_brands", []) if brand not in updates["brands"]]
    if updates.get("avoid_brands"):
        constraints["brands"] = [brand for brand in constraints.get("brands", []) if brand not in updates["avoid_brands"]]
    old_tags = [] if reset else previous.get("tags", [])
    merged_tags = list(dict.fromkeys(old_tags + tags))
    if constraints.get("network") == "unlocked":
        merged_tags = [tag for tag in merged_tags if tag != "locked"]
    elif constraints.get("network") == "locked":
        merged_tags = [tag for tag in merged_tags if tag != "unlocked"]
    terms = []
    for brand in constraints.get("brands", []):
        terms.extend(BRAND_ALIASES[brand][:2])
    if constraints.get("storage_gb"):
        terms.append(f"{constraints['storage_gb']}GB")
    if constraints.get("model_terms"):
        terms.append(constraints["model_terms"])
    if constraints.get("network"):
        terms.append(constraints["network"])
    for key, suffix in (("ram_gb", "GB RAM"), ("battery_mah", "mAh")):
        if constraints.get(key):
            terms.append(f"{constraints[key]} {suffix}")
    for rule in INTENT_RULES:
        if rule["tag"] in merged_tags and rule["tag"] not in ("locked", "unlocked", "storage"):
            terms.extend(rule["keywords"][:2])
    # Preserve explicit model names from the current request as well as state.
    rewrite = " ".join(dict.fromkeys(terms + [query]))
    focus = []
    if any(contains(query, phrase) for phrase in ("may dau tien", "cai dau tien", "first one")):
        focus = previous.get("last_result_ids", [])[:1]
    elif any(contains(query, phrase) for phrase in ("hai may tren", "so sanh hai may", "compare them")):
        focus = previous.get("last_result_ids", [])[:2]
    return {"source": "conversation_rules_v1", "original_query": query,
            "query_rewrite": rewrite, "constraints": constraints, "tags": merged_tags,
            "reset": reset, "filters": constraints, "focus_doc_ids": [] if reset else focus}


def catalog_facts(doc):
    details = doc.get("details") or {}
    title = str(doc.get("title") or "")
    brand_source = " ".join(str(details.get(key) or "") for key in ("Brand", "Manufacturer")) + " " + title
    brands = [brand for brand, aliases in BRAND_ALIASES.items() if any(contains(brand_source, alias) for alias in aliases)]
    storage_source = str(details.get("Memory Storage Capacity") or "")
    capacities = re.findall(r"\b(\d+(?:\.\d+)?)\s*(GB|TB)\b", storage_source, re.I)
    if not capacities:
        capacities = re.findall(r"\b(\d+(?:\.\d+)?)\s*(GB|TB)\b(?!\s*RAM)", title, re.I)
    values = {int(float(value) * (1024 if unit.lower() == "tb" else 1)) for value, unit in capacities}
    network_source = title + " " + str(details.get("Wireless Carrier") or "")
    says_locked = contains(network_source, "not unlocked") or contains(network_source, "locked")
    says_unlocked = contains(network_source, "unlocked") and not contains(network_source, "not unlocked")
    # Title and carrier field can disagree (e.g. "Locked to AT&T" vs "Unlocked for All Carriers");
    # conflicting explicit evidence stays unverified instead of favouring one side.
    if says_locked and says_unlocked:
        network = None
    elif says_unlocked:
        network = "unlocked"
    elif says_locked:
        network = "locked"
    else:
        network = None
    ram = re.search(r"\bram\s*(\d{1,3})\s*gb\b|\b(\d{1,3})\s*gb\s*ram\b", normalize(title))
    for key in ("RAM", "RAM Memory Installed Size", "Memory RAM"):
        match = re.search(r"\b(\d+)\s*gb\b", str(details.get(key) or ""), re.I)
        if match:
            ram_value = int(match.group(1))
            break
    else:
        ram_value = int(ram.group(1) or ram.group(2)) if ram else None
    battery = re.search(r"\b(\d{3,5})\s*mah\b", title + " " + str(details.get("Battery Capacity") or ""), re.I)
    return {"brands": brands, "storage_gb": next(iter(values)) if len(values) == 1 else None,
            "network": network, "has_image": bool(doc.get("product_image_paths")),
            "ram_gb": ram_value, "battery_mah": int(battery.group(1)) if battery else None,
            "price": None, "price_currency": None,
            "model_text": normalize(title + " " + str(details.get("Model Name") or "")),
            "source": "catalog title and explicit technical metadata; no review inference"}


def constraint_verdict(facts, constraints):
    failures, unknown = [], []
    if constraints.get("brands"):
        if not facts["brands"]:
            unknown.append("brand")
        elif not set(constraints["brands"]) & set(facts["brands"]):
            failures.append("brand")
    if set(constraints.get("avoid_brands", [])) & set(facts["brands"]):
        failures.append("avoid_brand")
    if constraints.get("model_terms") and not contains(facts["model_text"], constraints["model_terms"]):
        failures.append("model")
    for field in ("storage_gb", "network", "ram_gb", "battery_mah"):
        if constraints.get(field) is not None:
            if facts[field] is None:
                unknown.append(field)
            elif facts[field] != constraints[field]:
                failures.append(field)
    if constraints.get("max_price") is not None:
        if facts.get("price") is None or facts.get("price_currency") != constraints.get("price_currency"):
            unknown.append("price_in_requested_currency")
        elif facts["price"] > constraints["max_price"]:
            failures.append("price")
    if constraints.get("require_image") and not facts["has_image"]:
        failures.append("image")
    return {"eligible": not failures and not unknown, "violations": failures, "unknown": unknown}


@dataclass
class Conversation:
    user_index: int | None
    visitor_id: str | None = None
    plan: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    updated: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, max_sessions=256, ttl_seconds=1800):
        self.sessions = {}
        self.guard = threading.RLock()
        self.max_sessions, self.ttl_seconds = max_sessions, ttl_seconds

    def reset_visitor(self, visitor_id):
        with self.guard:
            self.sessions = {key: value for key, value in self.sessions.items() if value.visitor_id != visitor_id}

    @contextmanager
    def transaction(self, session_id=None, user_index=None, visitor_id=None):
        created = not session_id
        with self.guard:
            now = time.monotonic()
            self.sessions = {key: value for key, value in self.sessions.items()
                             if now - value.updated < self.ttl_seconds}
            if session_id:
                if not isinstance(session_id, str) or session_id not in self.sessions:
                    raise ValueError("Session expired or unknown; start a new conversation.")
                session = self.sessions[session_id]
                if session.visitor_id != visitor_id:
                    raise ValueError("Conversation belongs to another visitor; start a new conversation.")
                if user_index is not None and user_index != session.user_index:
                    raise ValueError("Start a new conversation to change the demo profile.")
            else:
                if len(self.sessions) >= self.max_sessions:
                    raise ValueError("Conversation capacity reached; retry later.")
                session_id = secrets.token_urlsafe(24)
                session = Conversation(user_index=user_index, visitor_id=visitor_id)
                self.sessions[session_id] = session
            session.updated = now
        with session.lock:
            try:
                yield session_id, session
                session.updated = time.monotonic()
            except Exception:
                if created:
                    with self.guard:
                        self.sessions.pop(session_id, None)
                raise
