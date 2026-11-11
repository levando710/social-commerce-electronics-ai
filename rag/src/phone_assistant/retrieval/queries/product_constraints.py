"""Conservative accessory/device constraints: maker differs from compatible platform."""
from __future__ import annotations

import re

from phone_assistant.conversation.conversation_state import catalog_facts, constraint_verdict, contains, next_plan, normalize, parse_turn
from phone_assistant.retrieval.queries.query_understanding import BRAND_ALIASES

PASSIVE = {"case", "screen_protector", "mount"}
COMPATIBLE = PASSIVE | {"charger", "cable", "headphones"}
PATTERNS = {
    "screen_protector": r"\b(?:screen protector|tempered glass|cuong luc|mieng dan)\b",
    "case": r"\b(?:case|cases|cover|pouch|holster|op lung|bao da)\b",
    "power_bank": r"\b(?:power bank|portable charger|sac du phong)\b",
    "charger": r"\b(?:charger|charging adapter|wall adapter|wireless charging|bo sac|cu sac)\b",
    "cable": r"\b(?:cable|cord|charging wire|cap sac|day sac)\b",
    "mount": r"\b(?:mount|phone holder|phone stand|gia do)\b",
    "headphones": r"\b(?:headphones|earbuds|earphones|headset|tai nghe)\b",
    "battery": r"\b(?:(?:replacement|spare) battery|battery for|pin thay the)\b",
    "storage_device": r"\b(?:flash drive|memory card|micro sd|microsd|ssd|hard drive|the nho)\b",
}


def product_types(text):
    text = normalize(text)
    text = re.sub(r"\bin case\b", "", text)
    types = [kind for kind, pattern in PATTERNS.items() if re.search(pattern, text)]
    if types:
        # "case-friendly screen protector" is not itself a phone case.
        if "screen_protector" in types and re.search(r"\bcase[ -]friendly\b", text):
            types.remove("case")
        return types
    if re.search(r"\b(?:laptop|notebook|macbook)\b", text):
        return ["laptop"]
    if re.search(r"\b(?:tablet|ipad|may tinh bang)\b", text):
        return ["tablet"]
    if re.search(r"\b(?:phone|smartphone|iphone|galaxy|pixel|nokia|dien thoai)\b", text):
        return ["phone"]
    return []


def model_names(text):
    value = normalize(text).replace("-", " ")
    patterns = (r"\biphone\s*\d{1,2}(?:\s*(?:pro\s*max|pro|mini|plus))?\b",
                r"\bgalaxy\s*(?:s\d{1,2}|a\d{1,2}|note\s*\d{1,2})(?:\s*(?:ultra|plus|fe|\+))?\b",
                r"\bpixel\s*\d{1,2}(?:\s*(?:pro|xl|a))?\b")
    result = []
    for pattern in patterns:
        for match in re.finditer(pattern, value):
            model = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", match.group(0))
            model = re.sub(r"\s+", " ", model).strip()
            model = model.replace("promax", "pro max")
            result.append(model)
    return sorted(set(result))


def brands_in(text):
    return [brand for brand, aliases in BRAND_ALIASES.items() if any(contains(text, alias) for alias in aliases)]


def connectors(text):
    text = normalize(text)
    return [key for key, pattern in {"usb_c": r"\b(?:usb[ -]?c|type[ -]?c)\b", "lightning": r"\blightning\b",
        "micro_usb": r"\bmicro[ -]?usb\b", "usb_a": r"\busb[ -]?a\b"}.items() if re.search(pattern, text)]


def battery_requirement(text):
    value = normalize(text)
    number = r"(?:\d{1,3}(?:,\d{3})+|\d{3,5})"
    match = re.search(r"(?<![\d.,])(" + number + r")(?:\s*(?:to|[-–]|den)\s*(" + number + r"))?\s*mah\b", value)
    if not match:
        return {}
    first = int(match.group(1).replace(",", ""))
    if match.group(2):
        second = int(match.group(2).replace(",", ""))
        return {"battery_min_mah": min(first, second), "battery_max_mah": max(first, second)}
    prefix = value[max(0, match.start() - 32):match.start()]
    if re.search(r"(?:at least|minimum|min|toi thieu|it nhat)\s*$", prefix):
        return {"battery_min_mah": first}
    if re.search(r"(?:over|more than|above|hon|tren)\s*$", prefix):
        return {"battery_min_mah": first + 1}
    if re.search(r"(?:under|less than|below|duoi)\s*$", prefix):
        return {"battery_max_mah": first - 1}
    if re.search(r"(?:at most|maximum|max|toi da)\s*$", prefix):
        return {"battery_max_mah": first}
    if re.search(r"(?:around|about|approximately|khoang)\s*$", prefix):
        return {}  # Approximate capacity is a preference; do not invent a tolerance.
    return {"battery_mah": first}


def connected_facts(doc):
    facts = catalog_facts(doc)
    details = doc.get("details") or {}
    title = str(doc.get("title") or "")
    kinds = product_types(title)
    maker = str(details.get("Brand") or details.get("Manufacturer") or "").strip()
    facts["brands"] = brands_in(maker) if maker else []
    facts["manufacturer_name"] = normalize(maker) if maker else None
    facts["product_types"] = kinds
    compatibility = " ".join(str(details.get(k) or "") for k in ("Compatible Phone Models", "Compatible Devices"))
    if set(kinds) & COMPATIBLE:
        # The title is an explicit product claim, not a review-derived compatibility guess.
        compatibility += " " + title
    facts["compatibility_models"] = model_names(compatibility)
    facts["compatibility_brands"] = brands_in(compatibility)
    facts["connectors"] = connectors(" ".join(str(details.get(k) or "") for k in ("Connector Type", "Connectivity Technology")) + " " + title)
    battery = battery_requirement(title + " " + str(details.get("Battery Capacity") or ""))
    facts["battery_mah"] = battery.get("battery_mah")
    if set(kinds) & PASSIVE:
        for key in ("storage_gb", "ram_gb", "battery_mah", "network"):
            facts[key] = None  # Device names on a passive cover are not its own hardware.
    facts["source"] = "explicit product title/specs; manufacturer and compatibility separated; no review inference"
    return facts


class ProductQueryPlanner:
    def __init__(self, documents):
        self.makers = sorted({normalize((d.get("details") or {}).get("Brand") or "") for d in documents}
            - {"", "generic", "unknown", "good", "new", "case", "phone", "mobile", "smartphone", "one", "any", "just", "need", "for", "from", "with", "clear", "black", "blue"}, key=len, reverse=True)

    def __call__(self, query, previous=None):
        plan = next_plan(query, previous)
        constraints = plan["constraints"]
        old = (previous or {}).get("constraints", {}) if not plan["reset"] else {}
        types = product_types(query)
        # Device names in an accessory follow-up retain the accessory context.
        explicit_accessory = [kind for kind in types if kind in PATTERNS]
        device_word = re.search(r"\b(?:phone|smartphone|dien thoai|tablet|laptop|notebook)\b", normalize(query))
        for_word = re.search(r"\b(?:for|cho)\b", normalize(query))
        explicit_device = device_word and (for_word is None or device_word.start() < for_word.start())
        existing_device_context = re.search(r"\b(?:something|anything|accessor\w*|for my|protect my|using my|carry my|with my|on my|cho dien thoai cua)\b", normalize(query))
        if explicit_accessory:
            constraints["product_type"] = explicit_accessory[0]
        elif types and not existing_device_context and (explicit_device or not (old.get("product_type") in COMPATIBLE and not plan["reset"])):
            constraints["product_type"] = types[0]
        if constraints.get("product_type") != old.get("product_type"):
            constraints.pop("manufacturer_names", None)
            constraints.pop("connectors", None)
        kind = constraints.get("product_type")
        if re.search(r"\bmah\b", normalize(query)):
            for key in ("battery_mah", "battery_min_mah", "battery_max_mah", "compatibility_battery_mah"):
                constraints.pop(key, None)
            constraints.update(battery_requirement(query))
        updates, _ = parse_turn(query)
        models = model_names(query)
        platform = updates.get("brands", [])
        manufacturers = [m for m in self.makers if len(m) >= 3 and contains(query, m)
                         and (not brands_in(m) or kind not in COMPATIBLE or contains(query, "by " + m) or contains(query, m + " branded"))]
        if manufacturers:
            constraints["manufacturer_names"] = [manufacturers[0]]
        if kind in COMPATIBLE:
            if platform:
                if set(platform) != set(old.get("compatibility_brands", [])):
                    constraints.pop("compatibility_models", None)
                constraints["compatibility_brands"] = platform
            if models:
                constraints["compatibility_models"] = models
            constraints.pop("brands", None); constraints.pop("model_terms", None)
            if kind in PASSIVE:
                for key in ("storage_gb", "ram_gb", "battery_mah", "network"):
                    if key in constraints:
                        constraints["compatibility_" + key] = constraints.pop(key)
        else:
            for key in list(constraints):
                if key.startswith("compatibility_"):
                    constraints.pop(key)
        if connectors(query):
            constraints["connectors"] = connectors(query)
        # Preserve the user's words; append only carried explicit constraints.
        extra = [kind or ""] + constraints.get("compatibility_models", []) + constraints.get("manufacturer_names", [])
        plan.update(source="product_constraints_v1", query_rewrite=" ".join(dict.fromkeys([query] + [x for x in extra if x])),
                    constraints=constraints, filters=constraints)
        return plan


def connected_verdict(facts, constraints):
    result = constraint_verdict(facts, constraints)
    unknown, failures = result["unknown"], result["violations"]
    kind = constraints.get("product_type")
    if kind:
        if not facts["product_types"]:
            unknown.append("product_type")
        elif kind not in facts["product_types"]:
            failures.append("product_type")
    for key in ("manufacturer_names", "compatibility_models", "compatibility_brands", "connectors"):
        required = constraints.get(key)
        if not required:
            continue
        observed = [facts["manufacturer_name"]] if key == "manufacturer_names" and facts["manufacturer_name"] else facts.get(key, [])
        if not observed:
            unknown.append(key)
        else:
            matches = set(required) <= set(observed) if key == "connectors" else bool(set(required) & set(observed))
            if not matches:
                failures.append(key)
    for key in constraints:
        if key in ("compatibility_storage_gb", "compatibility_ram_gb", "compatibility_battery_mah", "compatibility_network"):
            unknown.append(key)
    for key, direction in (("battery_min_mah", "min"), ("battery_max_mah", "max")):
        if key in constraints:
            if facts["battery_mah"] is None:
                unknown.append(key)
            elif (direction == "min" and facts["battery_mah"] < constraints[key]) or (direction == "max" and facts["battery_mah"] > constraints[key]):
                failures.append(key)
    return {"eligible": not failures and not unknown, "violations": failures, "unknown": unknown}
