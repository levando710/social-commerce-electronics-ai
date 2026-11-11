"""Assign an electronics family and role from catalog categories (no title guessing).

The older title-rule detector in ``product_constraints.product_types`` is left untouched so every frozen
experiment keeps its behaviour.  This module reads the explicit Amazon category path instead:
level 2 gives the product family, and the deepest level says whether the item is a device, an accessory,
a component or a service.  Products without a usable path stay ``unassigned``; nothing is inferred from
text here.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

FAMILY_BY_LEVEL2 = {
    "Computers & Accessories": "computers",
    "Camera & Photo": "camera",
    "Television & Video": "tv_video",
    "Video Projectors": "tv_video",
    "Headphones, Earbuds & Accessories": "headphones",
    "Home Audio": "home_audio",
    "Portable Audio & Video": "portable_audio",
    "Car & Vehicle Electronics": "car_electronics",
    "Vehicle Electronics Accessories": "car_electronics",
    "Security & Surveillance": "security",
    "GPS, Finders & Accessories": "gps",
    "Wearable Technology": "wearables",
    "eBook Readers & Accessories": "ebook_readers",
    "Power Accessories": "power",
    "Household Batteries, Chargers & Accessories": "power",
    "Accessories & Supplies": "general_accessories",
    "AmazonBasics Accessories": "general_accessories",
    "Amazon Device Accessories": "general_accessories",
    "Gaming Accessories": "gaming",
    "PC Gaming": "gaming",
    "Electronics Warranties": "services",
    "Service Plans": "services",
}
ACCESSORY_FAMILIES = {"general_accessories", "power"}
SERVICE = re.compile(r"warrant|service plan|protection plan|trade-?in", re.I)
ACCESSORY = re.compile(r"accessor|cases?\b|bags?\b|adapters?|cables?|mounts?|cord|tripods?|monopods?|stands?|"
                       r"lighting|chargers?|straps?|covers?|screen protectors?|replacement|supplies", re.I)
COMPONENT = re.compile(r"components?|parts\b", re.I)


def category_path(product):
    path = [str(part).strip() for part in (product.get("categories") or []) if str(part).strip()]
    return path[1:] if path and path[0].lower() == "electronics" else path


def category_type(product):
    """-> {"family", "role", "path", "source"}; ``family`` is ``unassigned`` when no category path exists."""
    path = category_path(product)
    if not path:
        return {"family": "unassigned", "role": "unknown", "path": [], "source": "no_categories"}
    family = FAMILY_BY_LEVEL2.get(path[0], "other")
    below = " ".join(path[1:])  # levels under the family; the family name itself ("... & Accessories") says nothing
    if SERVICE.search(" ".join(path)) or family == "services":
        role = "service"
    elif COMPONENT.search(below):
        role = "component"
    elif ACCESSORY.search(below) or family in ACCESSORY_FAMILIES:
        role = "accessory"
    else:
        role = "device"
    return {"family": family, "role": role, "path": path, "source": "category_path"}


def schema_from_data(products, min_coverage=0.3, min_items=20, generic=frozenset({
        "Date First Available", "Item Weight", "Manufacturer", "Product Dimensions", "Is Discontinued By Manufacturer",
        "Best Sellers Rank", "Package Dimensions", "Item Dimensions  LxWxH", "Item Dimensions LxWxH",
        "Country of Origin", "Item model number", "ASIN"})):
    """Per family, the structured fields present in at least ``min_coverage`` of its products (data-driven proposal)."""
    members = defaultdict(list)
    for product in products:
        members[category_type(product)["family"]].append(product)
    schema = {}
    for family, items in sorted(members.items()):
        if len(items) < min_items:
            continue
        counts = Counter(key for item in items for key, value in (item.get("details") or {}).items()
                         if str(value).strip() and key not in generic)
        schema[family] = {"items": len(items),
                          "attributes": {key: round(count / len(items), 3) for key, count in counts.most_common()
                                         if count / len(items) >= min_coverage}}
    return schema


def coverage_report(products, rule_types=None):
    """How many products receive a family/role, and (optionally) how the rule detector agrees on devices."""
    assigned = [category_type(p) for p in products]
    report = {"items": len(products),
              "with_family": sum(a["family"] not in ("unassigned",) for a in assigned),
              "families": dict(Counter(a["family"] for a in assigned).most_common()),
              "roles": dict(Counter(a["role"] for a in assigned).most_common())}
    if rule_types is not None:
        pairs = Counter()
        for product, assignment in zip(products, assigned):
            kinds = rule_types(product) or ["(none)"]
            pairs[(kinds[0], assignment["family"], assignment["role"])] += 1
        report["rule_type_vs_category"] = [{"rule_type": k[0], "family": k[1], "role": k[2], "items": n}
                                           for k, n in pairs.most_common(40)]
        report["rule_detector_silent"] = sum(n for (rule, _, _), n in pairs.items() if rule == "(none)")
    return report
