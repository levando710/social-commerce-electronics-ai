"""Explicit catalog spec hierarchy; never infer hardware facts from reviews or photos."""
from __future__ import annotations

import re

SPEC_GROUPS = {
    "identity": ("Brand", "Manufacturer", "Item model number", "Model Name"),
    "compute": ("CPU", "Processor", "Processor Brand", "Processor Speed", "RAM", "RAM Memory Installed Size", "Memory RAM"),
    "storage": ("Memory Storage Capacity", "Storage Capacity", "Digital Storage Capacity", "Hard Drive"),
    "display": ("Screen Size", "Display Size", "Resolution", "Display Resolution Maximum"),
    "power": ("Battery Capacity", "Batteries", "Input Voltage", "Output Voltage", "Total USB Ports"),
    "connectivity": ("Wireless Carrier", "Cellular Technology", "Wireless network technology",
                     "Connectivity technologies", "Connectivity Technology", "Connector Type"),
    "physical_and_compatibility": ("Color", "Material", "Form Factor", "Compatible Phone Models",
                                   "Compatible Devices", "Water Resistance Level", "Other camera features"),
}


def normalized(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def explicit_specs(product):
    details = product.get("details") or {}
    result = []
    for group, keys in SPEC_GROUPS.items():
        for key in keys:
            value = details.get(key)
            if value is not None and str(value).strip():
                attribute = normalized(key)
                token = f"{attribute}={normalized(value)}"
                result.append((token, attribute, group))
    return sorted(set(result))


def catalog_text(product):
    parts = [str(product.get("title") or ""), str(product.get("store") or "")]
    for key in ("features", "description", "categories"):
        values = product.get(key) or []
        parts.extend([values] if isinstance(values, str) else map(str, values))
    parts.extend(f"{attribute}: {token.split('=', 1)[1]}" for token, attribute, _ in explicit_specs(product))
    return "\n".join(part for part in parts if part.strip())


def make_hierarchy(products):
    specs, value_to_attribute, attribute_to_group = {}, {}, {}
    for product in products:
        triples = explicit_specs(product)
        specs[product["parent_asin"]] = [token for token, _, _ in triples]
        for token, attribute, group in triples:
            value_to_attribute[token] = attribute
            attribute_to_group[attribute] = group
    return specs, {"value_to_attribute": value_to_attribute, "attribute_to_group": attribute_to_group,
                   "policy": "fixed explicit catalog field allowlist; group -> attribute -> value; no inferred facts"}
