"""Read the shared layout map when locating immutable research inputs."""

import json
from pathlib import Path


def rag_reference(root: Path, relative: str) -> str:
    config = root / "config/layout.json"
    mapping = json.loads(config.read_text(encoding="utf-8"))["rag"] if config.is_file() else {}
    head, separator, tail = relative.replace("\\", "/").partition("/")
    return "rag/" + mapping.get(head, head) + (separator + tail if separator else "")
