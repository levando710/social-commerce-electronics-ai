"""Determine image type from file bytes, including downloaded .img files."""
from PIL import Image
from phone_assistant.common.project_layout import asset_path


def image_mime(path):
    with Image.open(asset_path(path)) as image:
        mime = Image.MIME.get(image.format)
    if not mime or not mime.startswith("image/"):
        raise ValueError("Unsupported image format")
    return mime
