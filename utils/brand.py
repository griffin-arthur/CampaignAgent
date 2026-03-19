"""Load brand configuration files for injection into agent system prompts."""

import json
from pathlib import Path

BRAND_DIR = Path(__file__).resolve().parent.parent / "brand"


def load_brand_file(filename: str) -> str:
    """Load a brand file as a string."""
    path = BRAND_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Brand file not found: {path}")
    return path.read_text()


def load_brand_json(filename: str) -> dict:
    """Load a brand JSON file as a dict."""
    return json.loads(load_brand_file(filename))


def load_voice_guide() -> str:
    return load_brand_file("voice_guide.md")


def load_email_templates() -> str:
    return load_brand_file("email_templates.md")


def load_blog_style_guide() -> str:
    return load_brand_file("blog_style_guide.md")


def load_social_guidelines() -> str:
    return load_brand_file("social_guidelines.md")


def load_link_targets() -> dict:
    return load_brand_json("link_targets.json")


def load_utm_conventions() -> dict:
    return load_brand_json("utm_conventions.json")
