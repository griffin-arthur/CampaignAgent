"""Deterministic UTM link generation — no LLM involved."""

from urllib.parse import urlencode

from slugify import slugify


def generate_utm(
    campaign_name: str,
    source: str,
    medium: str,
    content: str,
    base_url: str = "https://www.arthur.ai",
) -> str:
    """Build a full URL with UTM parameters.

    All values are slugified for consistency (lowercase, hyphens, no special chars).
    """
    params = {
        "utm_source": slugify(source),
        "utm_medium": slugify(medium),
        "utm_campaign": slugify(campaign_name),
        "utm_content": slugify(content),
    }
    return f"{base_url}?{urlencode(params)}"


def generate_utm_matrix(
    campaign_name: str,
    assets: list[dict],
) -> list[dict]:
    """Generate a complete UTM matrix for all planned assets.

    Each asset dict should have: asset_id, channel, medium, base_url (optional).
    Returns a list of dicts with asset_id and full utm_url.
    """
    matrix = []
    for asset in assets:
        url = generate_utm(
            campaign_name=campaign_name,
            source=asset["channel"],
            medium=asset["medium"],
            content=asset["asset_id"],
            base_url=asset.get("base_url", "https://www.arthur.ai"),
        )
        matrix.append({"asset_id": asset["asset_id"], "url": url})
    return matrix
