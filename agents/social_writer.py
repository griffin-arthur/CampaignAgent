"""Agent 3b: Social Writer — drafts LinkedIn organic posts and sponsored ads."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models import CampaignState
from utils.brand import load_voice_guide, load_social_guidelines


def _get_utm_link(state: CampaignState, asset_id: str) -> str:
    for link in state.channel_plan.utm_links:
        if link.asset_id == asset_id:
            return link.url
    return ""


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert B2B social media copywriter specializing in LinkedIn content for enterprise tech companies.

## Brand Voice Guide
{voice_guide}

## Social Media Guidelines
{social_guidelines}

## Instructions
Write ALL LinkedIn content as specified in the campaign plan. Output in this exact format:

# ORGANIC POSTS

---
## Organic Post [N]: [Angle]
**Angle:** [stat-led / question-led / narrative-led / CTA-led]

[Full post text — ready to paste into LinkedIn]

**Link:** [UTM link, or "None — engagement-only post"]
---

# SPONSORED CONTENT ADS

---
## Ad Variant [Letter]
**Headline:** [max 70 characters]

**Body:**
[100-150 words]

**CTA Button:** [Book a Demo / Learn More]
**Destination URL:** [UTM link]
---

Rules:
- Use the provided UTM links — never construct your own.
- Organic posts: 500-1,200 characters. Hook in first 2-3 lines (before "See more").
- Line breaks between every 1-2 sentences for mobile readability.
- Max 5 hashtags per post.
- One engagement-only post (no link) — typically the question-led post.
- Ad headlines must be under 70 characters.
"""


def write_social(state: CampaignState) -> dict:
    """Draft LinkedIn organic posts and sponsored ads."""
    voice_guide = load_voice_guide()
    social_guidelines = load_social_guidelines()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        voice_guide=voice_guide,
        social_guidelines=social_guidelines,
    )

    # Build UTM context
    utm_section = "## Pre-generated UTM Links\n"
    for link in state.channel_plan.utm_links:
        if "linkedin" in link.asset_id or "organic" in link.asset_id or "ad" in link.asset_id:
            utm_section += f"- {link.asset_id}: {link.url}\n"

    # Also include content links that posts might reference
    for link in state.channel_plan.utm_links:
        if "blog" in link.asset_id:
            utm_section += f"- {link.asset_id}: {link.url}\n"

    strategy_json = state.parsed_strategy.model_dump_json(indent=2)

    # Check for regen feedback
    feedback_section = ""
    if state.checkpoint_2_decision and state.checkpoint_2_decision.feedback:
        social_feedback = [f for f in state.checkpoint_2_decision.feedback if f.asset_type == "social"]
        if social_feedback:
            feedback_section = "\n## Revision Feedback (incorporate this):\n"
            for fb in social_feedback:
                if fb.asset_index is not None:
                    feedback_section += f"- Post/Ad {fb.asset_index}: {fb.feedback}\n"
                else:
                    feedback_section += f"- All social content: {fb.feedback}\n"

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192, temperature=0.7)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Write all LinkedIn content for this campaign.\n\n"
                f"## Campaign Strategy\n{strategy_json}\n\n"
                f"{utm_section}"
                f"{feedback_section}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"social_draft_raw": response.content}
