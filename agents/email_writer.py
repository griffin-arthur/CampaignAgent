"""Agent 3a: Email Writer — drafts the full email nurture sequence."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models import CampaignState
from utils.brand import load_voice_guide, load_email_templates


def _get_utm_link(state: CampaignState, asset_id: str) -> str:
    """Look up pre-generated UTM link by asset ID."""
    for link in state.channel_plan.utm_links:
        if link.asset_id == asset_id:
            return link.url
    return ""


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert B2B email copywriter. You write email sequences for enterprise marketing campaigns.

## Brand Voice Guide
{voice_guide}

## Email Templates & Guidelines
{email_templates}

## Instructions
Write the COMPLETE email nurture sequence as specified. For each email, output in this exact format:

---
## Email [N] of [Total]: [Angle]

**Subject Line A:** [first variant]
**Subject Line B:** [second variant for A/B testing]

**Preview Text:** [40-90 characters]

**Body:**
[Full email body copy — 150-250 words, short paragraphs]

**CTA:** [CTA text] → [UTM link]
---

Rules:
- Each email must reference its position in the sequence. Email 2 should acknowledge that the reader \
received Email 1 (subtly, not explicitly "as we mentioned in our last email").
- Use the provided UTM links for all CTAs — never construct your own links.
- Follow the nurture arc: problem awareness → urgency → solution + CTA.
- Match the tone and constraints from the campaign strategy exactly.
- Subject lines should be 6-10 words, no ALL CAPS, no exclamation marks.
"""


def write_emails(state: CampaignState) -> dict:
    """Draft the email nurture sequence."""
    voice_guide = load_voice_guide()
    email_templates = load_email_templates()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        voice_guide=voice_guide,
        email_templates=email_templates,
    )

    # Build context with UTM links for each email
    email_assets = [a for a in state.channel_plan.assets if "email" in a.asset_type]
    total_emails = sum(a.count for a in email_assets)
    if total_emails == 0:
        total_emails = 3  # default

    utm_links_section = "## Pre-generated UTM Links\n"
    for i in range(1, total_emails + 1):
        link = _get_utm_link(state, f"nurture-email-{i}")
        utm_links_section += f"- Email {i}: {link}\n"

    # Also provide any content links from the strategy
    for link in state.channel_plan.utm_links:
        if "blog" in link.asset_id or "shadow" in link.asset_id:
            utm_links_section += f"- {link.asset_id}: {link.url}\n"

    strategy_json = state.parsed_strategy.model_dump_json(indent=2)

    # Check for regeneration feedback
    feedback_section = ""
    if state.checkpoint_2_decision and state.checkpoint_2_decision.feedback:
        email_feedback = [f for f in state.checkpoint_2_decision.feedback if f.asset_type == "email"]
        if email_feedback:
            feedback_section = "\n## Revision Feedback (incorporate this):\n"
            for fb in email_feedback:
                if fb.asset_index is not None:
                    feedback_section += f"- Email {fb.asset_index}: {fb.feedback}\n"
                else:
                    feedback_section += f"- All emails: {fb.feedback}\n"

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192, temperature=0.7)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Write a {total_emails}-email nurture sequence for this campaign.\n\n"
                f"## Campaign Strategy\n{strategy_json}\n\n"
                f"{utm_links_section}"
                f"{feedback_section}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"email_draft_raw": response.content}
