"""Agent 3c: Blog Writer — drafts the campaign anchor blog post."""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models import CampaignState
from utils.brand import load_voice_guide, load_blog_style_guide, load_link_targets


def _get_utm_link(state: CampaignState, asset_id: str) -> str:
    for link in state.channel_plan.utm_links:
        if link.asset_id == asset_id:
            return link.url
    return ""


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert B2B content writer specializing in long-form blog posts for enterprise technology companies.

## Brand Voice Guide
{voice_guide}

## Blog Style Guide
{blog_style_guide}

## Instructions
Write the complete campaign anchor blog post. Output in this exact format:

---
**Meta Title:** [50-60 characters, includes primary keyword]
**Meta Description:** [150-160 characters]

# [H1 — Blog Post Title]

[Full blog post in Markdown — 1,500-2,000 words]

[Include 3-5 internal links naturally woven into the content]

[End with a clear CTA section]
---

Rules:
- This is anchor content — emails and social posts drive traffic here.
- Lead with the reader's problem, not Arthur's product.
- Use concrete examples and scenarios relevant to the target industry.
- Primary keyword should appear in: meta title, H1, first 100 words, at least one H2, meta description.
- Internal links must use the exact URLs provided — never construct your own.
- Data and citations: reference specific regulatory frameworks accurately.
- Close with a CTA that feels like a natural next step, not a hard sell.
- Do NOT make specific compliance guarantees.
"""


def write_blog(state: CampaignState) -> dict:
    """Draft the anchor blog post."""
    voice_guide = load_voice_guide()
    blog_style_guide = load_blog_style_guide()
    link_targets = load_link_targets()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        voice_guide=voice_guide,
        blog_style_guide=blog_style_guide,
    )

    # Build internal links context
    links_section = "## Internal Links to Include\n"
    for name, url in link_targets.get("primary_links", {}).items():
        links_section += f"- {name}: {url}\n"

    # Blog UTM link
    blog_utm = ""
    for link in state.channel_plan.utm_links:
        if "blog" in link.asset_id:
            blog_utm = link.url
            break

    strategy_json = state.parsed_strategy.model_dump_json(indent=2)

    # Regen feedback
    feedback_section = ""
    if state.checkpoint_2_decision and state.checkpoint_2_decision.feedback:
        blog_feedback = [f for f in state.checkpoint_2_decision.feedback if f.asset_type == "blog"]
        if blog_feedback:
            feedback_section = "\n## Revision Feedback (incorporate this):\n"
            for fb in blog_feedback:
                feedback_section += f"- {fb.feedback}\n"

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192, temperature=0.7)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Write the anchor blog post for this campaign.\n\n"
                f"## Campaign Strategy\n{strategy_json}\n\n"
                f"{links_section}\n"
                f"## Blog UTM Link (for self-reference): {blog_utm}\n"
                f"{feedback_section}"
            )
        ),
    ]

    response = llm.invoke(messages)
    return {"blog_draft_raw": response.content}
