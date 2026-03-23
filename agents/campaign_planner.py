"""Agent 2: Campaign Planner — produces channel plan, UTM matrix, and task breakdown."""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models import AssetSpec, ChannelPlan, CampaignState, Task, UTMLink
from utils.brand import load_utm_conventions, load_link_targets
from utils.utm import generate_utm

SYSTEM_PROMPT = """\
You are a B2B campaign planner specializing in multi-channel marketing campaigns. \
Given a parsed campaign strategy, produce a detailed channel plan.

Output ONLY valid JSON with this schema — no commentary, no markdown fences:

{
  "assets": [
    {
      "asset_type": "<e.g. nurture-email, organic-post, ad-variant, blog-post>",
      "channel": "<e.g. email, linkedin_organic, linkedin_ads, blog>",
      "description": "<brief description of this specific asset>",
      "count": <int — how many of this type to produce>
    }
  ],
  "tasks": [
    {
      "title": "<task title>",
      "description": "<what needs to be done>",
      "owner_suggestion": "<role, e.g. Content Marketing, Demand Gen, Marketing Ops>",
      "due_date": "<relative to launch, e.g. L-14 means 14 days before launch>",
      "dependencies": ["<task title that must complete first>"],
      "status": "draft"
    }
  ]
}

Rules:
- For each channel in the strategy, define specific assets with counts.
- Standard channel playbook:
  * Email: 3-part nurture sequence (problem awareness → urgency → solution + CTA)
  * LinkedIn organic: 4 posts (stat-led, question-led, narrative-led, CTA-led)
  * LinkedIn ads: 2 sponsored content variants for A/B testing
  * Blog: 1 anchor long-form post
- Tasks should cover: content creation, review/approval, design (if applicable), \
scheduling, UTM setup, list segmentation, campaign tracking setup.
- Task due dates are relative to launch date (L = launch day, L-7 = 7 days before, L+3 = 3 days after).
- Include post-launch tasks: performance monitoring, optimization, reporting.
- Be specific in descriptions — each task should be actionable.
"""


def plan_campaign(state: CampaignState) -> dict:
    """Generate channel plan with UTM matrix and task breakdown."""
    llm = ChatAnthropic(model="claude-opus-4-6-20250620", max_tokens=4096, temperature=0)

    strategy_json = state.parsed_strategy.model_dump_json(indent=2)

    human_content = f"Create a campaign plan for this strategy:\n\n{strategy_json}"

    # If there's feedback from a previous checkpoint 1 review, include it
    feedback_text = getattr(state, "checkpoint_1_plan_feedback", "")
    prev_plan = getattr(state, "channel_plan", None)
    if feedback_text and prev_plan:
        prev_json = prev_plan.model_dump_json(indent=2) if hasattr(prev_plan, "model_dump_json") else json.dumps(prev_plan, indent=2)
        human_content += (
            f"\n\n---\nPREVIOUS PLAN (needs revision):\n{prev_json}"
            f"\n\nREVIEWER FEEDBACK — apply these changes:\n{feedback_text}"
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()

    plan_data = json.loads(content)

    # Build typed asset specs
    assets = [AssetSpec(**a) for a in plan_data.get("assets", [])]
    tasks = [Task(**t) for t in plan_data.get("tasks", [])]

    # Generate deterministic UTM links for every asset
    link_targets = load_link_targets()
    utm_links: list[UTMLink] = []

    for asset in assets:
        for i in range(1, asset.count + 1):
            asset_id = f"{asset.asset_type}-{i}" if asset.count > 1 else asset.asset_type

            # Determine medium based on channel
            medium_map = {
                "email": "email",
                "linkedin_organic": "social",
                "linkedin_ads": "paid-social",
                "blog": "organic",
            }
            medium = medium_map.get(asset.channel, "other")

            # Determine base URL based on asset type
            if "blog" in asset.asset_type:
                base_url = link_targets.get("primary_links", {}).get("adg_product", "https://www.arthur.ai")
            elif "ad" in asset.asset_type:
                base_url = link_targets.get("primary_links", {}).get("book_demo", "https://www.arthur.ai/book-demo")
            else:
                base_url = link_targets.get("homepage", "https://www.arthur.ai")

            # Source is the platform name (without _organic / _ads suffix)
            source = asset.channel.split("_")[0]

            url = generate_utm(
                campaign_name=state.parsed_strategy.campaign_name,
                source=source,
                medium=medium,
                content=asset_id,
                base_url=base_url,
            )
            utm_links.append(UTMLink(asset_id=asset_id, url=url))

    channel_plan = ChannelPlan(assets=assets, utm_links=utm_links, tasks=tasks)

    return {"channel_plan": channel_plan, "current_step": "plan_ready"}
