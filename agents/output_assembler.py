"""Agent 4: Output Assembler — packages all approved assets into deliverables."""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

from models import CampaignState


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def assemble_output(state: CampaignState) -> dict:
    """Package all approved assets into deliverable files."""
    campaign_name = state.parsed_strategy.campaign_name or "campaign"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{campaign_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Combined campaign document (Markdown)
    combined_doc = _build_combined_doc(state)
    (run_dir / "campaign_assets.md").write_text(combined_doc)

    # 2. UTM spreadsheet (CSV)
    _write_utm_csv(state, run_dir / "utm_matrix.csv")

    # 3. Campaign task list (JSON)
    tasks_data = [t.model_dump() for t in state.channel_plan.tasks]
    (run_dir / "campaign_tasks.json").write_text(json.dumps(tasks_data, indent=2))

    # 4. Individual channel files
    channels_dir = run_dir / "channels"
    channels_dir.mkdir(exist_ok=True)

    if state.draft_assets.emails:
        email_content = _format_emails(state)
        (channels_dir / "email_sequence.md").write_text(email_content)

    if state.draft_assets.social_posts:
        social_content = _format_social(state)
        (channels_dir / "linkedin_content.md").write_text(social_content)

    if state.draft_assets.blog.body:
        blog_content = _format_blog(state)
        (channels_dir / "blog_post.md").write_text(blog_content)

    # 5. Campaign summary (brief metadata)
    summary = {
        "campaign_name": state.parsed_strategy.campaign_name,
        "goal": state.parsed_strategy.goal,
        "channels": state.parsed_strategy.channels,
        "timeline": state.parsed_strategy.timeline.model_dump(),
        "total_assets": len(state.channel_plan.assets),
        "total_utm_links": len(state.channel_plan.utm_links),
        "total_tasks": len(state.channel_plan.tasks),
        "generated_at": timestamp,
        "output_directory": str(run_dir),
    }
    (run_dir / "campaign_summary.json").write_text(json.dumps(summary, indent=2))

    return {"final_output_path": str(run_dir), "current_step": "complete"}


def _build_combined_doc(state: CampaignState) -> str:
    """Build a single combined Markdown document with all assets."""
    sections = []
    sections.append(f"# Campaign: {state.parsed_strategy.campaign_name}")
    sections.append(f"\n**Goal:** {state.parsed_strategy.goal}")
    sections.append(f"**Audience:** {state.parsed_strategy.target_audience}")
    sections.append(f"**Timeline:** {state.parsed_strategy.timeline.launch_date} → {state.parsed_strategy.timeline.end_date}")
    sections.append(f"**CTA:** {state.parsed_strategy.offer_or_cta}")

    sections.append("\n---\n")

    # Emails
    if state.draft_assets.emails:
        sections.append("## Email Nurture Sequence\n")
        sections.append(_format_emails(state))

    # Social
    if state.draft_assets.social_posts:
        sections.append("\n---\n\n## LinkedIn Content\n")
        sections.append(_format_social(state))

    # Blog
    if state.draft_assets.blog.body:
        sections.append("\n---\n\n## Blog Post\n")
        sections.append(_format_blog(state))

    # UTM Reference
    sections.append("\n---\n\n## UTM Link Reference\n")
    for link in state.channel_plan.utm_links:
        sections.append(f"- **{link.asset_id}**: `{link.url}`")

    return "\n".join(sections)


def _format_emails(state: CampaignState) -> str:
    lines = []
    for email in state.draft_assets.emails:
        lines.append(f"### Email {email.position}")
        if email.subject_lines:
            for i, subj in enumerate(email.subject_lines):
                lines.append(f"**Subject Line {'AB'[i] if i < 2 else i+1}:** {subj}")
        lines.append(f"**Preview Text:** {email.preview_text}")
        lines.append(f"\n{email.body}")
        lines.append(f"\n**CTA:** {email.cta} → {email.utm_link}")
        lines.append("\n---\n")
    return "\n".join(lines)


def _format_social(state: CampaignState) -> str:
    lines = []
    organic = [p for p in state.draft_assets.social_posts if p.post_type == "organic"]
    sponsored = [p for p in state.draft_assets.social_posts if p.post_type == "sponsored"]

    if organic:
        lines.append("### Organic Posts\n")
        for i, post in enumerate(organic, 1):
            lines.append(f"#### Post {i}: {post.angle}")
            lines.append(post.content)
            if post.link:
                lines.append(f"\n**Link:** {post.link}")
            lines.append("\n---\n")

    if sponsored:
        lines.append("### Sponsored Content Ads\n")
        for i, ad in enumerate(sponsored, 1):
            lines.append(f"#### Ad Variant {chr(64+i)}")
            lines.append(ad.content)
            if ad.link:
                lines.append(f"\n**Destination:** {ad.link}")
            lines.append("\n---\n")

    return "\n".join(lines)


def _format_blog(state: CampaignState) -> str:
    blog = state.draft_assets.blog
    lines = []
    if blog.meta_title:
        lines.append(f"**Meta Title:** {blog.meta_title}")
    if blog.meta_description:
        lines.append(f"**Meta Description:** {blog.meta_description}")
    lines.append(f"\n{blog.body}")
    return "\n".join(lines)


def _write_utm_csv(state: CampaignState, path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "full_url"])
        for link in state.channel_plan.utm_links:
            writer.writerow([link.asset_id, link.url])
