"""LangGraph orchestration — wires all agents into a stateful workflow with human checkpoints."""

import json
import operator
from typing import Annotated, Any

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from models import (
    CampaignState,
    ParsedStrategy,
    ChannelPlan,
    CheckpointDecision,
    DraftAssets,
    EmailDraft,
    SocialPost,
    BlogDraft,
    AssetSpec,
    UTMLink,
    Task,
)
from agents.strategy_parser import parse_strategy
from agents.campaign_planner import plan_campaign
from agents.email_writer import write_emails
from agents.social_writer import write_social
from agents.blog_writer import write_blog
from agents.output_assembler import assemble_output


# ---------------------------------------------------------------------------
# Node functions (thin wrappers that operate on CampaignState)
# ---------------------------------------------------------------------------

def _to_pydantic_state(state: dict):
    """Convert dict state to an object with attribute access for agent functions."""
    from types import SimpleNamespace

    def _convert(key, val):
        if key == "parsed_strategy" and isinstance(val, dict):
            return ParsedStrategy(**val)
        if key == "channel_plan" and isinstance(val, dict):
            cp = val.copy()
            cp["assets"] = [AssetSpec(**a) if isinstance(a, dict) else a for a in cp.get("assets", [])]
            cp["utm_links"] = [UTMLink(**u) if isinstance(u, dict) else u for u in cp.get("utm_links", [])]
            cp["tasks"] = [Task(**t) if isinstance(t, dict) else t for t in cp.get("tasks", [])]
            return ChannelPlan(**cp)
        if key == "draft_assets" and isinstance(val, dict):
            da = val.copy()
            da["emails"] = [EmailDraft(**e) if isinstance(e, dict) else e for e in da.get("emails", [])]
            da["social_posts"] = [SocialPost(**s) if isinstance(s, dict) else s for s in da.get("social_posts", [])]
            da["blog"] = BlogDraft(**da["blog"]) if isinstance(da.get("blog"), dict) else da.get("blog", BlogDraft())
            return DraftAssets(**da)
        if key in ("checkpoint_1_decision", "checkpoint_2_decision") and isinstance(val, dict):
            return CheckpointDecision(**val)
        if key == "blog_draft" and isinstance(val, dict):
            return BlogDraft(**val)
        if key == "email_drafts" and isinstance(val, list):
            return [EmailDraft(**e) if isinstance(e, dict) else e for e in val]
        if key == "social_drafts" and isinstance(val, list):
            return [SocialPost(**s) if isinstance(s, dict) else s for s in val]
        return val

    # Build namespace with defaults + converted values
    defaults = {
        "raw_brief": "",
        "parsed_strategy": ParsedStrategy(),
        "channel_plan": ChannelPlan(),
        "checkpoint_1_decision": None,
        "checkpoint_1_strategy_feedback": "",
        "checkpoint_1_plan_feedback": "",
        "email_drafts": [],
        "social_drafts": [],
        "blog_draft": BlogDraft(),
        "draft_assets": DraftAssets(),
        "checkpoint_2_decision": None,
        "final_output_path": "",
        "current_step": "start",
        "error": "",
    }
    for key, val in state.items():
        defaults[key] = _convert(key, val)

    return SimpleNamespace(**defaults)


def node_parse_strategy(state: CampaignState) -> dict:
    """Agent 1: Parse the raw brief into structured strategy."""
    pstate = _to_pydantic_state(state)
    return parse_strategy(pstate)


def node_plan_campaign(state: CampaignState) -> dict:
    """Agent 2: Generate channel plan, UTMs, task breakdown."""
    pstate = _to_pydantic_state(state)
    return plan_campaign(pstate)


def node_checkpoint_1(state: CampaignState) -> dict:
    """Human checkpoint 1: Review strategy + plan before content generation."""
    strategy = state.get("parsed_strategy", {})
    plan = state.get("channel_plan", {})

    # Serialize for display
    strat_dump = strategy.model_dump() if hasattr(strategy, "model_dump") else strategy
    plan_dump = plan.model_dump() if hasattr(plan, "model_dump") else plan

    decision = interrupt({
        "checkpoint": "checkpoint_1",
        "message": "Review the parsed strategy, channel plan, UTM matrix, and task timeline.",
        "parsed_strategy": strat_dump,
        "channel_plan": plan_dump,
    })

    if isinstance(decision, dict):
        checkpoint_decision = CheckpointDecision(**decision)
    else:
        checkpoint_decision = decision

    # Extract per-section feedback from the edits dict (set by UI)
    edits = checkpoint_decision.edits or {}
    result = {
        "checkpoint_1_decision": checkpoint_decision,
        "current_step": "checkpoint_1_approved" if checkpoint_decision.approved else "checkpoint_1_feedback",
        "checkpoint_1_strategy_feedback": edits.get("strategy_feedback", ""),
        "checkpoint_1_plan_feedback": edits.get("plan_feedback", ""),
    }
    return result


def node_write_emails(state: CampaignState) -> dict:
    """Agent 3a: Draft email sequence."""
    pstate = _to_pydantic_state(state)
    result = write_emails(pstate)
    raw = result.get("email_draft_raw", "")
    emails = _parse_email_drafts(raw)
    return {"email_drafts": emails}


def node_write_social(state: CampaignState) -> dict:
    """Agent 3b: Draft LinkedIn content."""
    pstate = _to_pydantic_state(state)
    result = write_social(pstate)
    raw = result.get("social_draft_raw", "")
    posts = _parse_social_drafts(raw)
    return {"social_drafts": posts}


def node_write_blog(state: CampaignState) -> dict:
    """Agent 3c: Draft blog post."""
    pstate = _to_pydantic_state(state)
    result = write_blog(pstate)
    raw = result.get("blog_draft_raw", "")
    blog = _parse_blog_draft(raw)
    return {"blog_draft": blog}


def node_merge_drafts(state: CampaignState) -> dict:
    """Merge parallel writer outputs into a single DraftAssets object."""
    emails = state.get("email_drafts", [])
    social = state.get("social_drafts", [])
    blog = state.get("blog_draft", BlogDraft())

    # Convert dicts back to Pydantic if needed
    if emails and isinstance(emails[0], dict):
        emails = [EmailDraft(**e) for e in emails]
    if social and isinstance(social[0], dict):
        social = [SocialPost(**s) for s in social]
    if isinstance(blog, dict):
        blog = BlogDraft(**blog)

    return {
        "draft_assets": DraftAssets(emails=emails, social_posts=social, blog=blog),
        "current_step": "drafts_ready",
    }


def node_checkpoint_2(state: CampaignState) -> dict:
    """Human checkpoint 2: Review all draft assets."""
    assets = state.get("draft_assets", {})
    assets_dump = assets.model_dump() if hasattr(assets, "model_dump") else assets

    decision = interrupt({
        "checkpoint": "checkpoint_2",
        "message": "Review all draft assets. Approve, edit, or request regeneration with feedback.",
        "draft_assets": assets_dump,
    })

    if isinstance(decision, dict):
        checkpoint_decision = CheckpointDecision(**decision)
    else:
        checkpoint_decision = decision

    return {"checkpoint_2_decision": checkpoint_decision, "current_step": "checkpoint_2_approved"}


def node_assemble_output(state: CampaignState) -> dict:
    """Agent 4: Package everything into deliverables."""
    pstate = _to_pydantic_state(state)
    return assemble_output(pstate)




# ---------------------------------------------------------------------------
# Parsing helpers (raw LLM markdown → structured models)
# ---------------------------------------------------------------------------

def _parse_email_drafts(raw: str) -> list[EmailDraft]:
    """Best-effort parse of email writer output into EmailDraft objects."""
    emails = []
    sections = raw.split("## Email ")
    for section in sections[1:]:  # skip preamble
        email = EmailDraft()
        lines = section.strip().split("\n")

        # Extract position from first line
        first_line = lines[0] if lines else ""
        for ch in first_line:
            if ch.isdigit():
                email.position = int(ch)
                break

        # Parse fields
        body_lines = []
        in_body = False
        for line in lines[1:]:
            stripped = line.strip()
            if stripped.startswith("**Subject Line A:**"):
                email.subject_lines.append(stripped.replace("**Subject Line A:**", "").strip())
                in_body = False
            elif stripped.startswith("**Subject Line B:**"):
                email.subject_lines.append(stripped.replace("**Subject Line B:**", "").strip())
                in_body = False
            elif stripped.startswith("**Preview Text:**"):
                email.preview_text = stripped.replace("**Preview Text:**", "").strip()
                in_body = False
            elif stripped.startswith("**Body:**"):
                in_body = True
            elif stripped.startswith("**CTA:**"):
                in_body = False
                cta_parts = stripped.replace("**CTA:**", "").strip()
                if "→" in cta_parts:
                    parts = cta_parts.split("→", 1)
                    email.cta = parts[0].strip()
                    email.utm_link = parts[1].strip()
                else:
                    email.cta = cta_parts
            elif in_body:
                body_lines.append(line)

        email.body = "\n".join(body_lines).strip()
        if email.body or email.subject_lines:
            emails.append(email)

    # Fallback: if parsing failed, create a single email with the raw content
    if not emails:
        emails.append(EmailDraft(position=1, body=raw))

    return emails


def _parse_social_drafts(raw: str) -> list[SocialPost]:
    """Best-effort parse of social writer output into SocialPost objects."""
    posts = []

    # Split organic posts
    if "# ORGANIC POSTS" in raw or "## Organic Post" in raw:
        organic_sections = raw.split("## Organic Post ")
        for section in organic_sections[1:]:
            post = SocialPost(post_type="organic")
            lines = section.strip().split("\n")

            # Extract angle
            for line in lines:
                if line.strip().startswith("**Angle:**"):
                    post.angle = line.replace("**Angle:**", "").strip().strip("*")
                    break

            # Extract link
            for line in lines:
                if line.strip().startswith("**Link:**"):
                    link_text = line.replace("**Link:**", "").strip()
                    if "none" not in link_text.lower() and "engagement" not in link_text.lower():
                        post.link = link_text
                    break

            # Content is everything between angle and link/end
            content_lines = []
            capture = False
            for line in lines[1:]:
                stripped = line.strip()
                if stripped.startswith("**Angle:**"):
                    capture = True
                    continue
                if stripped.startswith("**Link:**") or stripped.startswith("---"):
                    break
                if capture:
                    content_lines.append(line)

            post.content = "\n".join(content_lines).strip()
            if post.content:
                posts.append(post)

    # Split sponsored ads
    if "# SPONSORED" in raw or "## Ad Variant" in raw:
        ad_sections = raw.split("## Ad Variant ")
        for section in ad_sections[1:]:
            post = SocialPost(post_type="sponsored")
            lines = section.strip().split("\n")

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("**Headline:**"):
                    post.angle = stripped.replace("**Headline:**", "").strip()
                elif stripped.startswith("**Destination URL:**") or stripped.startswith("**Destination:**"):
                    post.link = stripped.split(":", 1)[1].strip() if ":" in stripped else ""

            content_lines = []
            capture = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("**Body:**"):
                    capture = True
                    continue
                if capture and (stripped.startswith("**CTA") or stripped.startswith("**Destination")):
                    break
                if capture:
                    content_lines.append(line)

            post.content = "\n".join(content_lines).strip()
            if post.content:
                posts.append(post)

    # Fallback
    if not posts:
        posts.append(SocialPost(post_type="organic", content=raw))

    return posts


def _parse_blog_draft(raw: str) -> BlogDraft:
    """Best-effort parse of blog writer output into BlogDraft."""
    blog = BlogDraft()

    lines = raw.split("\n")
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**Meta Title:**"):
            blog.meta_title = stripped.replace("**Meta Title:**", "").strip()
        elif stripped.startswith("**Meta Description:**"):
            blog.meta_description = stripped.replace("**Meta Description:**", "").strip()
        else:
            body_lines.append(line)

    blog.body = "\n".join(body_lines).strip()

    if not blog.body:
        blog.body = raw

    return blog


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _fan_out_to_writers(state: CampaignState) -> list[str]:
    """After checkpoint 1 approval, fan out to all three writer nodes.
    If rejected with strategy feedback → re-parse then re-plan.
    If rejected with only plan feedback → re-plan only.
    """
    decision = state.get("checkpoint_1_decision")
    if isinstance(decision, dict):
        decision = CheckpointDecision(**decision)
    if decision and decision.approved:
        return ["write_emails", "write_social", "write_blog"]

    # Check which kind of feedback was given
    strategy_fb = state.get("checkpoint_1_strategy_feedback", "")
    if strategy_fb:
        return ["parse_strategy"]  # will cascade: parse → plan → checkpoint_1 again
    return ["plan_campaign"]


def _fan_out_after_checkpoint_2(state: CampaignState) -> list[str]:
    """After checkpoint 2, either assemble or re-run all writers."""
    decision = state.get("checkpoint_2_decision")
    if isinstance(decision, dict):
        decision = CheckpointDecision(**decision)
    if not decision or decision.approved:
        return ["assemble_output"]

    feedback = decision.feedback
    if not feedback or all(f.approved for f in feedback):
        return ["assemble_output"]

    return ["write_emails", "write_social", "write_blog"]


def build_graph() -> StateGraph:
    """Build the campaign orchestrator LangGraph."""

    graph = StateGraph(CampaignState)

    # Add nodes
    graph.add_node("parse_strategy", node_parse_strategy)
    graph.add_node("plan_campaign", node_plan_campaign)
    graph.add_node("checkpoint_1", node_checkpoint_1)
    graph.add_node("write_emails", node_write_emails)
    graph.add_node("write_social", node_write_social)
    graph.add_node("write_blog", node_write_blog)
    graph.add_node("merge_drafts", node_merge_drafts)
    graph.add_node("checkpoint_2", node_checkpoint_2)
    graph.add_node("assemble_output", node_assemble_output)

    # Linear flow: parse → plan → checkpoint 1
    graph.set_entry_point("parse_strategy")
    graph.add_edge("parse_strategy", "plan_campaign")
    graph.add_edge("plan_campaign", "checkpoint_1")

    # After checkpoint 1: fan out to all writers (parallel) or loop back to planning/parsing
    graph.add_conditional_edges(
        "checkpoint_1",
        _fan_out_to_writers,
        ["write_emails", "write_social", "write_blog", "plan_campaign", "parse_strategy"],
    )

    # All writers converge into merge_drafts, then checkpoint 2
    graph.add_edge("write_emails", "merge_drafts")
    graph.add_edge("write_social", "merge_drafts")
    graph.add_edge("write_blog", "merge_drafts")
    graph.add_edge("merge_drafts", "checkpoint_2")

    # After checkpoint 2: assemble output or regen all writers
    graph.add_conditional_edges(
        "checkpoint_2",
        _fan_out_after_checkpoint_2,
        ["assemble_output", "write_emails", "write_social", "write_blog"],
    )

    graph.add_edge("assemble_output", END)

    return graph


def create_app(checkpointer=None):
    """Create the compiled LangGraph app with checkpointing.

    Uses PostgresSaver if DATABASE_URL is set, otherwise falls back to MemorySaver.
    """
    if checkpointer is None:
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            import psycopg

            conn = psycopg.connect(db_url)
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
        else:
            checkpointer = MemorySaver()

    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
