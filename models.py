"""Pydantic models for the campaign orchestrator state and data structures."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core data models (output of Agent 1 / input to everything else)
# ---------------------------------------------------------------------------

class Timeline(BaseModel):
    launch_date: str = ""
    end_date: str = ""


class ParsedStrategy(BaseModel):
    campaign_name: str = ""
    goal: str = ""
    target_audience: str = ""
    key_messages: list[str] = Field(default_factory=list)
    offer_or_cta: str = ""
    channels: list[str] = Field(default_factory=list)
    timeline: Timeline = Field(default_factory=Timeline)
    tone: str = ""
    constraints: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Campaign plan models (output of Agent 2)
# ---------------------------------------------------------------------------

class AssetSpec(BaseModel):
    asset_type: str = ""        # e.g. "nurture-email", "organic-post", "ad-variant", "blog-post"
    channel: str = ""           # e.g. "email", "linkedin", "blog"
    description: str = ""       # brief description of the asset
    count: int = 1              # how many of this type


class UTMLink(BaseModel):
    asset_id: str = ""          # e.g. "nurture-email-1"
    url: str = ""               # full URL with UTM params


class Task(BaseModel):
    title: str = ""
    description: str = ""
    owner_suggestion: str = ""
    due_date: str = ""          # relative to launch, e.g. "L-14" (14 days before launch)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "draft"


class ChannelPlan(BaseModel):
    assets: list[AssetSpec] = Field(default_factory=list)
    utm_links: list[UTMLink] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Writer outputs
# ---------------------------------------------------------------------------

class EmailDraft(BaseModel):
    position: int = 1               # 1, 2, 3 in sequence
    subject_lines: list[str] = Field(default_factory=list)  # A/B variants
    preview_text: str = ""
    body: str = ""
    cta: str = ""
    utm_link: str = ""


class SocialPost(BaseModel):
    post_type: str = ""             # "organic" or "sponsored"
    angle: str = ""                 # e.g. "stat-led", "regulatory-hook"
    content: str = ""
    link: str = ""                  # with UTM, empty if engagement-only


class BlogDraft(BaseModel):
    meta_title: str = ""
    meta_description: str = ""
    body: str = ""                  # full markdown


class DraftAssets(BaseModel):
    emails: list[EmailDraft] = Field(default_factory=list)
    social_posts: list[SocialPost] = Field(default_factory=list)
    blog: BlogDraft = Field(default_factory=BlogDraft)


# ---------------------------------------------------------------------------
# Feedback models for human-in-the-loop
# ---------------------------------------------------------------------------

class AssetFeedback(BaseModel):
    asset_type: str = ""            # "email", "social", "blog"
    asset_index: int | None = None  # which specific asset (None = all)
    feedback: str = ""              # freeform feedback for regen
    approved: bool = False


class CheckpointDecision(BaseModel):
    approved: bool = False
    edits: dict = Field(default_factory=dict)       # field-level overrides
    feedback: list[AssetFeedback] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level graph state (TypedDict for LangGraph compatibility)
# ---------------------------------------------------------------------------

from typing import TypedDict


class CampaignState(TypedDict, total=False):
    """Full state object that flows through the LangGraph.

    Using TypedDict so LangGraph can merge partial updates from parallel nodes.
    Each writer writes to its own key to avoid overwrite conflicts during fan-out.
    """

    # Input
    raw_brief: str

    # Agent 1 output
    parsed_strategy: ParsedStrategy

    # Agent 2 output
    channel_plan: ChannelPlan

    # Checkpoint 1
    checkpoint_1_decision: CheckpointDecision
    checkpoint_1_strategy_feedback: str
    checkpoint_1_plan_feedback: str

    # Agent 3 outputs — separate keys so parallel writers don't clobber each other
    email_drafts: list[EmailDraft]
    social_drafts: list[SocialPost]
    blog_draft: BlogDraft

    # Combined (assembled after writers finish)
    draft_assets: DraftAssets

    # Checkpoint 2
    checkpoint_2_decision: CheckpointDecision

    # Agent 4 output
    final_output_path: str

    # Control flow
    current_step: str
    error: str
