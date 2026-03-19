"""Streamlit UI for the Campaign Orchestrator Agent."""

import json
import uuid

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from models import (
    CampaignState,
    CheckpointDecision,
    AssetFeedback,
)
from langgraph.types import Command
from graph import create_app

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Campaign Orchestrator",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "app" not in st.session_state:
    st.session_state.app = create_app()
if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input → running_parse → checkpoint_1 → running_writers → checkpoint_2 → complete
if "state_snapshot" not in st.session_state:
    st.session_state.state_snapshot = None
if "error" not in st.session_state:
    st.session_state.error = None


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _to_dict(obj):
    """Normalize a value to a plain dict/list — handles Pydantic models, dicts, and primitives."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Campaign Orchestrator Agent")
st.caption("Paste a campaign brief → get channel-ready draft assets, UTMs, and a task plan.")

# ---------------------------------------------------------------------------
# Stage: INPUT
# ---------------------------------------------------------------------------
if st.session_state.stage == "input":
    st.header("1. Campaign Brief")

    brief = st.text_area(
        "Paste your campaign brief below",
        height=400,
        placeholder="Paste your full campaign strategy brief here...",
    )

    # Or upload a file
    uploaded = st.file_uploader("Or upload a brief (.md, .txt)", type=["md", "txt"])
    if uploaded:
        brief = uploaded.read().decode("utf-8")

    if st.button("🚀 Launch Campaign Agent", type="primary", disabled=not brief):
        st.session_state.stage = "running_parse"
        st.session_state.brief = brief
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: RUNNING PARSE + PLAN (Agents 1 & 2)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "running_parse":
    st.header("Parsing brief & building campaign plan...")

    with st.spinner("Agent 1: Parsing strategy... then Agent 2: Building campaign plan..."):
        try:
            app = st.session_state.app
            config = get_config()

            # Run the graph — it will interrupt at checkpoint_1
            result = app.invoke(
                {"raw_brief": st.session_state.brief},
                config=config,
            )

            # Get the current state from the checkpoint
            snapshot = app.get_state(config)
            st.session_state.state_snapshot = snapshot
            st.session_state.stage = "checkpoint_1"
            st.rerun()

        except Exception as e:
            st.error(f"Error during parsing/planning: {e}")
            st.session_state.error = str(e)
            st.session_state.stage = "input"

# ---------------------------------------------------------------------------
# Stage: CHECKPOINT 1 — Review strategy + plan
# ---------------------------------------------------------------------------
elif st.session_state.stage == "checkpoint_1":
    st.header("Checkpoint 1: Review Strategy & Plan")
    st.info("Review the parsed strategy and campaign plan below. Edit any fields, then approve to begin content generation.")

    snapshot = st.session_state.state_snapshot
    if snapshot is None:
        st.error("No state snapshot found. Please restart.")
        st.session_state.stage = "input"
        st.stop()

    # Extract state values from snapshot, normalizing to dicts
    state_values = snapshot.values if hasattr(snapshot, 'values') else snapshot
    _raw_strategy = state_values.get("parsed_strategy", {})
    _raw_plan = state_values.get("channel_plan", {})
    strategy = _raw_strategy.model_dump() if hasattr(_raw_strategy, "model_dump") else (_raw_strategy if isinstance(_raw_strategy, dict) else {})
    plan = _raw_plan.model_dump() if hasattr(_raw_plan, "model_dump") else (_raw_plan if isinstance(_raw_plan, dict) else {})

    # ---- Strategy display ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Parsed Strategy")
        st.markdown(f"**Campaign:** {strategy.get('campaign_name', 'N/A')}")
        st.markdown(f"**Goal:** {strategy.get('goal', 'N/A')}")
        audience = strategy.get('target_audience', 'N/A') or 'N/A'
        st.markdown(f"**Audience:** {audience[:200]}...")
        st.markdown(f"**CTA:** {strategy.get('offer_or_cta', 'N/A')}")

        st.markdown("**Key Messages:**")
        for msg in strategy.get("key_messages", []):
            st.markdown(f"- {msg[:150]}...")

        st.markdown(f"**Channels:** {', '.join(strategy.get('channels', []))}")

        # --- Timeline: clear date display ---
        timeline = strategy.get("timeline", {})
        launch = timeline.get("launch_date", "TBD") or "TBD"
        end = timeline.get("end_date", "TBD") or "TBD"
        st.markdown("**Campaign Timeline:**")
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.metric(label="Launch Date", value=launch)
        with tcol2:
            st.metric(label="End Date", value=end)

        st.markdown(f"**Tone:** {strategy.get('tone', 'N/A')}")

        if strategy.get("constraints"):
            st.markdown("**Constraints:**")
            for c in strategy["constraints"]:
                st.markdown(f"- {c}")

    with col2:
        st.subheader("Channel Plan")

        # Assets table
        assets = plan.get("assets", [])
        if assets:
            st.markdown("**Planned Assets:**")
            for asset in assets:
                a = _to_dict(asset)
                st.markdown(f"- **{a.get('asset_type', '')}** ({a.get('channel', '')}) × {a.get('count', 1)} — {a.get('description', '')}")

        # UTM Matrix
        utm_links = plan.get("utm_links", [])
        if utm_links:
            with st.expander(f"UTM Matrix ({len(utm_links)} links)", expanded=False):
                for link in utm_links:
                    l = _to_dict(link)
                    st.code(f"{l.get('asset_id', '')}: {l.get('url', '')}", language=None)

        # --- Tasks: clear timeline with date column ---
        tasks = plan.get("tasks", [])
        if tasks:
            st.markdown("**Task Timeline:**")
            # Map relative codes to human-readable labels
            _phase_labels = {
                "L-": "Before Launch",
                "L":  "Launch Day",
                "L+": "Post-Launch",
            }
            def _due_label(due: str) -> str:
                """Turn 'L-14' into '14 days before launch', etc."""
                due = due.strip()
                if not due:
                    return "TBD"
                if due.startswith("L-"):
                    days = due[2:]
                    return f"{days} days before launch"
                if due.startswith("L+"):
                    days = due[2:]
                    return f"{days} days after launch"
                if due == "L" or due == "L+0" or due == "L-0":
                    return "Launch day"
                return due  # already a real date or unknown format

            # Group tasks by phase
            pre_launch = [t for t in tasks if _to_dict(t).get("due_date", "").startswith("L-")]
            launch_day = [t for t in tasks if _to_dict(t).get("due_date", "").strip() in ("L", "L+0", "L-0")]
            post_launch = [t for t in tasks if _to_dict(t).get("due_date", "").startswith("L+") and _to_dict(t).get("due_date", "").strip() not in ("L+0",)]
            other = [t for t in tasks if t not in pre_launch and t not in launch_day and t not in post_launch]

            for phase_name, phase_tasks in [
                ("Pre-Launch", pre_launch),
                ("Launch Day", launch_day),
                ("Post-Launch", post_launch),
                ("Other", other),
            ]:
                if not phase_tasks:
                    continue
                with st.expander(f"{phase_name} ({len(phase_tasks)} tasks)", expanded=(phase_name == "Pre-Launch")):
                    for task in phase_tasks:
                        t = _to_dict(task)
                        due = _due_label(t.get("due_date", ""))
                        deps = t.get("dependencies", [])
                        dep_str = f"  \n  Depends on: {', '.join(deps)}" if deps else ""
                        st.markdown(
                            f"**{t.get('title', '')}** · 📅 {due} · 👤 {t.get('owner_suggestion', 'Unassigned')}  \n"
                            f"  {t.get('description', '')}{dep_str}"
                        )

    st.divider()

    # ---- Feedback section ----
    st.subheader("Feedback")
    st.caption("Leave blank to approve as-is. Provide feedback to regenerate the strategy and/or plan.")

    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        strategy_feedback = st.text_area(
            "Strategy feedback (parsed brief)",
            placeholder="e.g. 'Add a fourth persona: Head of Data Science' or 'Goal should emphasize pipeline, not just demos'",
            key="cp1_strategy_fb",
            height=120,
        )
    with fb_col2:
        plan_feedback = st.text_area(
            "Campaign plan feedback",
            placeholder="e.g. 'Add a webinar channel' or 'Move the blog post earlier in the timeline' or 'Add a 4th email to the nurture'",
            key="cp1_plan_fb",
            height=120,
        )

    st.divider()

    # ---- Action buttons ----
    col_approve, col_regen, col_restart = st.columns(3)

    with col_approve:
        if st.button("✅ Approve & Generate Content", type="primary"):
            st.session_state.stage = "running_writers"
            st.rerun()

    with col_regen:
        if st.button("🔄 Regenerate with Feedback"):
            has_fb = bool(strategy_feedback.strip() or plan_feedback.strip())
            if not has_fb:
                st.warning("Add feedback to at least one section before regenerating.")
            else:
                st.session_state.stage = "running_regen_plan"
                st.session_state.cp1_strategy_fb = strategy_feedback.strip()
                st.session_state.cp1_plan_fb = plan_feedback.strip()
                st.rerun()

    with col_restart:
        if st.button("🗑️ Start Over"):
            st.session_state.stage = "input"
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.app = create_app()
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: RUNNING REGEN PLAN (re-run strategy parser / campaign planner with feedback)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "running_regen_plan":
    st.header("Regenerating plan with your feedback...")
    with st.spinner("Re-running agents with feedback..."):
        try:
            app = st.session_state.app
            config = get_config()

            strategy_fb = st.session_state.get("cp1_strategy_fb", "")
            plan_fb = st.session_state.get("cp1_plan_fb", "")

            decision = CheckpointDecision(
                approved=False,
                edits={
                    "strategy_feedback": strategy_fb,
                    "plan_feedback": plan_fb,
                },
            )
            app.invoke(Command(resume=decision.model_dump()), config=config)

            snapshot = app.get_state(config)
            st.session_state.state_snapshot = snapshot
            st.session_state.stage = "checkpoint_1"
            st.rerun()
        except Exception as e:
            st.error(f"Error during regeneration: {e}")
            import traceback
            st.code(traceback.format_exc())
            if st.button("Back to Checkpoint 1"):
                st.session_state.stage = "checkpoint_1"
                st.rerun()

# ---------------------------------------------------------------------------
# Stage: RUNNING WRITERS (Agents 3a/3b/3c)
# ---------------------------------------------------------------------------
elif st.session_state.stage == "running_writers":
    st.header("Generating Content...")
    with st.spinner("Writing emails, LinkedIn posts, and blog content in parallel..."):
        try:
            app = st.session_state.app
            config = get_config()

            decision = CheckpointDecision(approved=True)
            app.invoke(Command(resume=decision.model_dump()), config=config)

            snapshot = app.get_state(config)
            st.session_state.state_snapshot = snapshot
            st.session_state.stage = "checkpoint_2"
            st.rerun()
        except Exception as e:
            st.error(f"Error during content generation: {e}")
            import traceback
            st.code(traceback.format_exc())
            if st.button("Back to Checkpoint 1"):
                st.session_state.stage = "checkpoint_1"
                st.rerun()

# ---------------------------------------------------------------------------
# Stage: CHECKPOINT 2 — Review draft assets
# ---------------------------------------------------------------------------
elif st.session_state.stage == "checkpoint_2":
    st.header("Checkpoint 2: Review Draft Assets")
    st.info("Review all generated content below. Approve as-is, or provide feedback for regeneration.")

    snapshot = st.session_state.state_snapshot
    if snapshot is None:
        st.error("No state snapshot found.")
        st.session_state.stage = "input"
        st.stop()

    state_values = snapshot.values if hasattr(snapshot, 'values') else snapshot
    draft_assets = _to_dict(state_values.get("draft_assets", {}))

    # --- Emails ---
    st.subheader("📧 Email Nurture Sequence")
    emails = draft_assets.get("emails", [])
    email_feedback_inputs = {}

    if emails:
        for i, raw_email in enumerate(emails):
            email = _to_dict(raw_email) if not isinstance(raw_email, dict) else raw_email
            with st.expander(f"Email {email.get('position', i+1)}", expanded=(i == 0)):
                subjects = email.get("subject_lines", [])
                if subjects:
                    for j, subj in enumerate(subjects):
                        st.markdown(f"**Subject Line {'AB'[j] if j < 2 else j+1}:** {subj}")
                st.markdown(f"**Preview:** {email.get('preview_text', '')}")
                st.markdown("---")
                st.markdown(email.get("body", ""))
                st.markdown(f"**CTA:** {email.get('cta', '')} → `{email.get('utm_link', '')}`")

                email_feedback_inputs[i] = st.text_input(
                    f"Feedback for Email {email.get('position', i+1)} (leave blank to approve)",
                    key=f"email_fb_{i}",
                )
    else:
        st.warning("No email drafts generated.")

    # --- Social ---
    st.subheader("💼 LinkedIn Content")
    social_posts = draft_assets.get("social_posts", [])
    social_feedback_inputs = {}

    if social_posts:
        social_posts_d = [_to_dict(p) if not isinstance(p, dict) else p for p in social_posts]
        organic = [p for p in social_posts_d if p.get("post_type") == "organic"]
        sponsored = [p for p in social_posts_d if p.get("post_type") == "sponsored"]

        if organic:
            st.markdown("**Organic Posts:**")
            for i, post in enumerate(organic):
                with st.expander(f"Organic Post {i+1}: {post.get('angle', '')}", expanded=False):
                    st.markdown(post.get("content", ""))
                    if post.get("link"):
                        st.markdown(f"**Link:** `{post['link']}`")
                    social_feedback_inputs[f"organic_{i}"] = st.text_input(
                        f"Feedback for Organic Post {i+1}",
                        key=f"social_organic_fb_{i}",
                    )

        if sponsored:
            st.markdown("**Sponsored Ads:**")
            for i, ad in enumerate(sponsored):
                with st.expander(f"Ad Variant {chr(65+i)}: {ad.get('angle', '')}", expanded=False):
                    st.markdown(ad.get("content", ""))
                    if ad.get("link"):
                        st.markdown(f"**Destination:** `{ad['link']}`")
                    social_feedback_inputs[f"ad_{i}"] = st.text_input(
                        f"Feedback for Ad Variant {chr(65+i)}",
                        key=f"social_ad_fb_{i}",
                    )
    else:
        st.warning("No social drafts generated.")

    # --- Blog ---
    st.subheader("📝 Blog Post")
    blog = _to_dict(draft_assets.get("blog", {}))
    blog_feedback_input = ""

    if blog and blog.get("body"):
        st.markdown(f"**Meta Title:** {blog.get('meta_title', '')}")
        st.markdown(f"**Meta Description:** {blog.get('meta_description', '')}")
        with st.expander("Full Blog Post", expanded=False):
            st.markdown(blog.get("body", ""))
        blog_feedback_input = st.text_input(
            "Feedback for Blog Post (leave blank to approve)",
            key="blog_fb",
        )
    else:
        st.warning("No blog draft generated.")

    st.divider()

    # --- Approval Controls ---
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("✅ Approve All & Assemble", type="primary"):
            app = st.session_state.app
            config = get_config()

            decision = CheckpointDecision(approved=True)
            app.invoke(Command(resume=decision.model_dump()), config=config)

            snapshot = app.get_state(config)
            st.session_state.state_snapshot = snapshot
            st.session_state.stage = "complete"
            st.rerun()

    with col_b:
        if st.button("🔄 Regenerate with Feedback"):
            # Collect all feedback
            feedback_items = []

            for i, fb_text in email_feedback_inputs.items():
                if fb_text.strip():
                    feedback_items.append(AssetFeedback(
                        asset_type="email",
                        asset_index=i + 1,
                        feedback=fb_text.strip(),
                        approved=False,
                    ))

            for key, fb_text in social_feedback_inputs.items():
                if fb_text.strip():
                    feedback_items.append(AssetFeedback(
                        asset_type="social",
                        feedback=fb_text.strip(),
                        approved=False,
                    ))

            if blog_feedback_input.strip():
                feedback_items.append(AssetFeedback(
                    asset_type="blog",
                    feedback=blog_feedback_input.strip(),
                    approved=False,
                ))

            if not feedback_items:
                st.warning("No feedback provided — add feedback to at least one asset before regenerating.")
            else:
                app = st.session_state.app
                config = get_config()

                decision = CheckpointDecision(approved=False, feedback=feedback_items)
                app.invoke(Command(resume=decision.model_dump()), config=config)

                snapshot = app.get_state(config)
                st.session_state.state_snapshot = snapshot
                st.session_state.stage = "checkpoint_2"
                st.rerun()

    with col_c:
        if st.button("🗑️ Start Over"):
            st.session_state.stage = "input"
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.app = create_app()
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: COMPLETE
# ---------------------------------------------------------------------------
elif st.session_state.stage == "complete":
    st.header("🎉 Campaign Package Complete!")

    snapshot = st.session_state.state_snapshot
    state_values = snapshot.values if hasattr(snapshot, 'values') else snapshot

    output_path = state_values.get("final_output_path", "")

    if output_path:
        st.success(f"All deliverables saved to: `{output_path}`")

        st.markdown("### Deliverables")
        st.markdown(f"""
- **`campaign_assets.md`** — Combined document with all channel copy
- **`utm_matrix.csv`** — Every link, every UTM parameter
- **`campaign_tasks.json`** — Structured task list for PM tool import
- **`campaign_summary.json`** — Campaign metadata
- **`channels/`** — Individual channel files (email, LinkedIn, blog)
        """)

        # Show campaign summary
        import os
        summary_path = os.path.join(output_path, "campaign_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            st.json(summary)
    else:
        st.warning("Output path not found in state. Check the output/ directory.")

    if st.button("🆕 Start New Campaign"):
        st.session_state.stage = "input"
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.app = create_app()
        st.rerun()
