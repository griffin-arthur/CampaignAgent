"""Agent 1: Strategy Parser — extracts structured campaign data from a raw brief."""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models import CampaignState, ParsedStrategy

SYSTEM_PROMPT = """\
You are a marketing strategy parser. Your job is to read a raw campaign brief and extract \
structured data into a precise JSON schema.

You must output ONLY valid JSON matching this exact schema — no commentary, no markdown fences:

{
  "campaign_name": "<string>",
  "goal": "<string — the primary campaign objective>",
  "target_audience": "<string — comprehensive description of all target personas>",
  "key_messages": ["<string>", ...],
  "offer_or_cta": "<string — primary call to action>",
  "channels": ["<string>", ...],
  "timeline": {
    "launch_date": "<string>",
    "end_date": "<string>"
  },
  "tone": "<string — tone/voice description>",
  "constraints": ["<string>", ...]
}

Rules:
- Extract ALL key messages mentioned in the brief, including supporting messages.
- For channels, normalize to: "email", "linkedin_organic", "linkedin_ads", "blog". Add others only if explicitly mentioned.
- For constraints, include compliance guardrails, content restrictions, and any explicit "do not" instructions.
- For tone, synthesize the brief's tone guidance into a concise description.
- If a field is not mentioned in the brief, use a reasonable default or empty string/list.
- Be thorough — capture every detail that downstream agents will need to produce content.
"""


def parse_strategy(state: CampaignState) -> dict:
    """Parse raw brief into structured strategy. Returns state update dict."""
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=4096, temperature=0)

    human_content = f"Parse the following campaign brief:\n\n{state.raw_brief}"

    # If there's feedback from a previous checkpoint 1 review, include it
    feedback_text = getattr(state, "checkpoint_1_strategy_feedback", "")
    prev_strategy = getattr(state, "parsed_strategy", None)
    if feedback_text and prev_strategy:
        prev_json = prev_strategy.model_dump_json(indent=2) if hasattr(prev_strategy, "model_dump_json") else json.dumps(prev_strategy, indent=2)
        human_content += (
            f"\n\n---\nPREVIOUS PARSE (needs revision):\n{prev_json}"
            f"\n\nREVIEWER FEEDBACK — apply these changes:\n{feedback_text}"
        )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    content = response.content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()

    parsed = json.loads(content)
    strategy = ParsedStrategy(**parsed)

    return {"parsed_strategy": strategy, "current_step": "strategy_parsed"}
