"""
SaleBee Revenue Intelligence — LangGraph Workflow Demo

This is a simplified, runnable version of the production SaleBee workflow.
Uses mock LLM responses by default (set OPENAI_API_KEY env var for real GPT-4o).

Production version adds:
- Audio input (FFmpeg preprocessing + STT)
- Parallel transcription + notes processing nodes
- FastAPI REST endpoint
- Azure deployment
"""

from __future__ import annotations

import json
import os
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State — shared across all nodes in the graph
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # Input
    transcript: str

    # Populated by extract_insights node
    deal_stage: Optional[str]
    key_topics: Optional[list[str]]
    sentiment: Optional[str]

    # Populated by assess_risk node
    risk_flags: Optional[list[str]]
    next_actions: Optional[list[str]]

    # Final output
    crm_payload: Optional[dict]
    summary: Optional[str]


# ---------------------------------------------------------------------------
# LLM setup (falls back to mock if no API key)
# ---------------------------------------------------------------------------

def get_llm():
    """Return real LLM if API key is set, otherwise return mock."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return None


def llm_call(prompt: str, mock_response: dict) -> dict:
    """Call LLM or return mock response if no API key."""
    llm = get_llm()
    if llm is None:
        print("  [mock mode — set OPENAI_API_KEY for real GPT-4o]")
        return mock_response

    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        # Strip markdown code fences if present
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return mock_response


# ---------------------------------------------------------------------------
# Node 1: Extract Insights
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """
Analyze this sales call transcript and extract structured insights.

Transcript:
{transcript}

Respond with JSON only:
{{
  "deal_stage": "discovery|evaluation|proposal|negotiation|closed_won|closed_lost",
  "key_topics": ["topic1", "topic2"],
  "sentiment": "positive|neutral|cautiously_positive|negative"
}}
"""

EXTRACT_MOCK = {
    "deal_stage": "evaluation",
    "key_topics": ["budget timing", "competitor presence", "technical requirements"],
    "sentiment": "cautiously_positive"
}

def extract_insights_node(state: GraphState) -> GraphState:
    print("\n→ Node 1: Extracting insights from transcript...")
    result = llm_call(
        EXTRACT_PROMPT.format(transcript=state["transcript"]),
        EXTRACT_MOCK
    )
    return {
        **state,
        "deal_stage": result.get("deal_stage"),
        "key_topics": result.get("key_topics", []),
        "sentiment": result.get("sentiment"),
    }


# ---------------------------------------------------------------------------
# Node 2: Assess Risk + Next Actions
# ---------------------------------------------------------------------------

RISK_PROMPT = """
Based on this sales call transcript and initial analysis, identify risks and next actions.

Transcript: {transcript}
Deal stage: {deal_stage}
Sentiment: {sentiment}

Respond with JSON only:
{{
  "risk_flags": ["risk1", "risk2"],
  "next_actions": ["action1", "action2", "action3"]
}}
"""

RISK_MOCK = {
    "risk_flags": [
        "Budget not available until Q1 — deal may slip",
        "Competitor actively engaged — not sole vendor"
    ],
    "next_actions": [
        "Schedule technical demo before end of month",
        "Send ROI breakdown to address Q1 budget cycle",
        "Follow up in 2 weeks on competitor evaluation status"
    ]
}

def assess_risk_node(state: GraphState) -> GraphState:
    print("→ Node 2: Assessing risk and next actions...")
    result = llm_call(
        RISK_PROMPT.format(
            transcript=state["transcript"],
            deal_stage=state.get("deal_stage", "unknown"),
            sentiment=state.get("sentiment", "unknown")
        ),
        RISK_MOCK
    )
    return {
        **state,
        "risk_flags": result.get("risk_flags", []),
        "next_actions": result.get("next_actions", []),
    }


# ---------------------------------------------------------------------------
# Node 3: Generate CRM Output
# ---------------------------------------------------------------------------

def generate_output_node(state: GraphState) -> GraphState:
    print("→ Node 3: Generating CRM payload...")
    summary = (
        f"Call analysis: {state.get('deal_stage', 'unknown')} stage. "
        f"Sentiment: {state.get('sentiment', 'unknown')}. "
        f"Key topics: {', '.join(state.get('key_topics', []))}."
    )
    crm_payload = {
        "deal_stage": state.get("deal_stage"),
        "sentiment": state.get("sentiment"),
        "key_topics": state.get("key_topics", []),
        "risk_flags": state.get("risk_flags", []),
        "next_actions": state.get("next_actions", []),
        "call_summary": summary,
    }
    return {**state, "crm_payload": crm_payload, "summary": summary}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("extract_insights", extract_insights_node)
    workflow.add_node("assess_risk", assess_risk_node)
    workflow.add_node("generate_output", generate_output_node)

    workflow.set_entry_point("extract_insights")
    workflow.add_edge("extract_insights", "assess_risk")
    workflow.add_edge("assess_risk", "generate_output")
    workflow.add_edge("generate_output", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_transcript(transcript: str) -> dict:
    """
    Run the SaleBee workflow on a transcript string.
    Returns the final CRM payload dict.
    """
    graph = build_graph()
    final_state = graph.invoke({"transcript": transcript})
    return final_state["crm_payload"]
