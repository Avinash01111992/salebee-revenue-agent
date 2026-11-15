# SaleBee — Revenue Intelligence Agent

> LangGraph-powered AI workflow that turns raw sales call recordings into structured revenue intelligence: deal stage, next actions, risk flags, and CRM-ready notes.

**Built and owned:** LangGraph workflow design, FastAPI backend, FFmpeg audio pipeline, STT integration, prompt engineering for structured extraction, API design.

---

## The Problem

Sales managers spend 20–30 minutes per call reviewing recordings to update CRM and prep coaching. Reps forget to log. Deal visibility is always stale.

**Goal:** Upload an audio file. Get back structured JSON with everything the CRM needs — in under 2 minutes.

---

## How It Works

```
Sales Call Recording (MP3 / WAV / M4A)
            │
            ▼
    ┌───────────────┐
    │  FFmpeg        │  Normalize audio: mono, 16kHz, noise floor
    │  Preprocessing │  Removes codec variation across recording sources
    └──────┬────────┘
           │
           ▼
    ┌───────────────────────────────────────┐
    │         LangGraph Workflow             │
    │                                        │
    │   START                                │
    │     │                                  │
    │     ├──────────────┬──────────────┐   │
    │     ▼              ▼              │   │
    │  [transcribe]  [process_notes]   │   │  ← Parallel execution
    │     │              │              │   │
    │     └──────────────┘              │   │
    │             │                     │   │
    │             ▼                     │   │
    │      [extract_insights]           │   │
    │             │                     │   │
    │             ▼                     │   │
    │      [generate_output]            │   │
    │             │                     │   │
    │            END                    │   │
    └───────────────────────────────────┘
            │
            ▼
    Structured JSON response
```

### Why parallel nodes?

`transcribe` (audio → text via STT) and `process_notes` (parse any pre-call notes the rep uploaded) are independent. Running them in parallel cuts latency by ~40% on a typical 30-minute call.

---

## LangGraph Implementation

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class GraphState(TypedDict):
    # Inputs
    audio_path: str
    pre_call_notes: Optional[str]
    
    # Populated by parallel nodes
    transcript: Optional[str]
    parsed_notes: Optional[dict]
    
    # Populated by extract_insights
    deal_stage: Optional[str]
    next_actions: Optional[list[str]]
    risk_flags: Optional[list[str]]
    sentiment: Optional[str]
    
    # Final output
    crm_payload: Optional[dict]
    summary: Optional[str]


def transcribe_node(state: GraphState) -> GraphState:
    """STT: audio file → clean transcript."""
    transcript = stt_client.transcribe(
        audio_path=state["audio_path"],
        language="en",
        diarize=True  # speaker separation
    )
    return {**state, "transcript": transcript}


def process_notes_node(state: GraphState) -> GraphState:
    """Parse pre-call notes if provided; return empty dict if not."""
    if not state.get("pre_call_notes"):
        return {**state, "parsed_notes": {}}
    parsed = parse_notes(state["pre_call_notes"])
    return {**state, "parsed_notes": parsed}


def extract_insights_node(state: GraphState) -> GraphState:
    """LLM extraction: transcript + notes → structured revenue intelligence."""
    response = llm.invoke(EXTRACTION_PROMPT.format(
        transcript=state["transcript"],
        pre_call_context=state.get("parsed_notes", {})
    ))
    insights = parse_structured_output(response)
    return {**state, **insights}


def generate_output_node(state: GraphState) -> GraphState:
    """Assemble CRM payload and human-readable summary."""
    crm_payload = {
        "deal_stage": state["deal_stage"],
        "next_actions": state["next_actions"],
        "risk_flags": state["risk_flags"],
        "call_summary": state["summary"],
        "sentiment": state["sentiment"],
    }
    return {**state, "crm_payload": crm_payload}


# Build the graph
workflow = StateGraph(GraphState)

workflow.add_node("transcribe", transcribe_node)
workflow.add_node("process_notes", process_notes_node)
workflow.add_node("extract_insights", extract_insights_node)
workflow.add_node("generate_output", generate_output_node)

# Parallel fan-out from START
workflow.set_entry_point("transcribe")
workflow.add_edge("transcribe", "extract_insights")
workflow.add_edge("process_notes", "extract_insights")

# Sequential after merge
workflow.add_edge("extract_insights", "generate_output")
workflow.add_edge("generate_output", END)

app = workflow.compile()
```

---

## API

### `POST /analyze`

Accepts multipart form data:

```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio=@sales_call.mp3" \
  -F "notes=Customer mentioned budget approval in Q1. Key contact: Sarah (VP Procurement)."
```

**Response:**

```json
{
  "deal_stage": "evaluation",
  "next_actions": [
    "Send proposal by Friday",
    "Schedule technical demo with Sarah's team",
    "Follow up on Q1 budget timeline"
  ],
  "risk_flags": [
    "Competing vendor mentioned (not named)",
    "Decision delayed to Q1 — budget cycle dependency"
  ],
  "sentiment": "cautiously_positive",
  "summary": "30-min discovery call with VP Procurement. Customer confirmed pain point around manual reporting. Budget exists but approval required in Q1. Competitor is in conversation. Next step is technical demo.",
  "processing_time_seconds": 47.3
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Workflow orchestration | LangGraph |
| Backend | FastAPI (Python 3.11) |
| Audio preprocessing | FFmpeg |
| Speech-to-text | Azure AI Speech / Whisper |
| LLM | GPT-4o |
| Hosting | Azure Container Apps |

---

## Repository Structure

```
salebee-revenue-agent/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── workflow/
│   │   ├── graph.py             # LangGraph definition
│   │   ├── nodes/
│   │   │   ├── transcribe.py
│   │   │   ├── process_notes.py
│   │   │   ├── extract_insights.py
│   │   │   └── generate_output.py
│   │   └── state.py             # GraphState TypedDict
│   ├── audio/
│   │   └── preprocessor.py      # FFmpeg normalization
│   └── prompts/
│       └── extraction.py        # Structured extraction prompt
├── tests/
│   ├── test_workflow.py
│   └── fixtures/                # Sample audio + expected outputs
├── Dockerfile
└── requirements.txt
```

---

## Contact

**Avinash Sadolalu Chandaraju**  
avinashavinu@gmail.com  
[LinkedIn](https://www.linkedin.com/in/avinash-s-c-b07575276/) | [GitHub](https://github.com/Avinash01111992)
