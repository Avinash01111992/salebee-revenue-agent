# SaleBee Demo — Runnable LangGraph Workflow

A self-contained demo of the SaleBee revenue intelligence workflow using **mock data** (no audio required, no API keys for the LangGraph logic itself).

Paste a sales call transcript → get back structured deal intelligence JSON.

## Quick Start

```bash
cd demo
pip install -r requirements.txt
python run_demo.py
```

Or run with your own transcript:

```bash
python run_demo.py --transcript "We had a good call with the VP of Procurement. She mentioned budget is locked for Q4 but opens in January. A competitor is also in the conversation. She wants a technical demo before moving forward."
```

## What it does

1. Takes a transcript (text or mock)
2. Runs it through a LangGraph workflow with 3 nodes: `extract_insights` → `assess_risk` → `generate_output`
3. Returns structured JSON: deal stage, next actions, risk flags, sentiment

## Files

- `workflow.py` — LangGraph graph definition (the real architecture)
- `run_demo.py` — CLI runner
- `requirements.txt` — Dependencies
