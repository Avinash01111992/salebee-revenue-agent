"""
SaleBee Demo Runner

Usage:
    python run_demo.py
    python run_demo.py --transcript "Your transcript here"
"""

import argparse
import json
from workflow import analyze_transcript

MOCK_TRANSCRIPT = """
We had a 45-minute call with Sarah, VP of Procurement at Meridian Industries.
She's evaluating vendors for their field reporting automation project.

She liked our demo but mentioned that budget approval for new tools requires
sign-off from the CFO, which won't happen until January at the earliest.
They're also looking at two other vendors — she didn't name them but said
one has a cheaper price point.

Her main concern is integration with their existing ERP system. She wants to
see a technical demo with their IT team before moving forward.

Action agreed: we send a technical proposal by Friday, and schedule a demo
with her IT lead next week.
"""

def main():
    parser = argparse.ArgumentParser(description="SaleBee revenue intelligence demo")
    parser.add_argument(
        "--transcript",
        type=str,
        default=None,
        help="Sales call transcript to analyze (uses mock transcript if not provided)"
    )
    args = parser.parse_args()

    transcript = args.transcript or MOCK_TRANSCRIPT

    print("=" * 60)
    print("SaleBee Revenue Intelligence — LangGraph Demo")
    print("=" * 60)
    print(f"\nTranscript ({len(transcript)} chars):")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    print("\nRunning workflow...")

    result = analyze_transcript(transcript)

    print("\n" + "=" * 60)
    print("CRM PAYLOAD OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("\nDone.")

if __name__ == "__main__":
    main()
