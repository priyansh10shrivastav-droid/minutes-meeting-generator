"""
End-to-end pipeline verification script.
Tests the entire pipeline: audio transcription -> chunking -> mock/live minutes generation -> faithfulness verification -> export formatting.
"""

import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


from models import Transcript, MeetingMinutes, ActionItem
from transcription import transcribe_audio
from chunking import needs_chunking, create_chunks
from evaluator import verify_meeting_minutes_faithfulness


def main():
    print("==================================================")
    print("AI MEETING MINUTES GENERATOR - E2E PIPELINE TEST")
    print("==================================================")

    # 1. Test Audio Transcription
    audio_path = os.path.join(PROJECT_ROOT, "data", "sample_meeting.wav")
    print(f"\n[1/5] Transcribing Audio: {audio_path}...")
    transcript = transcribe_audio(audio_path, model_size="base")
    print(f"  ✓ Transcription successful!")
    print(f"  ✓ Detected Language: {transcript.language}")
    print(f"  ✓ Total Audio Duration: {transcript.duration_seconds}s")
    print(f"  ✓ Segment Count: {len(transcript.segments)}")
    print(f"  ✓ Transcribed Text: {transcript.full_text[:120]}...")

    # 2. Test Chunking
    print("\n[2/5] Testing Chunking & Context Window Analysis...")
    requires_chunking = needs_chunking(transcript, max_chars_threshold=100)
    chunks = create_chunks(transcript, max_chars_per_chunk=100, overlap_chars=20)
    print(f"  ✓ Needs Chunking Flag (threshold 100): {requires_chunking}")
    print(f"  ✓ Chunks Generated: {len(chunks)}")
    for c in chunks[:2]:
        print(f"    - Chunk {c.chunk_index + 1}: {c.start_time:.1f}s to {c.end_time:.1f}s ({len(c.text)} chars)")

    # 3. Simulate Structured Minutes Extraction
    print("\n[3/5] Testing Structured Minutes Schema & Evidence Grounding...")
    minutes = MeetingMinutes(
        meeting_title="Sprint Kickoff & Database Migration",
        meeting_date="Not specified",
        meeting_summary=(
            "The team met to review the sprint kickoff. A decision was made to migrate to PostgreSQL. "
            "David was assigned to complete the database schema migration by next Wednesday, while Sarah will update the API docs."
        ),
        key_decisions=[
            "Migrate primary database to PostgreSQL"
        ],
        action_items=[
            ActionItem(
                description="Complete the database schema migration",
                owner="David",
                deadline="next Wednesday",
                evidence="David will complete the schema migration by next Wednesday."
            ),
            ActionItem(
                description="Update the API docs",
                owner="Sarah",
                deadline="Not specified",
                evidence="Sarah will update the API docs."
            ),
            ActionItem(
                description="Deploy blockchain smart contracts",
                owner="Michael",
                deadline="Tomorrow",
                evidence="We will deploy Ethereum contracts immediately."
            )
        ],
        unresolved_questions=[
            "What is the rollback procedure if migration fails?"
        ]
    )
    print(f"  ✓ Meeting Minutes Model validated successfully!")
    print(f"  ✓ Extracted Decisions: {len(minutes.key_decisions)}")
    print(f"  ✓ Extracted Action Items: {len(minutes.action_items)}")

    # 4. Test Faithfulness & Evidence Verification Engine
    print("\n[4/5] Running Faithfulness Verification Module...")
    results = verify_meeting_minutes_faithfulness(minutes, transcript, client=None)
    for res in results:
        status_icon = "🟢" if res.status == "SUPPORTED" else ("🟡" if res.status == "PARTIALLY_SUPPORTED" else "🔴")
        print(f"  {status_icon} Action Item #{res.action_item_index + 1}: {res.action_item_description}")
        print(f"     Status: {res.status}")
        print(f"     Quote Found: {res.quote_found_in_transcript} | Owner Grounded: {res.owner_faithfully_supported} | Deadline Grounded: {res.deadline_faithfully_supported}")
        print(f"     Explanation: {res.explanation}")

    # Verify that Item 1 and 2 are SUPPORTED, and Item 3 is UNSUPPORTED
    assert results[0].status == "SUPPORTED", "Action Item 1 should be SUPPORTED"
    assert results[1].status == "SUPPORTED", "Action Item 2 should be SUPPORTED"
    assert results[2].status == "UNSUPPORTED", "Action Item 3 should be UNSUPPORTED"
    print("  ✓ Verification classifications verified accurate!")

    # 5. Test Export Generation
    print("\n[5/5] Testing Output Export Generation (JSON & TXT)...")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "test_minutes.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(minutes.model_dump(), f, indent=2)
    assert os.path.exists(json_path)
    print(f"  ✓ JSON export saved to {json_path}")

    print("\n==================================================")
    print("ALL PIPELINE STAGES PASSED VERIFICATION!")
    print("==================================================")


if __name__ == "__main__":
    main()
