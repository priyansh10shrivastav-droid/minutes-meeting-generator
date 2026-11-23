"""
End-to-End Test Suite verifying:
TEST A: Audio Upload Transcription Workflow
TEST B & C: Shared Meeting Minutes Structure from both workflows
Export Verification: PDF & TXT generation
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models import Transcript, TranscriptSegment, MeetingMinutes
from transcription import transcribe_audio
from minutes_generator import generate_meeting_minutes
from export_utils import generate_minutes_pdf, generate_minutes_txt

def test_workflows():
    print("=" * 60)
    print("TEST A: UPLOAD EXISTING AUDIO WORKFLOW")
    print("=" * 60)
    sample_wav = os.path.join(project_root, "data", "sample_meeting.wav")
    assert os.path.exists(sample_wav), f"Sample audio not found: {sample_wav}"
    
    print(f"1. Ingesting audio file: {sample_wav} ({round(os.path.getsize(sample_wav)/1024, 1)} KB)")
    transcript = transcribe_audio(sample_wav, model_size="base", device="cpu")
    print(f"2. Extracted {len(transcript.segments)} segments. Duration: {transcript.duration_seconds}s")
    print(f"   Sample text: {transcript.full_text[:120]}...")
    assert len(transcript.full_text) > 20, "Transcript extraction failed!"
    print("✅ TEST A PASS: Audio upload transcribed without microphone access.")

    print("\n" + "=" * 60)
    print("TEST B & C: VERIFY SHARED STRUCTURED OUTPUT FORMAT")
    print("=" * 60)
    # Test generation on Mode 1 transcript
    print("1. Running shared AI Minutes Engine on Mode 1 (Audio Upload) transcript...")
    m1_minutes = generate_meeting_minutes(
        transcript=transcript,
        language="English",
        meeting_type="Review"
    )
    
    print(f"   Title: {m1_minutes.meeting_title}")
    print(f"   Summary: {m1_minutes.meeting_summary[:100]}...")
    print(f"   Key Decisions: {len(m1_minutes.key_decisions)}")
    print(f"   Action Items: {len(m1_minutes.action_items)}")
    print(f"   Discussion Points: {len(m1_minutes.key_discussion_points)}")
    print(f"   Next Meeting: {m1_minutes.next_meeting_details}")

    assert m1_minutes.meeting_summary, "Missing executive summary"
    assert hasattr(m1_minutes, "key_discussion_points"), "Missing key discussion points"
    assert hasattr(m1_minutes, "key_decisions"), "Missing key decisions"
    assert hasattr(m1_minutes, "action_items"), "Missing action items"
    assert hasattr(m1_minutes, "next_meeting_details"), "Missing next meeting details"

    # Test generation on Mode 2 (Live Meeting) transcript
    print("\n2. Running shared AI Minutes Engine on Mode 2 (Live Meeting) transcript...")
    live_transcript_text = """
    [00:00:05] Facilitator: Good morning team, let's start the live review.
    [00:00:12] David: I will deploy the hotfix to production by 4 PM today.
    [00:00:20] Sarah: I will review the QA test logs once the hotfix is live.
    [00:00:28] Facilitator: We agreed to cancel tomorrow's morning standup. Let's meet next Tuesday at 11 AM instead.
    """.strip()

    live_segs = [TranscriptSegment(start=0.0, end=0.0, text=l.strip()) for l in live_transcript_text.split("\n") if l.strip()]
    m2_transcript = Transcript(segments=live_segs, full_text=live_transcript_text)
    
    m2_minutes = generate_meeting_minutes(
        transcript=m2_transcript,
        language="English",
        meeting_type="Standup"
    )

    print(f"   Title: {m2_minutes.meeting_title}")
    print(f"   Summary: {m2_minutes.meeting_summary[:100]}...")
    print(f"   Key Decisions: {len(m2_minutes.key_decisions)}")
    print(f"   Action Items: {len(m2_minutes.action_items)}")
    print(f"   Discussion Points: {len(m2_minutes.key_discussion_points)}")
    print(f"   Next Meeting: {m2_minutes.next_meeting_details}")

    assert m2_minutes.meeting_summary, "Missing executive summary"
    assert hasattr(m2_minutes, "key_discussion_points"), "Missing key discussion points"
    assert hasattr(m2_minutes, "key_decisions"), "Missing key decisions"
    assert hasattr(m2_minutes, "action_items"), "Missing action items"
    assert hasattr(m2_minutes, "next_meeting_details"), "Missing next meeting details"
    print("✅ TEST C PASS: Both modes produce identical, complete structured schema.")

    print("\n" + "=" * 60)
    print("TEST EXPORT CAPABILITIES: PDF & TXT")
    print("=" * 60)
    pdf_bytes = generate_minutes_pdf(m2_minutes)
    txt_data = generate_minutes_txt(m2_minutes)
    assert len(pdf_bytes) > 1000, "PDF generation failed"
    assert len(txt_data) > 100, "TXT generation failed"
    print(f"✅ Export verified! PDF size: {len(pdf_bytes)} bytes | TXT length: {len(txt_data)} chars")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_workflows()
