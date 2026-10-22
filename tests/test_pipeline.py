"""
Comprehensive test suite for AI Meeting Minutes Generator.
Tests data models, chunking engine, transcription, faithfulness evaluator,
and end-to-end pipeline integrity.
"""

import os
import sys
import json
import unittest

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import (
    Transcript,
    TranscriptSegment,
    ActionItem,
    MeetingMinutes,
    FaithfulnessVerificationResult
)
from chunking import needs_chunking, create_chunks, estimate_tokens
from evaluator import (
    clean_quote_text,
    lexical_quote_check,
    verify_action_item_heuristic,
    verify_meeting_minutes_faithfulness
)
from transcription import transcribe_audio


class TestDataModels(unittest.TestCase):
    """Test Pydantic schemas and serialization behavior."""

    def test_transcript_segment_formatting(self):
        seg = TranscriptSegment(start=72.0, end=95.5, text="We should deploy to production.", speaker="DevLead")
        self.assertEqual(seg.timestamp_str, "[01:12 - 01:35]")
        self.assertIn("DevLead: We should deploy to production.", seg.format_line())

    def test_action_item_defaults(self):
        item = ActionItem(
            description="Prepare benchmark report",
            evidence="Alex: We need a benchmark report."
        )
        self.assertEqual(item.owner, "Not specified")
        self.assertEqual(item.deadline, "Not specified")
        self.assertEqual(item.verification_status, "SUPPORTED")

    def test_meeting_minutes_serialization(self):
        minutes = MeetingMinutes(
            meeting_title="Architecture Review",
            meeting_summary="Discussed microservice migration.",
            key_decisions=["Migrate auth service to Go"],
            action_items=[
                ActionItem(
                    description="Write OpenAPI specs",
                    owner="Alice",
                    deadline="Friday",
                    evidence='[00:15 - 00:30] "Alice: I will write the OpenAPI specs by Friday."'
                )
            ],
            unresolved_questions=["Who manages deployment keys?"]
        )
        data = minutes.model_dump()
        self.assertEqual(data["meeting_title"], "Architecture Review")
        self.assertEqual(len(data["action_items"]), 1)
        self.assertEqual(data["action_items"][0]["owner"], "Alice")


class TestChunkingEngine(unittest.TestCase):
    """Test transcript chunking and overlap logic."""

    def setUp(self):
        # Create a synthetic transcript with 50 segments
        segments = []
        for i in range(50):
            start = i * 10.0
            end = start + 9.5
            text = f"This is segment number {i} discussing architectural component {i} in great detail."
            segments.append(TranscriptSegment(start=start, end=end, text=text, speaker=f"Speaker{i%3}"))
        
        full_text = " ".join(s.text for s in segments)
        self.transcript = Transcript(
            segments=segments,
            full_text=full_text,
            duration_seconds=500.0
        )

    def test_token_estimation(self):
        sample = "Hello world this is a test"
        tokens = estimate_tokens(sample)
        self.assertGreater(tokens, 0)

    def test_needs_chunking(self):
        short_transcript = Transcript(
            segments=[TranscriptSegment(start=0.0, end=5.0, text="Brief checkin.")],
            full_text="Brief checkin.",
            duration_seconds=5.0
        )
        self.assertFalse(needs_chunking(short_transcript, max_chars_threshold=1000))
        # With low threshold, self.transcript should require chunking
        self.assertTrue(needs_chunking(self.transcript, max_chars_threshold=500))

    def test_create_chunks_overlapping(self):
        chunks = create_chunks(self.transcript, max_chars_per_chunk=1000, overlap_chars=250)
        self.assertGreater(len(chunks), 1)
        
        # Verify chunks have chronological timestamps
        for i in range(len(chunks) - 1):
            self.assertLessEqual(chunks[i].start_time, chunks[i+1].start_time)
            # Verify overlap exists: next chunk start_time is before current chunk end_time
            self.assertLess(chunks[i+1].start_time, chunks[i].end_time)


class TestFaithfulnessEvaluator(unittest.TestCase):
    """Test deterministic and heuristic faithfulness verification."""

    def setUp(self):
        self.transcript_text = (
            "Hello team. Today we decided to migrate the primary database to PostgreSQL. "
            "David will complete the schema migration and deliver the scripts by next Wednesday. "
            "Sarah volunteered to update the API documentation. "
            "We have not decided who will manage the benchmark testing."
        )

    def test_lexical_quote_matching(self):
        quote = 'David: "David will complete the schema migration and deliver the scripts by next Wednesday."'
        found, ratio = lexical_quote_check(quote, self.transcript_text)
        self.assertTrue(found)
        self.assertGreaterEqual(ratio, 0.75)

    def test_supported_action_item(self):
        item = ActionItem(
            description="Complete schema migration and deliver scripts",
            owner="David",
            deadline="next Wednesday",
            evidence="David will complete the schema migration and deliver the scripts by next Wednesday."
        )
        res = verify_action_item_heuristic(item, 0, self.transcript_text)
        self.assertEqual(res.status, "SUPPORTED")
        self.assertTrue(res.quote_found_in_transcript)
        self.assertTrue(res.owner_faithfully_supported)
        self.assertTrue(res.deadline_faithfully_supported)

    def test_unsupported_action_item_hallucinated_quote(self):
        item = ActionItem(
            description="Build a mobile app in React Native",
            owner="Not specified",
            deadline="Not specified",
            evidence="We should definitely launch an iOS app next month."
        )
        res = verify_action_item_heuristic(item, 1, self.transcript_text)
        self.assertEqual(res.status, "UNSUPPORTED")
        self.assertFalse(res.quote_found_in_transcript)

    def test_partially_supported_action_item_invented_owner(self):
        # The task is mentioned, but owner was invented (Alex was not mentioned for API docs)
        item = ActionItem(
            description="Update the API documentation",
            owner="Michael",  # Fabricated owner
            deadline="Not specified",
            evidence="Sarah volunteered to update the API documentation."
        )
        res = verify_action_item_heuristic(item, 2, self.transcript_text)
        self.assertEqual(res.status, "PARTIALLY_SUPPORTED")
        self.assertFalse(res.owner_faithfully_supported)

    def test_verify_meeting_minutes_batch(self):
        transcript = Transcript(
            segments=[
                TranscriptSegment(start=0.0, end=10.0, text=self.transcript_text)
            ],
            full_text=self.transcript_text
        )
        minutes = MeetingMinutes(
            meeting_title="DB Migration",
            meeting_summary="Discussed DB migration",
            key_decisions=["Migrate to PostgreSQL"],
            action_items=[
                ActionItem(
                    description="Schema migration",
                    owner="David",
                    deadline="next Wednesday",
                    evidence="David will complete the schema migration and deliver the scripts by next Wednesday."
                ),
                ActionItem(
                    description="Redesign website logo",
                    owner="Unknown",
                    deadline="Tomorrow",
                    evidence="We must redesign the brand logo immediately."
                )
            ]
        )
        results = verify_meeting_minutes_faithfulness(minutes, transcript, client=None)
        self.assertEqual(len(results), 2)
        self.assertEqual(minutes.action_items[0].verification_status, "SUPPORTED")
        self.assertEqual(minutes.action_items[1].verification_status, "UNSUPPORTED")


class TestAudioTranscription(unittest.TestCase):
    """Test faster-whisper on actual audio file."""

    def test_sample_audio_transcription(self):
        audio_path = os.path.join(PROJECT_ROOT, "data", "sample_meeting.wav")
        if not os.path.exists(audio_path):
            self.skipTest("Sample audio file does not exist.")
        
        transcript = transcribe_audio(audio_path, model_size="base")
        self.assertIsInstance(transcript, Transcript)
        self.assertGreater(len(transcript.segments), 0)
        self.assertIn("PostgreSQL", transcript.full_text)
        self.assertIn("David", transcript.full_text)


if __name__ == "__main__":
    unittest.main()
