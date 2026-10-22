"""
Data models for AI Meeting Minutes Generator.
Defines Pydantic schemas for transcripts, action items, meeting minutes,
chunk analysis, and faithfulness verification.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


VerificationStatus = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED"]


class TranscriptSegment(BaseModel):
    """Represents a single timestamped segment in the transcript."""
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed spoken text")
    speaker: Optional[str] = Field(default=None, description="Speaker name or identifier if available")

    @property
    def timestamp_str(self) -> str:
        """Returns human-readable [MM:SS - MM:SS] timestamp."""
        start_min, start_sec = divmod(int(self.start), 60)
        end_min, end_sec = divmod(int(self.end), 60)
        return f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"

    def format_line(self) -> str:
        """Returns formatted transcript line."""
        prefix = f"{self.timestamp_str} "
        if self.speaker:
            prefix += f"{self.speaker}: "
        return f"{prefix}{self.text.strip()}"


class Transcript(BaseModel):
    """Complete meeting transcript containing segments and metadata."""
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Timestamped segments")
    full_text: str = Field(..., description="Consolidated plain text transcript")
    language: Optional[str] = Field(default="en", description="Detected language code")
    duration_seconds: Optional[float] = Field(default=0.0, description="Total duration of audio in seconds")

    @property
    def formatted_transcript(self) -> str:
        """Returns full transcript formatted with timestamps."""
        if not self.segments:
            return self.full_text
        return "\n".join(seg.format_line() for seg in self.segments)


class ActionItem(BaseModel):
    """Structured action item extracted from meeting transcript."""
    description: str = Field(
        ...,
        description="Specific actionable task agreed upon. Do NOT invent tasks not in transcript."
    )
    owner: str = Field(
        default="Not specified",
        description="Assigned owner. If not explicitly mentioned in transcript, MUST be 'Not specified'."
    )
    deadline: str = Field(
        default="Not specified",
        description="Explicit deadline/timeframe. If not explicitly mentioned in transcript, MUST be 'Not specified'."
    )
    evidence: str = Field(
        ...,
        description="Verbatim transcript quote and/or timestamp reference supporting this action item."
    )
    verification_status: VerificationStatus = Field(
        default="SUPPORTED",
        description="Status from faithfulness evaluation: SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED."
    )
    verification_explanation: Optional[str] = Field(
        default=None,
        description="Detailed explanation of faithfulness verification evaluation."
    )


class MeetingMinutes(BaseModel):
    """Final comprehensive structured meeting minutes."""
    meeting_title: str = Field(
        default="Meeting Minutes",
        description="Title or topic of the meeting derived strictly from the discussion."
    )
    meeting_date: Optional[str] = Field(
        default="Not specified",
        description="Meeting date if mentioned, otherwise 'Not specified'."
    )
    meeting_summary: str = Field(
        ...,
        description="Executive summary highlighting core discussion, context, and outcomes without hallucinating facts."
    )
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Explicit decisions agreed upon during the meeting. Mark 'None' or omit if no explicit decisions."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="List of action items with owners, deadlines, and verifiable transcript evidence."
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Questions, open debates, or blockers left unresolved at meeting conclusion."
    )


class IntermediateChunkAnalysis(BaseModel):
    """Partial analysis result from an individual transcript chunk during Map-Reduce."""
    chunk_index: int = Field(..., description="Index of this chunk")
    summary_points: List[str] = Field(
        default_factory=list,
        description="Key points and topics covered in this chunk."
    )
    decisions: List[str] = Field(
        default_factory=list,
        description="Candidate decisions mentioned in this chunk."
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Candidate action items with supporting quotes from this chunk."
    )
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Questions left unanswered or debated in this chunk."
    )


class FaithfulnessVerificationResult(BaseModel):
    """Diagnostic report for an individual action item verification."""
    action_item_index: int = Field(..., description="0-indexed position of action item")
    action_item_description: str = Field(..., description="Action item description")
    status: VerificationStatus = Field(..., description="SUPPORTED, PARTIALLY_SUPPORTED, or UNSUPPORTED")
    explanation: str = Field(..., description="Reasoning behind verification status")
    quote_found_in_transcript: bool = Field(..., description="True if evidence quote matches transcript text")
    owner_faithfully_supported: bool = Field(..., description="True if owner is explicitly supported or 'Not specified'")
    deadline_faithfully_supported: bool = Field(..., description="True if deadline is explicitly supported or 'Not specified'")
