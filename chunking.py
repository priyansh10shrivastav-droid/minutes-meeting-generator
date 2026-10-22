"""
Chunking module for long meeting transcripts.
Implements overlapping window chunking preserving timestamped segment boundaries
for Map-Reduce processing when transcripts exceed context thresholds.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from models import Transcript, TranscriptSegment


class TranscriptChunk(BaseModel):
    """Represents a chunk of a meeting transcript with metadata and overlapping context."""
    chunk_index: int = Field(..., description="0-indexed sequence number of chunk")
    start_time: float = Field(default=0.0, description="Start timestamp in seconds")
    end_time: float = Field(default=0.0, description="End timestamp in seconds")
    text: str = Field(..., description="Formatted transcript text in this chunk")
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Segments in this chunk")
    estimated_tokens: int = Field(default=0, description="Rough estimate of token count")

    @property
    def timestamp_header(self) -> str:
        """Formatted header showing time range of chunk."""
        start_min, start_sec = divmod(int(self.start_time), 60)
        end_min, end_sec = divmod(int(self.end_time), 60)
        return f"[Chunk {self.chunk_index + 1} | Time: {start_min:02d}:{start_sec:02d} to {end_min:02d}:{end_sec:02d}]"


def estimate_tokens(text: str) -> int:
    """Rough heuristic token estimator (~4 chars per token)."""
    return max(1, len(text) // 4)


def needs_chunking(transcript: Transcript, max_chars_threshold: int = 12000) -> bool:
    """
    Determines if transcript exceeds single-pass threshold.
    Default 12,000 characters is ~3,000 tokens, leaving ample room
    for prompts and structured output parsing in standard LLM context.
    """
    total_length = len(transcript.formatted_transcript)
    return total_length > max_chars_threshold


def create_chunks(
    transcript: Transcript,
    max_chars_per_chunk: int = 10000,
    overlap_chars: int = 1500
) -> List[TranscriptChunk]:
    """
    Splits transcript into overlapping chunks while preserving segment boundaries.
    
    Args:
        transcript: The Transcript model with segments.
        max_chars_per_chunk: Target max characters per chunk (~2500 tokens).
        overlap_chars: Character overlap between consecutive chunks to avoid split contexts.
        
    Returns:
        List of TranscriptChunk objects ready for Map-Reduce processing.
    """
    # If transcript has no discrete segments, fallback to text-based chunking
    if not transcript.segments:
        text = transcript.full_text
        if len(text) <= max_chars_per_chunk:
            return [
                TranscriptChunk(
                    chunk_index=0,
                    start_time=0.0,
                    end_time=transcript.duration_seconds or 0.0,
                    text=text,
                    segments=[],
                    estimated_tokens=estimate_tokens(text)
                )
            ]
        
        chunks: List[TranscriptChunk] = []
        step = max(1, max_chars_per_chunk - overlap_chars)
        idx = 0
        for i in range(0, len(text), step):
            chunk_slice = text[i : i + max_chars_per_chunk]
            chunks.append(
                TranscriptChunk(
                    chunk_index=idx,
                    start_time=0.0,
                    end_time=transcript.duration_seconds or 0.0,
                    text=chunk_slice,
                    segments=[],
                    estimated_tokens=estimate_tokens(chunk_slice)
                )
            )
            idx += 1
            if i + max_chars_per_chunk >= len(text):
                break
        return chunks

    # Segment-aware chunking preserving timestamps
    chunks = []
    current_segments: List[TranscriptSegment] = []
    current_char_len = 0
    chunk_index = 0
    
    i = 0
    num_segments = len(transcript.segments)
    
    while i < num_segments:
        seg = transcript.segments[i]
        line = seg.format_line() + "\n"
        seg_len = len(line)
        
        current_segments.append(seg)
        current_char_len += seg_len
        
        # Check if chunk reached target size
        if current_char_len >= max_chars_per_chunk or i == num_segments - 1:
            chunk_text = "\n".join(s.format_line() for s in current_segments)
            chunks.append(
                TranscriptChunk(
                    chunk_index=chunk_index,
                    start_time=current_segments[0].start,
                    end_time=current_segments[-1].end,
                    text=chunk_text,
                    segments=list(current_segments),
                    estimated_tokens=estimate_tokens(chunk_text)
                )
            )
            chunk_index += 1
            
            # If we've reached the end of all segments, break
            if i >= num_segments - 1:
                break
            
            # Calculate overlap for the next chunk
            overlap_accumulated = 0
            overlap_segments: List[TranscriptSegment] = []
            
            # Walk backwards from current segment to find overlapping segment window
            for rev_seg in reversed(current_segments):
                rev_len = len(rev_seg.format_line()) + 1
                if overlap_accumulated + rev_len <= overlap_chars:
                    overlap_segments.insert(0, rev_seg)
                    overlap_accumulated += rev_len
                else:
                    break
            
            # Reset for next chunk with overlapping segments
            current_segments = list(overlap_segments)
            current_char_len = overlap_accumulated
            
        i += 1
        
    return chunks
