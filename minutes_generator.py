"""
Meeting minutes generator powered by Google Gemini and Pydantic structured outputs.
Implements both single-pass and Map-Reduce workflows for handling long transcripts
with strict faithfulness enforcement.
"""

import os
import re
import json
import time
import logging
import httpx
from typing import Optional, List, Callable
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from models import (
    Transcript,
    MeetingMinutes,
    IntermediateChunkAnalysis,
    ActionItem
)
from chunking import needs_chunking, create_chunks, TranscriptChunk

load_dotenv()
logger = logging.getLogger(__name__)


SYSTEM_PROMPT_FAITHFULNESS = """You are a meticulous, highly disciplined executive meeting secretary.
Your top priority is ABSOLUTE FAITHFULNESS to the meeting transcript.

CRITICAL INTEGRITY RULES:
1. NEVER INVENT OR HALLUCINATE:
   - Never invent decisions, action items, owners, deadlines, or facts.
   - A discussion or proposal is NOT a decision unless explicit agreement was reached.
   - An idea or suggestion is NOT an action item unless explicitly assigned or committed to.

2. MANDATORY "Not specified" RULES:
   - If an action item does not have an explicit owner named in the transcript, you MUST set owner to "Not specified".
   - If an action item does not have an explicit deadline or timeframe stated in the transcript, you MUST set deadline to "Not specified".
   - Do NOT guess names or deduce deadlines based on external context.

3. PRESERVE DIRECT TRANSCRIPT EVIDENCE:
   - Every single action item MUST include verbatim quotes from the transcript with timestamp references (e.g. '[04:12 - 04:30] "I will finish the deployment by Friday"') in the evidence field.
   - If there is no explicit supporting quote in the transcript, DO NOT create the action item.

4. UNRESOLVED QUESTIONS:
   - Identify open questions, unresolved debates, or follow-ups that were left undecided at the end of the conversation.
"""

MAP_CHUNK_PROMPT = """Analyze the following transcript chunk from a meeting.
Extract:
- Summary points of topics discussed in this chunk
- Explicit decisions agreed upon in this chunk
- Action items explicitly committed to in this chunk (with verbatim quotes and timestamps)
- Questions or concerns left unanswered in this chunk

Remember:
- If owner is not explicitly stated, owner = "Not specified"
- If deadline is not explicitly stated, deadline = "Not specified"
- Always quote exact evidence with timestamps.
"""

REDUCE_SYNTHESIS_PROMPT = """Synthesize the intermediate analyses from all meeting chunks into a unified, coherent, and non-redundant set of structured meeting minutes.

Instructions:
1. Deduplicate decisions and action items that appeared across overlapping chunks.
2. Reconcile any conflicting mentions while maintaining strict faithfulness to what was said.
3. Ensure every action item retains its verbatim transcript evidence.
4. If owner or deadline was not specified, strictly keep them as "Not specified".
5. Provide a cohesive, high-level meeting summary and compile all genuine unresolved questions.
"""


def _build_ipv4_httpx_client() -> httpx.Client:
    """
    Creates an httpx.Client that binds to IPv4 (0.0.0.0).
    This prevents [Errno 11001] getaddrinfo failed errors that occur when the
    google-genai SDK's httpx transport tries IPv6 first inside Streamlit's
    async/event-loop context on Windows.
    """
    transport = httpx.HTTPTransport(
        local_address="0.0.0.0",  # Force IPv4 binding
        retries=3
    )
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(60.0, connect=15.0),
        follow_redirects=True
    )


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Initializes and returns Google Gemini client.
    Uses a custom IPv4-pinned httpx transport to prevent DNS resolution
    failures (Errno 11001) that occur in Streamlit's async context on Windows.
    """
    resolved_key = (
        api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if not resolved_key or resolved_key.strip() in ["", "your_gemini_api_key_here"]:
        raise ValueError(
            "Gemini API key is missing. Please provide your API key in the application "
            "or set GEMINI_API_KEY in your .env file."
        )
    return genai.Client(
        api_key=resolved_key.strip(),
        http_options=types.HttpOptions(
            httpx_client=_build_ipv4_httpx_client(),
            timeout=60000  # milliseconds
        )
    )


def _clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences (```json ... ```) if present in model output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _is_retryable_error(exc: Exception) -> bool:
    """Returns True for transient Gemini API errors worth retrying."""
    if isinstance(exc, genai_errors.ServerError):
        return True  # 503 UNAVAILABLE, 500 INTERNAL
    if isinstance(exc, genai_errors.ClientError):
        # 429 RESOURCE_EXHAUSTED (rate limit) is retryable
        return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
    # Also catch generic network/connection errors
    if isinstance(exc, (ConnectionError, TimeoutError, httpx.TimeoutException,
                        httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return False


def _gemini_generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_attempts: int = 4
) -> str:
    """
    Calls client.models.generate_content with exponential backoff retry.
    Retries on 503 UNAVAILABLE, 429 RESOURCE_EXHAUSTED, 500, and network errors.
    Waits: 5s → 15s → 45s between attempts (jittered).
    Returns the raw response text.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_error(exc):
                raise  # Non-retryable — bubble up immediately
            wait_secs = min(5 * (3 ** (attempt - 1)), 60)  # 5, 15, 45, capped at 60
            logger.warning(
                f"Gemini API transient error (attempt {attempt}/{max_attempts}): {exc}. "
                f"Retrying in {wait_secs}s..."
            )
            if attempt < max_attempts:
                time.sleep(wait_secs)

    raise RuntimeError(
        f"Gemini API failed after {max_attempts} attempts. Last error: {last_exc}"
    )



def generate_minutes_single_pass(
    client: genai.Client,
    transcript: Transcript,
    model: str = "gemini-3.8-flash"
) -> MeetingMinutes:
    """Generates structured meeting minutes in a single Gemini call for transcripts within context limits."""
    formatted_transcript = transcript.formatted_transcript

    prompt_content = (
        f"Please generate structured meeting minutes based strictly on the following transcript:\n\n"
        f"--- TRANSCRIPT START ---\n"
        f"{formatted_transcript}\n"
        f"--- TRANSCRIPT END ---\n"
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT_FAITHFULNESS,
        response_mime_type="application/json",
        response_schema=MeetingMinutes,
        temperature=0.1
    )

    try:
        raw_text = _gemini_generate_with_retry(
            client=client, model=model, contents=prompt_content, config=config
        )
        cleaned_json = _clean_json_response(raw_text)
        return MeetingMinutes.model_validate_json(cleaned_json)
    except Exception as e:
        logger.error(f"Error in single pass Gemini minutes generation: {e}", exc_info=True)
        raise RuntimeError(f"Gemini API error: {str(e)}") from e



def _analyze_chunk_map(
    client: genai.Client,
    chunk: TranscriptChunk,
    total_chunks: int,
    model: str = "gemini-3.8-flash"
) -> IntermediateChunkAnalysis:
    """Map step: analyzes a single chunk to extract candidate decisions, actions, and summaries."""
    prompt_content = (
        f"Chunk {chunk.chunk_index + 1} of {total_chunks} "
        f"({chunk.timestamp_header}):\n\n"
        f"{chunk.text}\n"
    )

    config = types.GenerateContentConfig(
        system_instruction=f"{SYSTEM_PROMPT_FAITHFULNESS}\n\n{MAP_CHUNK_PROMPT}",
        response_mime_type="application/json",
        response_schema=IntermediateChunkAnalysis,
        temperature=0.1
    )

    try:
        raw_text = _gemini_generate_with_retry(
            client=client, model=model, contents=prompt_content, config=config
        )
        cleaned_json = _clean_json_response(raw_text)
        parsed = IntermediateChunkAnalysis.model_validate_json(cleaned_json)
        parsed.chunk_index = chunk.chunk_index
        return parsed
    except Exception as e:
        logger.error(f"Error in map chunk {chunk.chunk_index}: {e}", exc_info=True)
        raise RuntimeError(f"Gemini chunk analysis error: {str(e)}") from e



def _synthesize_reduce(
    client: genai.Client,
    intermediate_analyses: List[IntermediateChunkAnalysis],
    model: str = "gemini-3.8-flash"
) -> MeetingMinutes:
    """Reduce step: synthesizes all intermediate chunk analyses into final minutes."""
    data_dump = [item.model_dump() for item in intermediate_analyses]
    intermediate_json = json.dumps(data_dump, indent=2)

    prompt_content = (
        f"Here are the intermediate analyses collected from all meeting transcript chunks:\n\n"
        f"{intermediate_json}\n\n"
        f"Synthesize these into the final MeetingMinutes structure, removing redundancies "
        f"and maintaining strict transcript faithfulness."
    )

    config = types.GenerateContentConfig(
        system_instruction=f"{SYSTEM_PROMPT_FAITHFULNESS}\n\n{REDUCE_SYNTHESIS_PROMPT}",
        response_mime_type="application/json",
        response_schema=MeetingMinutes,
        temperature=0.1
    )

    try:
        raw_text = _gemini_generate_with_retry(
            client=client, model=model, contents=prompt_content, config=config
        )
        cleaned_json = _clean_json_response(raw_text)
        return MeetingMinutes.model_validate_json(cleaned_json)
    except Exception as e:
        logger.error(f"Error in reduce step: {e}", exc_info=True)
        raise RuntimeError(f"Gemini synthesis reduce error: {str(e)}") from e



def generate_minutes_map_reduce(
    client: genai.Client,
    transcript: Transcript,
    model: str = "gemini-3.8-flash",
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> MeetingMinutes:
    """Processes long transcripts using overlapping chunking and Map-Reduce synthesis."""
    chunks = create_chunks(transcript, max_chars_per_chunk=10000, overlap_chars=1500)
    total_chunks = len(chunks)

    if progress_callback:
        progress_callback(f"Split transcript into {total_chunks} overlapping chunks.", 0.05)

    intermediate_results: List[IntermediateChunkAnalysis] = []

    # Map Step
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            fraction = 0.1 + 0.6 * (idx / total_chunks)
            progress_callback(
                f"Analyzing chunk {idx + 1}/{total_chunks} ({chunk.timestamp_header})...",
                fraction
            )

        analysis = _analyze_chunk_map(client, chunk, total_chunks, model=model)
        intermediate_results.append(analysis)

    # Reduce Step
    if progress_callback:
        progress_callback("Synthesizing chunk results into final meeting minutes...", 0.8)

    final_minutes = _synthesize_reduce(client, intermediate_results, model=model)

    if progress_callback:
        progress_callback("Minutes generation complete!", 1.0)

    return final_minutes


def generate_meeting_minutes(
    transcript: Transcript,
    api_key: Optional[str] = None,
    model: str = "gemini-3.8-flash",
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> MeetingMinutes:
    """
    Main entry point for generating meeting minutes from a Transcript using Gemini.
    Automatically decides whether single-pass or Map-Reduce is required.
    """
    client = get_gemini_client(api_key=api_key)

    if needs_chunking(transcript, max_chars_threshold=12000):
        if progress_callback:
            progress_callback("Long transcript detected. Initiating Map-Reduce pipeline with Gemini...", 0.02)
        return generate_minutes_map_reduce(
            client=client,
            transcript=transcript,
            model=model,
            progress_callback=progress_callback
        )
    else:
        if progress_callback:
            progress_callback("Generating meeting minutes via Gemini (single-pass)...", 0.3)
        minutes = generate_minutes_single_pass(client=client, transcript=transcript, model=model)
        if progress_callback:
            progress_callback("Minutes generated successfully!", 1.0)
        return minutes
