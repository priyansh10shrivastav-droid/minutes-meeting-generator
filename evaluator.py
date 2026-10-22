"""
Faithfulness and evidence verification module.
Evaluates generated action items against the original transcript using Google Gemini
or deterministic lexical heuristics to detect hallucinations, unsupported owners,
fabricated deadlines, or ungrounded claims.
"""

import re
import logging
import httpx
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from models import (
    Transcript,
    ActionItem,
    VerificationStatus,
    FaithfulnessVerificationResult,
    MeetingMinutes
)
# Reuse IPv4 client factory and retry wrapper from minutes_generator
from minutes_generator import _build_ipv4_httpx_client, _gemini_generate_with_retry

logger = logging.getLogger(__name__)


class LLMEvalResponse(BaseModel):
    """Schema for LLM faithfulness verification of an action item."""
    status: VerificationStatus = Field(
        ...,
        description="SUPPORTED if fully grounded, PARTIALLY_SUPPORTED if partially grounded or loosely inferred, UNSUPPORTED if fabricated or contradictory"
    )
    quote_found: bool = Field(
        ...,
        description="True if the supporting quote is present in the transcript"
    )
    owner_grounded: bool = Field(
        ...,
        description="True if owner is explicitly named in transcript, or owner is 'Not specified'"
    )
    deadline_grounded: bool = Field(
        ...,
        description="True if deadline is explicitly stated in transcript, or deadline is 'Not specified'"
    )
    explanation: str = Field(
        ...,
        description="Clear diagnostic reasoning explaining the assigned verification status"
    )


def clean_quote_text(quote: str) -> str:
    """Removes timestamp brackets and quotation marks for substring matching."""
    cleaned = re.sub(r"\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]", "", quote)
    cleaned = cleaned.strip(" '\"“”‘’\n\t")
    return cleaned


def _clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences if present in output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def lexical_quote_check(quote: str, transcript_text: str) -> Tuple[bool, float]:
    """
    Checks if quote or significant n-grams of quote appear in transcript.
    Returns (found_boolean, match_ratio).
    """
    cleaned_quote = clean_quote_text(quote).lower()
    lower_transcript = transcript_text.lower()

    if not cleaned_quote:
        return False, 0.0

    if cleaned_quote in lower_transcript:
        return True, 1.0

    # Test word overlap ratio for fuzzy quote matches
    quote_words = [w for w in re.findall(r"\b\w+\b", cleaned_quote) if len(w) > 2]
    if not quote_words:
        return False, 0.0

    matched_words = sum(1 for w in quote_words if w in lower_transcript)
    ratio = matched_words / len(quote_words)
    return (ratio >= 0.75), ratio


def verify_action_item_heuristic(
    item: ActionItem,
    index: int,
    full_transcript_text: str
) -> FaithfulnessVerificationResult:
    """
    Deterministic rule-based verification fallback when LLM is unavailable.
    Inspects quote existence, owner presence, and deadline presence in transcript text.
    """
    quote_found, ratio = lexical_quote_check(item.evidence, full_transcript_text)
    lower_transcript = full_transcript_text.lower()

    # Check owner
    owner_grounded = True
    if item.owner.strip().lower() not in ["not specified", "unspecified", "none", "n/a", "tbd"]:
        owner_name = item.owner.strip().lower()
        if owner_name not in lower_transcript:
            owner_grounded = False

    # Check deadline
    deadline_grounded = True
    if item.deadline.strip().lower() not in ["not specified", "unspecified", "none", "n/a", "tbd"]:
        deadline_str = item.deadline.strip().lower()
        deadline_words = [w for w in re.findall(r"\b\w+\b", deadline_str) if len(w) > 2]
        if deadline_words and not any(w in lower_transcript for w in deadline_words):
            deadline_grounded = False

    # Determine status
    if not quote_found:
        status: VerificationStatus = "UNSUPPORTED"
        explanation = (
            "The cited evidence quote could not be verified in the transcript. "
            "This action item may be ungrounded or hallucinated."
        )
    elif not owner_grounded or not deadline_grounded:
        status = "PARTIALLY_SUPPORTED"
        issues = []
        if not owner_grounded:
            issues.append(f"owner '{item.owner}' was not explicitly found in transcript")
        if not deadline_grounded:
            issues.append(f"deadline '{item.deadline}' was not explicitly found in transcript")
        explanation = f"Action item is mentioned, but {', '.join(issues)}."
    else:
        status = "SUPPORTED"
        explanation = "The action item, owner, deadline, and cited transcript evidence are fully grounded in the transcript."

    return FaithfulnessVerificationResult(
        action_item_index=index,
        action_item_description=item.description,
        status=status,
        explanation=explanation,
        quote_found_in_transcript=quote_found,
        owner_faithfully_supported=owner_grounded,
        deadline_faithfully_supported=deadline_grounded
    )


def verify_action_item_llm(
    client: genai.Client,
    item: ActionItem,
    index: int,
    transcript_text: str,
    model: str = "gemini-3.8-flash"
) -> FaithfulnessVerificationResult:
    """Uses Gemini to perform natural language inference (NLI) check of action item against transcript."""
    system_prompt = (
        "You are an impartial, highly rigorous verification auditor. Your task is to verify whether "
        "a generated action item is faithfully supported by the meeting transcript.\n\n"
        "Criteria:\n"
        "1. SUPPORTED: The task is explicitly committed to, the cited quote is genuine, and owner/deadline are either explicitly mentioned or marked 'Not specified'.\n"
        "2. PARTIALLY_SUPPORTED: The topic was discussed, but the owner or deadline was assumed without agreement, or the quote only partially supports it.\n"
        "3. UNSUPPORTED: The action item is fabricated, contrary to the transcript, or the cited quote does not support it.\n\n"
        "Remember: An owner or deadline of 'Not specified' is FAITHFUL if the transcript does not specify one."
    )

    user_prompt = (
        f"--- TRANSCRIPT CONTEXT ---\n{transcript_text[:12000]}\n--- END TRANSCRIPT ---\n\n"
        f"--- ACTION ITEM TO AUDIT ---\n"
        f"Index: {index + 1}\n"
        f"Description: {item.description}\n"
        f"Owner: {item.owner}\n"
        f"Deadline: {item.deadline}\n"
        f"Cited Evidence: {item.evidence}\n\n"
        f"Evaluate the faithfulness of this action item."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=LLMEvalResponse,
        temperature=0.0
    )

    try:
        raw_text = _gemini_generate_with_retry(
            client=client, model=model, contents=user_prompt, config=config
        )
        cleaned_json = _clean_json_response(raw_text)
        parsed = LLMEvalResponse.model_validate_json(cleaned_json)
        return FaithfulnessVerificationResult(
            action_item_index=index,
            action_item_description=item.description,
            status=parsed.status,
            explanation=parsed.explanation,
            quote_found_in_transcript=parsed.quote_found,
            owner_faithfully_supported=parsed.owner_grounded,
            deadline_faithfully_supported=parsed.deadline_grounded
        )
    except Exception as e:
        logger.warning(f"Gemini verification failed for item {index}: {e}. Falling back to heuristic check.")
        return verify_action_item_heuristic(item, index, transcript_text)



def verify_meeting_minutes_faithfulness(
    minutes: MeetingMinutes,
    transcript: Transcript,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.8-flash"
) -> List[FaithfulnessVerificationResult]:
    """
    Verifies all action items in meeting minutes against the transcript.
    Updates each action item's verification_status and verification_explanation in-place.
    Returns the comprehensive list of verification results.
    """
    verification_results: List[FaithfulnessVerificationResult] = []
    full_transcript_text = transcript.formatted_transcript

    for idx, item in enumerate(minutes.action_items):
        if client:
            result = verify_action_item_llm(
                client=client,
                item=item,
                index=idx,
                transcript_text=full_transcript_text,
                model=model
            )
        else:
            result = verify_action_item_heuristic(
                item=item,
                index=idx,
                full_transcript_text=full_transcript_text
            )

        # Update the ActionItem model with evaluation results
        item.verification_status = result.status
        item.verification_explanation = result.explanation
        verification_results.append(result)

    return verification_results
