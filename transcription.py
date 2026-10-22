"""
Speech-to-text transcription module using faster-whisper.
Transcribes audio files (MP3, WAV, M4A) into timestamped segments and structured Transcript models.
"""

import os
import tempfile
import logging
from typing import Optional, Callable
from faster_whisper import WhisperModel
from models import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

# Cache loaded Whisper models to avoid re-allocating memory on repeated runs
_MODEL_CACHE = {}


def get_whisper_model(
    model_size: str = "base",
    device: Optional[str] = None,
    compute_type: Optional[str] = None
) -> WhisperModel:
    """
    Retrieves or initializes a cached faster-whisper model.
    Defaults to CPU with int8 quantization for high-speed local inference.
    """
    if device is None:
        device = "cpu"
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"

    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key not in _MODEL_CACHE:
        logger.info(f"Loading faster-whisper model '{model_size}' on {device} ({compute_type})...")
        _MODEL_CACHE[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
    return _MODEL_CACHE[cache_key]


def transcribe_audio(
    audio_source: any,
    model_size: str = "base",
    device: Optional[str] = None,
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Transcript:
    """
    Transcribes audio into timestamped segments and consolidated plain text.

    Args:
        audio_source: File path (str), file-like object, or Streamlit UploadedFile.
        model_size: 'tiny', 'base', 'small', 'medium'.
        device: 'cpu' or 'cuda'.
        language: Optional language code (e.g., 'en', None for auto-detect).
        progress_callback: Optional function(message: str, fraction: float) for UI updates.

    Returns:
        Transcript object containing all segments and full text.
    """
    temp_path = None
    try:
        # Resolve audio source to a valid file path
        if isinstance(audio_source, str) and os.path.exists(audio_source):
            audio_path = audio_source
        else:
            # Handle UploadedFile or file-like bytes
            suffix = ".wav"
            if hasattr(audio_source, "name") and audio_source.name:
                ext = os.path.splitext(audio_source.name)[1].lower()
                if ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
                    suffix = ext
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                if hasattr(audio_source, "read"):
                    if hasattr(audio_source, "seek"):
                        audio_source.seek(0)
                    temp_file.write(audio_source.read())
                elif isinstance(audio_source, (bytes, bytearray)):
                    temp_file.write(audio_source)
                else:
                    raise ValueError(f"Unsupported audio source type: {type(audio_source)}")
            audio_path = temp_path

        if progress_callback:
            progress_callback(f"Loading Whisper model ({model_size})...", 0.1)

        model = get_whisper_model(model_size=model_size, device=device)

        if progress_callback:
            progress_callback("Running speech-to-text transcription...", 0.3)

        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        detected_language = info.language
        audio_duration = info.duration or 0.0

        segments: list[TranscriptSegment] = []
        text_lines: list[str] = []

        total_duration = max(1.0, audio_duration)

        for seg in segments_iter:
            segment_obj = TranscriptSegment(
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text.strip(),
                speaker=None
            )
            segments.append(segment_obj)
            text_lines.append(seg.text.strip())

            if progress_callback and audio_duration > 0:
                fraction = min(0.95, 0.3 + 0.65 * (seg.end / total_duration))
                progress_callback(
                    f"Transcribing: {segment_obj.timestamp_str} ({len(segments)} segments)",
                    fraction
                )

        full_text = " ".join(text_lines)

        if progress_callback:
            progress_callback("Transcription complete!", 1.0)

        return Transcript(
            segments=segments,
            full_text=full_text,
            language=detected_language,
            duration_seconds=round(audio_duration, 2)
        )

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Audio transcription error: {str(e)}") from e

    finally:
        # Clean up temporary audio file if created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
