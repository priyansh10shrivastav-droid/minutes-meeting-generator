"""
MeetNote — AI Meeting Minutes Generator
Supports two completely independent transcription workflows:
1. Upload Existing Audio (MP3, WAV, M4A -> faster-whisper STT -> Editable Transcript)
2. Live Meeting Transcription (Browser Mic -> Real-time Streaming Web Speech STT + Live Waveform -> Editable Transcript)
Both modes use the same meeting-minutes generation engine with Gemini & Faithfulness Verification.
"""

import os
import io
import json
import time
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from models import MeetingMinutes, Transcript, TranscriptSegment, ActionItem
from transcription import transcribe_audio
from minutes_generator import generate_meeting_minutes, get_gemini_client
from evaluator import verify_meeting_minutes_faithfulness
from export_utils import (
    generate_minutes_pdf,
    generate_minutes_txt,
    generate_minutes_markdown
)

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetNote — AI Meeting Minutes",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Declare Custom Live Transcription Component ─────────────────────────────
_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "live_transcription")
live_transcription_comp = components.declare_component("live_transcription", path=_component_dir)

# ─── Session State Initialization ────────────────────────────────────────────
default_state = {
    # Active workflow mode: "upload" or "live"
    "active_mode": "upload",
    # Mode 1: Upload state
    "uploaded_audio_bytes": None,
    "uploaded_audio_name": None,
    "upload_transcript_text": "",
    # Mode 2: Live state
    "live_transcript_text": "",
    "live_duration_seconds": 0,
    "live_is_stopped": False,
    # Shared generation outputs
    "meeting_minutes": None,
    "verification_results": None,
    "active_transcript_obj": None,
    # Recent meetings list
    "recent_meetings": [
        {"title": "Project Sync", "date": "Nov 19, 2026", "duration": "12 min"},
        {"title": "Client Discussion", "date": "Nov 17, 2026", "duration": "24 min"},
        {"title": "Intern Review", "date": "Nov 15, 2026", "duration": "18 min"},
        {"title": "Team Standup", "date": "Nov 13, 2026", "duration": "9 min"},
        {"title": "Development Update", "date": "Nov 11, 2026", "duration": "27 min"},
    ]
}

for key, val in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─── Custom CSS for High-Fidelity Design ──────────────────────────────────────
st.markdown("""
<style>
/* Base typography and background */
body, [data-testid="stAppViewContainer"] {
    background-color: #F8F9FA !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827;
}

[data-testid="stHeader"] {
    display: none;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
    padding-top: 1rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background-color: #111827 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Header badge */
.ai-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #4B5563;
    margin-bottom: 8px;
}

.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.main-sub {
    font-size: 0.95rem;
    color: #6B7280;
    margin-bottom: 22px;
}

/* Card components */
.mn-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    margin-bottom: 18px;
}

/* Mode Selector Cards */
.mode-card {
    background: #FFFFFF;
    border: 1.5px solid #E5E7EB;
    border-radius: 14px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    cursor: pointer;
    transition: all 0.15s ease;
}

.mode-card.active {
    border-color: #111827;
    box-shadow: 0 4px 12px rgba(17, 24, 39, 0.08);
}

.mode-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: #F3F4F6;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
}

.mode-card.active .mode-icon {
    background: #EEF2FF;
    color: #4F46E5;
}

/* Drag and Drop Container */
.dropzone-box {
    border: 2px dashed #D1D5DB;
    border-radius: 16px;
    padding: 36px 20px;
    text-align: center;
    background: #FAFAFA;
    transition: border-color 0.2s;
    margin-bottom: 16px;
}

/* Primary generate button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #111827 !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(17, 24, 39, 0.25) !important;
    width: 100% !important;
    transition: transform 0.1s ease, background 0.2s !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1F2937 !important;
    transform: translateY(-1px);
}

/* Metric stat cards */
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
}
.stat-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #6B7280;
    letter-spacing: 0.5px;
}
.stat-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #111827;
    margin-top: 2px;
}

/* Recent Meetings list items */
.recent-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 11px 0;
    border-bottom: 1px solid #F3F4F6;
}
.recent-item:last-child {
    border-bottom: none;
}
.recent-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #111827;
}
.recent-meta {
    font-size: 0.76rem;
    color: #6B7280;
    margin-top: 2px;
}

/* Evidence block */
.evidence-quote {
    background: #F8FAFC;
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    font-size: 0.85rem;
    color: #334155;
    font-style: italic;
    margin-top: 6px;
}

/* Status Badges */
.badge-sup {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #BBF7D0;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.76rem;
    font-weight: 700;
}
.badge-part {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.76rem;
    font-weight: 700;
}
.badge-unsup {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.76rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:12px 6px 18px;border-bottom:1px solid #F3F4F6;margin-bottom:14px;">
        <div style="font-size:24px;">🎙️</div>
        <div>
            <div style="font-size:1.15rem;font-weight:800;color:#111827;letter-spacing:-0.3px;">MeetNote</div>
            <div style="font-size:0.7rem;color:#6B7280;font-weight:500;">AI Meeting Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        ["🏠  Home", "📁  Meetings", "📑  Templates", "⚙️  Settings"],
        label_visibility="collapsed",
        index=0
    )

    st.write("")
    st.write("")

    # Sidebar settings info
    if "Settings" in nav_selection:
        st.markdown("**Model Configuration**")
        selected_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.8-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash-latest"],
            index=0
        )
        whisper_size = st.selectbox(
            "Whisper STT Model Size",
            ["tiny", "base", "small", "medium"],
            index=1
        )
    else:
        selected_model = "gemini-3.8-flash"
        whisper_size = "base"

    st.markdown("""
    <div style="position:fixed;bottom:20px;display:flex;align-items:center;gap:8px;font-size:0.85rem;color:#6B7280;">
        <span>❓</span> Help & Documentation
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content Layout ──────────────────────────────────────────────────────
col_left, col_right = st.columns([2.5, 1.1], gap="large")

with col_left:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="ai-badge">✨ AI Meeting Assistant</div>
    <div class="main-title">Generate Meeting Minutes</div>
    <div class="main-sub">Transcribe, summarize, and turn discussions into actionable notes.</div>
    """, unsafe_allow_html=True)

    # ── Mode Selector Cards (Side-by-Side) ─────────────────────────────────────
    m1_col, m2_col = st.columns(2)

    with m1_col:
        is_upload_active = st.session_state["active_mode"] == "upload"
        active_class = "active" if is_upload_active else ""
        card_html_1 = f"""
        <div class="mode-card {active_class}">
            <div class="mode-icon">📄</div>
            <div>
                <div style="font-weight:700;font-size:0.98rem;color:#111827;">Upload Recording</div>
                <div style="font-size:0.78rem;color:#6B7280;margin-top:2px;">Use an existing audio file</div>
            </div>
        </div>
        """
        st.markdown(card_html_1, unsafe_allow_html=True)
        if st.button("Select Upload Mode", key="btn_sel_upload", use_container_width=True):
            st.session_state["active_mode"] = "upload"
            st.rerun()

    with m2_col:
        is_live_active = st.session_state["active_mode"] == "live"
        active_class = "active" if is_live_active else ""
        card_html_2 = f"""
        <div class="mode-card {active_class}">
            <div class="mode-icon">🎙️</div>
            <div>
                <div style="font-weight:700;font-size:0.98rem;color:#111827;">Live Meeting</div>
                <div style="font-size:0.78rem;color:#6B7280;margin-top:2px;">Transcribe in real-time</div>
            </div>
        </div>
        """
        st.markdown(card_html_2, unsafe_allow_html=True)
        if st.button("Select Live Mode", key="btn_sel_live", use_container_width=True):
            st.session_state["active_mode"] = "live"
            st.rerun()

    st.write("")

    # =========================================================================
    # WORKFLOW 1: UPLOAD EXISTING AUDIO (Independent)
    # =========================================================================
    if st.session_state["active_mode"] == "upload":
        st.markdown('<div class="mn-card">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Existing Recording")
        st.markdown("<div style='font-size:0.85rem;color:#6B7280;margin-top:-6px;margin-bottom:14px;'>Upload pre-recorded meeting audio (.mp3, .wav, .m4a). No microphone required.</div>", unsafe_allow_html=True)

        audio_file = st.file_uploader(
            "Drop your audio file here",
            type=["mp3", "wav", "m4a"],
            label_visibility="collapsed",
            help="Supported: MP3, WAV, M4A up to 200MB"
        )

        if audio_file is not None:
            st.session_state["uploaded_audio_bytes"] = audio_file.getvalue()
            st.session_state["uploaded_audio_name"] = audio_file.name

        # Audio file information and player
        if st.session_state["uploaded_audio_bytes"]:
            fname = st.session_state["uploaded_audio_name"]
            size_mb = round(len(st.session_state["uploaded_audio_bytes"]) / (1024 * 1024), 2)

            file_info_col1, file_info_col2 = st.columns([3, 1])
            with file_info_col1:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;padding:10px 14px;margin-top:6px;margin-bottom:10px;">
                    <span style="font-size:20px;">🎵</span>
                    <div>
                        <div style="font-weight:600;font-size:0.9rem;color:#111827;">{fname}</div>
                        <div style="font-size:0.75rem;color:#6B7280;">File Size: {size_mb} MB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with file_info_col2:
                if st.button("🗑️ Remove File", key="btn_remove_audio"):
                    st.session_state["uploaded_audio_bytes"] = None
                    st.session_state["uploaded_audio_name"] = None
                    st.session_state["upload_transcript_text"] = ""
                    st.rerun()

            st.audio(st.session_state["uploaded_audio_bytes"])

            # Transcribe Audio button
            st.write("")
            if st.button("🎙️ Transcribe Audio Recording", key="btn_transcribe_upload", type="secondary", use_container_width=True):
                with st.spinner("Transcribing audio using local faster-whisper STT with timestamps..."):
                    try:
                        transcript_obj = transcribe_audio(
                            st.session_state["uploaded_audio_bytes"],
                            model_size=whisper_size,
                            device="cpu"
                        )
                        st.session_state["upload_transcript_text"] = transcript_obj.formatted_transcript
                        st.session_state["active_transcript_obj"] = transcript_obj
                        st.success(f"✅ Transcription complete! ({round(transcript_obj.duration_seconds, 1)}s audio detected)")
                    except Exception as e:
                        st.error(f"❌ Transcription error: {str(e)}")

        # Editable Transcript Section
        st.write("")
        st.markdown("**Extracted Transcript (Review & Edit):**")
        edited_upload_transcript = st.text_area(
            "Review & edit transcript:",
            value=st.session_state.get("upload_transcript_text", ""),
            height=180,
            label_visibility="collapsed",
            placeholder="Uploaded audio transcript will appear here after clicking 'Transcribe Audio Recording'. You can review and edit it before generating minutes."
        )
        st.session_state["upload_transcript_text"] = edited_upload_transcript
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # WORKFLOW 2: LIVE MEETING TRANSCRIPTION (Independent)
    # =========================================================================
    elif st.session_state["active_mode"] == "live":
        st.markdown('<div class="mn-card">', unsafe_allow_html=True)
        st.markdown("### 🎙️ Live Meeting Transcription")
        st.markdown("<div style='font-size:0.85rem;color:#6B7280;margin-top:-6px;margin-bottom:14px;'>Streams live microphone audio in real time using Web Speech API with audio waveform visualization.</div>", unsafe_allow_html=True)

        # Render custom component for real-time live transcription
        live_result = live_transcription_comp()

        if live_result and isinstance(live_result, dict):
            if live_result.get("transcript"):
                st.session_state["live_transcript_text"] = live_result["transcript"]
            if "duration_seconds" in live_result:
                st.session_state["live_duration_seconds"] = live_result["duration_seconds"]
            if live_result.get("is_stopped"):
                st.session_state["live_is_stopped"] = True

        st.write("")
        st.markdown("**Live Meeting Transcript (Review & Edit):**")
        edited_live_transcript = st.text_area(
            "Review & edit live transcript:",
            value=st.session_state.get("live_transcript_text", ""),
            height=180,
            label_visibility="collapsed",
            placeholder="Live recognized speech will stream here in real-time. Once stopped, you can edit names, terms, or dates here before generating minutes."
        )
        st.session_state["live_transcript_text"] = edited_live_transcript
        st.markdown('</div>', unsafe_allow_html=True)


    # =========================================================================
    # SHARED OPTIONS ROW & GENERATE BUTTON
    # =========================================================================
    st.markdown('<div class="mn-card" style="padding:18px 22px;">', unsafe_allow_html=True)
    o1, o2, o3 = st.columns([1, 1, 1.4])
    with o1:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#6B7280;text-transform:uppercase;'>Output Language</div>", unsafe_allow_html=True)
        output_lang = st.selectbox(
            "lang_sel",
            ["English", "Hindi", "Spanish", "French", "German"],
            label_visibility="collapsed"
        )
    with o2:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#6B7280;text-transform:uppercase;'>Meeting Type</div>", unsafe_allow_html=True)
        meeting_type = st.selectbox(
            "type_sel",
            ["General", "Standup", "Review", "Brainstorm", "Planning"],
            label_visibility="collapsed"
        )
    with o3:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#6B7280;text-transform:uppercase;'>Include in Minutes</div>", unsafe_allow_html=True)
        c_inc1, c_inc2 = st.columns(2)
        with c_inc1:
            inc_summary = st.checkbox("Summary", value=True, key="chk_inc_sum")
            inc_actions = st.checkbox("Action Items", value=True, key="chk_inc_act")
        with c_inc2:
            inc_keypts = st.checkbox("Key Points", value=True, key="chk_inc_kp")
            inc_next = st.checkbox("Next Meeting", value=True, key="chk_inc_next")

    st.write("")
    generate_clicked = st.button("✨  Generate Meeting Minutes", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Pipeline Trigger ──────────────────────────────────────────────────────
    if generate_clicked:
        # Determine current active transcript
        if st.session_state["active_mode"] == "upload":
            chosen_text = st.session_state.get("upload_transcript_text", "").strip()
            source_label = "Uploaded Audio"
        else:
            chosen_text = st.session_state.get("live_transcript_text", "").strip()
            source_label = "Live Meeting"

        if not chosen_text:
            if st.session_state["active_mode"] == "upload":
                st.error("⚠️ No transcript available. Please upload an audio file and click **Transcribe Audio Recording**, or paste text into the transcript box.")
            else:
                st.error("⚠️ No live transcript recorded. Please click **Start Live Meeting**, speak into your microphone, or paste/type text into the editor.")
            st.stop()

        # Build Transcript object
        lines = [l.strip() for l in chosen_text.split("\n") if l.strip()]
        segments = [
            TranscriptSegment(start=0.0, end=0.0, text=line)
            for line in lines
        ]
        active_transcript = Transcript(
            segments=segments,
            full_text=chosen_text,
            language="en",
            duration_seconds=st.session_state.get("live_duration_seconds", 0.0)
        )
        st.session_state["active_transcript_obj"] = active_transcript

        # Execute shared generation pipeline
        with st.status("🚀 Processing Meeting Minutes...", expanded=True) as status_box:
            progress_bar = st.progress(0.1)
            status_text = st.empty()

            def pipeline_progress(msg: str, frac: float):
                status_text.markdown(f"**{msg}**")
                progress_bar.progress(min(1.0, max(0.0, frac)))

            try:
                # 1. Generate Structured Minutes via Gemini
                minutes = generate_meeting_minutes(
                    transcript=active_transcript,
                    model=selected_model,
                    language=output_lang,
                    meeting_type=meeting_type,
                    include_features={
                        "Summary": inc_summary,
                        "Action Items": inc_actions,
                        "Key Points": inc_keypts,
                        "Next Meeting": inc_next
                    },
                    progress_callback=pipeline_progress
                )
                st.session_state["meeting_minutes"] = minutes

                # 2. Run Faithfulness Verification
                pipeline_progress("Auditing action items for faithfulness against transcript...", 0.85)
                client = get_gemini_client()
                audit_report = verify_meeting_minutes_faithfulness(
                    minutes=minutes,
                    transcript=active_transcript,
                    client=client,
                    model=selected_model
                )
                st.session_state["verification_results"] = audit_report

                # Prepend to recent meetings
                st.session_state["recent_meetings"].insert(0, {
                    "title": minutes.meeting_title,
                    "date": "Today",
                    "duration": f"{len(lines)} lines"
                })

                pipeline_progress("Completed successfully!", 1.0)
                status_box.update(label="✅ Meeting Minutes Generated & Audited!", state="complete", expanded=False)
                st.rerun()

            except Exception as exc:
                status_box.update(label="❌ Pipeline Execution Error", state="error", expanded=True)
                st.error(f"Error generating minutes: {str(exc)}")


    # =========================================================================
    # SHARED RESULTS OUTPUT SYSTEM
    # =========================================================================
    if st.session_state.get("meeting_minutes"):
        mins: MeetingMinutes = st.session_state["meeting_minutes"]
        audit = st.session_state.get("verification_results", {})

        st.write("")
        st.markdown("## 📊 Meeting Intelligence Output")

        # Top stat metrics
        total_actions = len(mins.action_items)
        supported_count = sum(1 for a in mins.action_items if a.verification_status == "SUPPORTED")
        faith_score = round((supported_count / max(1, total_actions)) * 100) if total_actions else 100

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f'<div class="stat-card"><div class="stat-title">Key Decisions</div><div class="stat-value">{len(mins.key_decisions)}</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="stat-card"><div class="stat-title">Action Items</div><div class="stat-value">{total_actions}</div></div>', unsafe_allow_html=True)
        with m_col3:
            st.markdown(f'<div class="stat-card"><div class="stat-title">Discussion Points</div><div class="stat-value">{len(mins.key_discussion_points)}</div></div>', unsafe_allow_html=True)
        with m_col4:
            st.markdown(f'<div class="stat-card"><div class="stat-title">Faithfulness Score</div><div class="stat-value" style="color:#10B981;">{faith_score}%</div></div>', unsafe_allow_html=True)

        st.write("")

        # 4 Output Tabs
        res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(["📋  Minutes", "🛡️  Faithfulness Audit", "📜  Transcript", "💾  Export"])

        # Tab 1: Minutes
        with res_tab1:
            st.markdown(f"### {mins.meeting_title}")
            if mins.meeting_date and mins.meeting_date != "Not specified":
                st.caption(f"📅 Date: {mins.meeting_date}")

            st.markdown("#### 1. Executive Summary")
            st.info(mins.meeting_summary)

            if mins.key_discussion_points:
                st.markdown("#### 2. Key Discussion Points")
                for pt in mins.key_discussion_points:
                    st.markdown(f"• {pt}")

            st.markdown("#### 3. Decisions Made")
            if mins.key_decisions:
                for d in mins.key_decisions:
                    st.markdown(f"✅ **{d}**")
            else:
                st.write("_No explicit decisions recorded._")

            st.markdown("#### 4. Action Items & Responsible Persons")
            if mins.action_items:
                for idx, item in enumerate(mins.action_items, 1):
                    with st.container():
                        st.markdown(f"**{idx}. {item.description}**")
                        c_a, c_b, c_c = st.columns(3)
                        with c_a:
                            st.markdown(f"👤 **Responsible:** `{item.owner}`")
                        with c_b:
                            st.markdown(f"📅 **Deadline:** `{item.deadline}`")
                        with c_c:
                            if item.verification_status == "SUPPORTED":
                                st.markdown('<span class="badge-sup">🟢 SUPPORTED</span>', unsafe_allow_html=True)
                            elif item.verification_status == "PARTIALLY_SUPPORTED":
                                st.markdown('<span class="badge-part">🟡 PARTIAL</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-unsup">🔴 UNSUPPORTED</span>', unsafe_allow_html=True)

                        st.markdown(f'<div class="evidence-quote"><b>Evidence Quote:</b> "{item.evidence}"</div>', unsafe_allow_html=True)
                        if item.verification_explanation:
                            st.caption(f"🔍 Audit Note: {item.verification_explanation}")
                        st.divider()
            else:
                st.write("_No action items identified._")

            if mins.next_meeting_details and mins.next_meeting_details != "Not specified":
                st.markdown("#### 5. Next Meeting Details")
                st.markdown(f"🗓️ **{mins.next_meeting_details}**")

            if mins.unresolved_questions:
                st.markdown("#### 6. Unresolved Questions")
                for q in mins.unresolved_questions:
                    st.markdown(f"❓ {q}")

        # Tab 2: Faithfulness Audit
        with res_tab2:
            st.markdown("### 🛡️ Hallucination & Evidence Audit")
            st.markdown("Every generated action item is audited against the original spoken transcript using Natural Language Inference (NLI) to ensure zero fabricated commitments or deadlines.")

            if mins.action_items:
                for idx, item in enumerate(mins.action_items, 1):
                    st.markdown(f"**Item {idx}:** {item.description}")
                    st.markdown(f"- **Owner Stated:** `{item.owner}`")
                    st.markdown(f"- **Deadline Stated:** `{item.deadline}`")
                    st.markdown(f"- **Transcript Evidence:** *\"{item.evidence}\"*")
                    st.markdown(f"- **Verification Status:** **{item.verification_status}**")
                    if item.verification_explanation:
                        st.markdown(f"- **Reasoning:** {item.verification_explanation}")
                    st.divider()
            else:
                st.write("No action items to audit.")

        # Tab 3: Transcript Viewer
        with res_tab3:
            st.markdown("### 📜 Meeting Transcript")
            active_tr = st.session_state.get("active_transcript_obj")
            if active_tr:
                st.text_area("Transcript text:", value=active_tr.formatted_transcript, height=350, disabled=True)

        # Tab 4: Export (PDF & TXT)
        with res_tab4:
            st.markdown("### 💾 Export Meeting Minutes")
            st.write("Download your structured meeting minutes in your preferred format:")

            # Generate export data
            try:
                pdf_bytes = generate_minutes_pdf(mins)
                txt_data = generate_minutes_txt(mins)
                md_data = generate_minutes_markdown(mins)
                json_data = json.dumps(mins.model_dump(), indent=2)

                exp1, exp2, exp3, exp4 = st.columns(4)
                with exp1:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with exp2:
                    st.download_button(
                        label="📥 Download TXT",
                        data=txt_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with exp3:
                    st.download_button(
                        label="📥 Download Markdown",
                        data=md_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with exp4:
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.json",
                        mime="application/json",
                        use_container_width=True
                    )
            except Exception as exp_err:
                st.error(f"Export error: {str(exp_err)}")


# ─── Right Column: Recent Meetings Widget ─────────────────────────────────────
with col_right:
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;align-items:center;gap:14px;margin-bottom:20px;">
        <div style="font-size:18px;cursor:pointer;color:#6B7280;">🔔</div>
        <div style="width:34px;height:34px;background:#111827;color:#FFFFFF;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;">P</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="mn-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        <span style="font-size:18px;">🕒</span>
        <span style="font-weight:700;font-size:1rem;color:#111827;">Recent Meetings</span>
    </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.get("recent_meetings", [])[:5]:
        st.markdown(f"""
        <div class="recent-item">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:16px;color:#9CA3AF;">📄</span>
                <div>
                    <div class="recent-title">{item['title']}</div>
                    <div class="recent-meta">{item['date']} · {item['duration']}</div>
                </div>
            </div>
            <div style="color:#9CA3AF;font-size:14px;">⋮</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    if st.button("View All Meetings →", key="btn_view_all_meetings", use_container_width=True):
        st.info("Showing past 5 meetings from session history.")
    st.markdown('</div>', unsafe_allow_html=True)
