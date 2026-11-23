"""
MeetNote — Executive Meeting Intelligence & Structured Minutes Generator
Supports two independent workflows:
1. Upload Existing Audio (MP3, WAV, M4A -> faster-whisper STT -> Editable Transcript)
2. Live Meeting Transcription (Browser Mic -> Real-time Streaming Web Speech STT + Waveform -> Editable Transcript)
Both workflows produce structured minutes (Summary, Discussion Points, Decisions, Action Items, Next Meeting Details)
and support persistent Recent Meetings history and clean exports (PDF, TXT, Markdown, JSON).
"""

import os
import io
import json
import time
import uuid
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
    page_title="MeetNote — Meeting Minutes Generator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Live Transcription Component ─────────────────────────────────────
_component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "live_transcription")
live_transcription_comp = components.declare_component("live_transcription", path=_component_dir)

# ─── Session State ───────────────────────────────────────────────────────────
if "active_mode" not in st.session_state:
    st.session_state["active_mode"] = "upload"

if "uploaded_audio_bytes" not in st.session_state:
    st.session_state["uploaded_audio_bytes"] = None
if "uploaded_audio_name" not in st.session_state:
    st.session_state["uploaded_audio_name"] = None
if "upload_transcript_text" not in st.session_state:
    st.session_state["upload_transcript_text"] = ""

if "live_transcript_text" not in st.session_state:
    st.session_state["live_transcript_text"] = ""
if "live_duration_seconds" not in st.session_state:
    st.session_state["live_duration_seconds"] = 0

if "meeting_minutes" not in st.session_state:
    st.session_state["meeting_minutes"] = None
if "active_transcript_obj" not in st.session_state:
    st.session_state["active_transcript_obj"] = None
if "verification_results" not in st.session_state:
    st.session_state["verification_results"] = None

# Real persistent storage for meetings within the session
if "saved_meetings" not in st.session_state:
    st.session_state["saved_meetings"] = []
if "selected_meeting_id" not in st.session_state:
    st.session_state["selected_meeting_id"] = None


# ─── Refined Professional CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* Global resets */
body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
}

[data-testid="stHeader"] {
    display: none;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0;
    padding-top: 1.2rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    padding: 9px 14px !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #475569 !important;
    cursor: pointer !important;
    transition: background 0.15s;
}

[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Brand header */
.app-brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #E2E8F0;
}

.app-brand-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: #0F172A;
    line-height: 1.1;
}

.app-brand-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    margin-top: 4px;
}

.workspace-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    font-size: 0.76rem;
    font-weight: 600;
    color: #4338CA;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Standard White Cards */
.content-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    margin-bottom: 18px;
}

/* Mode Selection Cards */
.mode-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 18px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.mode-card:hover {
    border-color: #94A3B8;
}

.mode-card.active {
    border-color: #0F172A;
    background: #FFFFFF;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
}

.mode-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
}

.mode-desc {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 2px;
}

/* Primary generate button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #0F172A !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.2) !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1E293B !important;
}

/* Stat Cards */
.metric-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.metric-lbl {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748B;
    letter-spacing: 0.5px;
}

.metric-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0F172A;
    margin-top: 2px;
}

/* Recent meeting item */
.mtg-card-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 10px;
    transition: border-color 0.15s;
}

.mtg-card-item:hover {
    border-color: #94A3B8;
}

.mtg-card-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #0F172A;
}

.mtg-card-meta {
    font-size: 0.74rem;
    color: #64748B;
    margin-top: 2px;
}

/* Evidence quote callout */
.evidence-callout {
    background: #F8FAFC;
    border-left: 3px solid #4F46E5;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    font-size: 0.84rem;
    color: #334155;
    font-style: italic;
    margin-top: 6px;
}

/* Status badges without emojis */
.status-pill {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}

.status-pill.supported {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #BBF7D0;
}

.status-pill.partial {
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}

.status-pill.unsupported {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:14px;border-bottom:1px solid #E2E8F0;margin-bottom:14px;">
        <div style="width:34px;height:34px;border-radius:8px;background:#0F172A;display:flex;align-items:center;justify-content:center;color:#FFFFFF;font-weight:700;font-size:0.95rem;">
            MN
        </div>
        <div>
            <div style="font-size:1.25rem;font-weight:800;letter-spacing:-0.4px;color:#0F172A;line-height:1.1;">MeetNote</div>
            <div style="font-size:0.75rem;color:#64748B;">Meeting Minutes Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_tab = st.radio(
        "Navigation",
        ["Home", "Meeting History", "Settings"],
        label_visibility="collapsed",
        index=0
    )

    st.write("")
    if "Settings" in nav_tab:
        st.markdown("**Engine Settings**")
        selected_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest"],
            index=0
        )
        whisper_size = st.selectbox(
            "Whisper STT Size",
            ["tiny", "base", "small", "medium"],
            index=1
        )
    else:
        selected_model = "gemini-3.5-flash"
        whisper_size = "base"


# ─── Top Brand Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="app-brand-row">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="width:46px;height:46px;border-radius:10px;background:#0F172A;display:flex;align-items:center;justify-content:center;color:#FFFFFF;font-weight:800;font-size:1.25rem;letter-spacing:-0.5px;box-shadow:0 2px 6px rgba(15,23,42,0.18);">
            MN
        </div>
        <div>
            <div class="app-brand-title">MeetNote</div>
            <div class="app-brand-subtitle">Transcribe audio recordings and live meetings into verified, structured meeting minutes.</div>
        </div>
    </div>
    <div style="text-align:right;">
        <span class="workspace-badge">Workspace</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Main Content Layout: Two Columns ────────────────────────────────────────
col_main, col_recent = st.columns([2.5, 1.1], gap="large")

with col_main:

    # ── Action Bar (if viewing a generated/selected meeting) ──
    if st.session_state.get("meeting_minutes"):
        top_bar_col1, top_bar_col2 = st.columns([3, 1])
        with top_bar_col1:
            curr_title = st.session_state["meeting_minutes"].meeting_title
            st.markdown(f"**Viewing:** `{curr_title}`")
        with top_bar_col2:
            if st.button("New Meeting", key="btn_new_meeting", use_container_width=True):
                st.session_state["meeting_minutes"] = None
                st.session_state["active_transcript_obj"] = None
                st.session_state["verification_results"] = None
                st.session_state["upload_transcript_text"] = ""
                st.session_state["live_transcript_text"] = ""
                st.session_state["selected_meeting_id"] = None
                st.rerun()

    # ── Mode Selection Cards (Upload vs Live) ─────────────────────────────────
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        is_upload = st.session_state["active_mode"] == "upload"
        cls_upload = "active" if is_upload else ""
        st.markdown(f"""
        <div class="mode-card {cls_upload}">
            <div class="mode-title">Upload Recording</div>
            <div class="mode-desc">Process pre-recorded audio (MP3, WAV, M4A)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Upload Workflow", key="btn_choose_upload", use_container_width=True):
            st.session_state["active_mode"] = "upload"
            st.rerun()

    with m_col2:
        is_live = st.session_state["active_mode"] == "live"
        cls_live = "active" if is_live else ""
        st.markdown(f"""
        <div class="mode-card {cls_live}">
            <div class="mode-title">Live Meeting</div>
            <div class="mode-desc">Capture microphone audio in real time</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Live Workflow", key="btn_choose_live", use_container_width=True):
            st.session_state["active_mode"] = "live"
            st.rerun()

    st.write("")

    # =========================================================================
    # WORKFLOW 1: UPLOAD EXISTING AUDIO (Independent)
    # =========================================================================
    if st.session_state["active_mode"] == "upload":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### Upload Audio Recording")
        st.caption("Upload previously recorded meeting audio. No microphone access is requested.")

        audio_file = st.file_uploader(
            "Upload meeting recording",
            type=["mp3", "wav", "m4a"],
            label_visibility="collapsed",
            help="Supported: MP3, WAV, M4A up to 200MB"
        )

        if audio_file is not None:
            st.session_state["uploaded_audio_bytes"] = audio_file.getvalue()
            st.session_state["uploaded_audio_name"] = audio_file.name

        # Audio file info & player
        if st.session_state["uploaded_audio_bytes"]:
            fname = st.session_state["uploaded_audio_name"]
            size_mb = round(len(st.session_state["uploaded_audio_bytes"]) / (1024 * 1024), 2)

            f_col1, f_col2 = st.columns([3, 1])
            with f_col1:
                st.markdown(f"""
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px 12px;margin:8px 0;">
                    <div style="font-weight:600;font-size:0.86rem;color:#0F172A;">{fname}</div>
                    <div style="font-size:0.75rem;color:#64748B;">Size: {size_mb} MB</div>
                </div>
                """, unsafe_allow_html=True)
            with f_col2:
                if st.button("Remove File", key="btn_remove_upload_file"):
                    st.session_state["uploaded_audio_bytes"] = None
                    st.session_state["uploaded_audio_name"] = None
                    st.session_state["upload_transcript_text"] = ""
                    st.rerun()

            st.audio(st.session_state["uploaded_audio_bytes"])

            st.write("")
            if st.button("Transcribe Audio Recording", key="btn_exec_transcribe", type="secondary", use_container_width=True):
                with st.spinner("Transcribing audio using local faster-whisper STT..."):
                    try:
                        transcript_obj = transcribe_audio(
                            st.session_state["uploaded_audio_bytes"],
                            model_size=whisper_size,
                            device="cpu"
                        )
                        st.session_state["upload_transcript_text"] = transcript_obj.formatted_transcript
                        st.session_state["active_transcript_obj"] = transcript_obj
                        st.success(f"Transcription complete: {round(transcript_obj.duration_seconds, 1)}s audio processed.")
                    except Exception as err:
                        st.error(f"Transcription error: {str(err)}")

        st.write("")
        st.markdown("**Transcript (Review & Edit):**")
        edited_upload_text = st.text_area(
            "Upload transcript review:",
            value=st.session_state.get("upload_transcript_text", ""),
            height=160,
            label_visibility="collapsed",
            placeholder="Transcribed text from your uploaded audio will appear here. You can review and edit before generating minutes."
        )
        st.session_state["upload_transcript_text"] = edited_upload_text
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================================
    # WORKFLOW 2: LIVE MEETING TRANSCRIPTION (Independent)
    # =========================================================================
    elif st.session_state["active_mode"] == "live":
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### Live Meeting Transcription")
        st.caption("Captures microphone speech in real time with live audio activity and incremental text streaming.")

        # Embed custom live transcription component
        live_data = live_transcription_comp()

        if live_data and isinstance(live_data, dict):
            if live_data.get("transcript"):
                st.session_state["live_transcript_text"] = live_data["transcript"]
            if "duration_seconds" in live_data:
                st.session_state["live_duration_seconds"] = live_data["duration_seconds"]

        st.write("")
        st.markdown("**Live Transcript (Review & Edit):**")
        edited_live_text = st.text_area(
            "Live transcript review:",
            value=st.session_state.get("live_transcript_text", ""),
            height=160,
            label_visibility="collapsed",
            placeholder="Spoken words will stream incrementally into the panel above. After stopping the meeting, the full transcript will be preserved here for review and editing."
        )
        st.session_state["live_transcript_text"] = edited_live_text
        st.markdown('</div>', unsafe_allow_html=True)


    # =========================================================================
    # SHARED CONFIGURATION & GENERATE BUTTON
    # =========================================================================
    st.markdown('<div class="content-card" style="padding:16px 20px;">', unsafe_allow_html=True)
    cfg1, cfg2, cfg3 = st.columns([1, 1, 1.4])
    with cfg1:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;'>Output Language</div>", unsafe_allow_html=True)
        out_language = st.selectbox(
            "lang_select",
            ["English", "Hindi", "Spanish", "French", "German"],
            label_visibility="collapsed"
        )
    with cfg2:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;'>Meeting Type</div>", unsafe_allow_html=True)
        m_type = st.selectbox(
            "type_select",
            ["General", "Standup", "Review", "Brainstorm", "Planning"],
            label_visibility="collapsed"
        )
    with cfg3:
        st.markdown("<div style='font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;'>Include Sections</div>", unsafe_allow_html=True)
        inc_c1, inc_c2 = st.columns(2)
        with inc_c1:
            inc_summary = st.checkbox("Summary", value=True, key="cfg_inc_sum")
            inc_actions = st.checkbox("Action Items", value=True, key="cfg_inc_act")
        with inc_c2:
            inc_keypts = st.checkbox("Key Points", value=True, key="cfg_inc_kp")
            inc_next = st.checkbox("Next Meeting", value=True, key="cfg_inc_next")

    st.write("")
    btn_generate = st.button("Generate Meeting Minutes", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Generate Execution ──
    if btn_generate:
        # Determine source transcript
        if st.session_state["active_mode"] == "upload":
            chosen_text = st.session_state.get("upload_transcript_text", "").strip()
            source_tag = "Uploaded Audio"
        else:
            chosen_text = st.session_state.get("live_transcript_text", "").strip()
            source_tag = "Live Meeting"

        if not chosen_text:
            if st.session_state["active_mode"] == "upload":
                st.error("No transcript available. Upload an audio recording and click Transcribe Audio Recording, or paste text into the transcript area.")
            else:
                st.error("No live transcript recorded. Click Start Live Meeting and speak into your microphone, or paste text into the transcript area.")
            st.stop()

        # Build Transcript model
        lines = [l.strip() for l in chosen_text.split("\n") if l.strip()]
        segments = [TranscriptSegment(start=0.0, end=0.0, text=l) for l in lines]
        active_transcript = Transcript(
            segments=segments,
            full_text=chosen_text,
            language="en",
            duration_seconds=st.session_state.get("live_duration_seconds", 0.0)
        )
        st.session_state["active_transcript_obj"] = active_transcript

        with st.status("Processing Meeting Minutes...", expanded=True) as status_box:
            progress_bar = st.progress(0.1)
            status_text = st.empty()

            def update_progress(msg: str, frac: float):
                status_text.markdown(f"**{msg}**")
                progress_bar.progress(min(1.0, max(0.0, frac)))

            try:
                # 1. Structured Minutes Generation
                minutes = generate_meeting_minutes(
                    transcript=active_transcript,
                    model=selected_model,
                    language=out_language,
                    meeting_type=m_type,
                    include_features={
                        "Summary": inc_summary,
                        "Action Items": inc_actions,
                        "Key Points": inc_keypts,
                        "Next Meeting": inc_next
                    },
                    progress_callback=update_progress
                )
                st.session_state["meeting_minutes"] = minutes

                # 2. Faithfulness Verification
                update_progress("Auditing action items for faithfulness against transcript...", 0.85)
                client = get_gemini_client()
                audit_report = verify_meeting_minutes_faithfulness(
                    minutes=minutes,
                    transcript=active_transcript,
                    client=client,
                    model=selected_model
                )
                st.session_state["verification_results"] = audit_report

                # 3. Save to Real Meeting History
                new_entry = {
                    "id": f"mtg_{uuid.uuid4().hex[:8]}",
                    "title": minutes.meeting_title,
                    "date": time.strftime("%b %d, %Y - %I:%M %p"),
                    "duration": f"{len(lines)} lines",
                    "minutes": minutes,
                    "transcript": active_transcript,
                    "verification": audit_report,
                    "source": source_tag
                }
                st.session_state["saved_meetings"].insert(0, new_entry)
                st.session_state["selected_meeting_id"] = new_entry["id"]

                update_progress("Completed successfully.", 1.0)
                status_box.update(label="Meeting Minutes Generated & Audited", state="complete", expanded=False)
                st.rerun()

            except Exception as e:
                status_box.update(label="Pipeline Execution Error", state="error", expanded=True)
                st.error(f"Error generating minutes: {str(e)}")


    # =========================================================================
    # SHARED RESULTS OUTPUT SYSTEM
    # =========================================================================
    if st.session_state.get("meeting_minutes"):
        mins: MeetingMinutes = st.session_state["meeting_minutes"]
        audit = st.session_state.get("verification_results", {})

        st.write("")
        st.markdown("### Meeting Intelligence Output")

        # Top stat metrics
        total_actions = len(mins.action_items)
        supported_count = sum(1 for a in mins.action_items if a.verification_status == "SUPPORTED")
        faith_score = round((supported_count / max(1, total_actions)) * 100) if total_actions else 100

        stat1, stat2, stat3, stat4 = st.columns(4)
        with stat1:
            st.markdown(f'<div class="metric-box"><div class="metric-lbl">Key Decisions</div><div class="metric-val">{len(mins.key_decisions)}</div></div>', unsafe_allow_html=True)
        with stat2:
            st.markdown(f'<div class="metric-box"><div class="metric-lbl">Action Items</div><div class="metric-val">{total_actions}</div></div>', unsafe_allow_html=True)
        with stat3:
            st.markdown(f'<div class="metric-box"><div class="metric-lbl">Discussion Points</div><div class="metric-val">{len(mins.key_discussion_points)}</div></div>', unsafe_allow_html=True)
        with stat4:
            st.markdown(f'<div class="metric-box"><div class="metric-lbl">Faithfulness Score</div><div class="metric-val" style="color:#10B981;">{faith_score}%</div></div>', unsafe_allow_html=True)

        st.write("")

        # 4 Output Tabs without emojis
        t_min, t_audit, t_trans, t_exp = st.tabs(["Minutes", "Faithfulness Audit", "Transcript", "Export"])

        # Tab 1: Minutes
        with t_min:
            st.markdown(f"### {mins.meeting_title}")
            if mins.meeting_date and mins.meeting_date != "Not specified":
                st.caption(f"Date: {mins.meeting_date}")

            st.markdown("#### 1. Executive Summary")
            st.info(mins.meeting_summary)

            if mins.key_discussion_points:
                st.markdown("#### 2. Key Discussion Points")
                for pt in mins.key_discussion_points:
                    st.markdown(f"• {pt}")

            st.markdown("#### 3. Decisions Made")
            if mins.key_decisions:
                for d in mins.key_decisions:
                    st.markdown(f"• [Agreed] **{d}**")
            else:
                st.write("_No explicit decisions recorded._")

            st.markdown("#### 4. Action Items & Responsible Persons")
            if mins.action_items:
                for idx, item in enumerate(mins.action_items, 1):
                    with st.container():
                        st.markdown(f"**{idx}. {item.description}**")
                        c_a, c_b, c_c = st.columns(3)
                        with c_a:
                            st.markdown(f"Responsible Person: `{item.owner}`")
                        with c_b:
                            st.markdown(f"Deadline: `{item.deadline}`")
                        with c_c:
                            if item.verification_status == "SUPPORTED":
                                st.markdown('<span class="status-pill supported">SUPPORTED</span>', unsafe_allow_html=True)
                            elif item.verification_status == "PARTIALLY_SUPPORTED":
                                st.markdown('<span class="status-pill partial">PARTIAL</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="status-pill unsupported">UNSUPPORTED</span>', unsafe_allow_html=True)

                        st.markdown(f'<div class="evidence-callout">Transcript Evidence: "{item.evidence}"</div>', unsafe_allow_html=True)
                        if item.verification_explanation:
                            st.caption(f"Audit Note: {item.verification_explanation}")
                        st.divider()
            else:
                st.write("_No action items identified._")

            if mins.next_meeting_details and mins.next_meeting_details != "Not specified":
                st.markdown("#### 5. Next Meeting Details")
                st.markdown(f"Next Meeting: **{mins.next_meeting_details}**")

            if mins.unresolved_questions:
                st.markdown("#### 6. Unresolved Questions")
                for q in mins.unresolved_questions:
                    st.markdown(f"• Open Question: {q}")

        # Tab 2: Faithfulness Audit
        with t_audit:
            st.markdown("#### Faithfulness & Evidence Audit")
            st.caption("Each extracted action item is audited against the source transcript using Natural Language Inference (NLI) to prevent fabricated commitments or deadlines.")

            if mins.action_items:
                for idx, item in enumerate(mins.action_items, 1):
                    st.markdown(f"**Item {idx}:** {item.description}")
                    st.markdown(f"- Owner: `{item.owner}`")
                    st.markdown(f"- Deadline: `{item.deadline}`")
                    st.markdown(f"- Evidence: *\"{item.evidence}\"*")
                    st.markdown(f"- Status: **{item.verification_status}**")
                    if item.verification_explanation:
                        st.markdown(f"- Audit Note: {item.verification_explanation}")
                    st.divider()
            else:
                st.write("No action items to audit.")

        # Tab 3: Transcript Viewer
        with t_trans:
            st.markdown("#### Meeting Transcript")
            active_tr = st.session_state.get("active_transcript_obj")
            if active_tr:
                st.text_area("Full transcript text:", value=active_tr.formatted_transcript, height=350, disabled=True)

        # Tab 4: Export (PDF & TXT)
        with t_exp:
            st.markdown("#### Export Meeting Minutes")
            st.caption("Download your structured meeting minutes in PDF, plain text, Markdown, or JSON formats.")

            try:
                pdf_bytes = generate_minutes_pdf(mins)
                txt_data = generate_minutes_txt(mins)
                md_data = generate_minutes_markdown(mins)
                json_data = json.dumps(mins.model_dump(), indent=2)

                x1, x2, x3, x4 = st.columns(4)
                with x1:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with x2:
                    st.download_button(
                        label="Download TXT",
                        data=txt_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with x3:
                    st.download_button(
                        label="Download Markdown",
                        data=md_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with x4:
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{mins.meeting_title.replace(' ', '_').lower()}_minutes.json",
                        mime="application/json",
                        use_container_width=True
                    )
            except Exception as exp_err:
                st.error(f"Export error: {str(exp_err)}")


# ─── Right Column: REAL Persistent Recent Meetings ────────────────────────────
with col_recent:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### Recent Meetings")
    st.caption("Recorded meetings from this session appear here. Click any meeting to open its full details.")

    saved_list = st.session_state.get("saved_meetings", [])

    if not saved_list:
        st.markdown("""
        <div style="border:1.5px dashed #E2E8F0;border-radius:8px;padding:20px;text-align:center;color:#64748B;font-size:0.84rem;margin:12px 0;">
            No recorded meetings yet.<br/>Upload audio or start a live meeting to save your first minutes.
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, mtg in enumerate(saved_list):
            is_active_mtg = st.session_state.get("selected_meeting_id") == mtg["id"]
            border_color = "#0F172A" if is_active_mtg else "#E2E8F0"
            bg_color = "#F8FAFC" if is_active_mtg else "#FFFFFF"

            st.markdown(f"""
            <div style="border:1px solid {border_color};background:{bg_color};border-radius:8px;padding:10px 12px;margin-bottom:8px;">
                <div style="font-weight:700;font-size:0.88rem;color:#0F172A;">{mtg['title']}</div>
                <div style="font-size:0.75rem;color:#64748B;margin-top:2px;">{mtg['date']} • {mtg['source']}</div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                if st.button("Open Details", key=f"btn_open_{mtg['id']}", use_container_width=True):
                    st.session_state["meeting_minutes"] = mtg["minutes"]
                    st.session_state["active_transcript_obj"] = mtg["transcript"]
                    st.session_state["verification_results"] = mtg["verification"]
                    st.session_state["selected_meeting_id"] = mtg["id"]
                    st.rerun()
            with btn_col2:
                if st.button("Delete", key=f"btn_del_{mtg['id']}", use_container_width=True):
                    st.session_state["saved_meetings"] = [m for m in st.session_state["saved_meetings"] if m["id"] != mtg["id"]]
                    if st.session_state.get("selected_meeting_id") == mtg["id"]:
                        st.session_state["meeting_minutes"] = None
                        st.session_state["selected_meeting_id"] = None
                    st.rerun()

        st.write("")
        if st.button("Clear History", key="btn_clear_all_meetings", use_container_width=True):
            st.session_state["saved_meetings"] = []
            st.session_state["meeting_minutes"] = None
            st.session_state["selected_meeting_id"] = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
