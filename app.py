"""
MeetNote — AI Meeting Minutes Generator
Complete functional rewrite: reliable CSS, proper HTML rendering, working pipeline.
"""

import os
import json
import time
import streamlit as st
from dotenv import load_dotenv

from models import MeetingMinutes, Transcript, TranscriptSegment, ActionItem
from transcription import transcribe_audio
from minutes_generator import generate_meeting_minutes, get_gemini_client
from evaluator import verify_meeting_minutes_faithfulness

load_dotenv()

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetNote — AI Meeting Minutes",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Transparent Streamlit Wrappers (so animated background shows through) ── */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section.main,
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stMain"] {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #07070d !important;
}

.block-container, [data-testid="stMainBlockContainer"] {
    position: relative !important;
    z-index: 10 !important;
}

/* ── Animated Aurora Rays Component ── */
.animated-rays-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
    background-color: #07070d;
}

.rays-aurora {
    position: absolute;
    inset: -40%;
    width: 180%;
    height: 180%;
    background-image:
        repeating-linear-gradient(
            100deg,
            #07070d 0%,
            #07070d 7%,
            transparent 10%,
            transparent 12%,
            #07070d 16%
        ),
        repeating-linear-gradient(
            100deg,
            #3b82f6 10%,
            #e879f9 16%,
            #6366f1 22%,
            #06b6d4 28%,
            #a855f7 34%,
            #3b82f6 40%
        );
    background-size: 260% 180%;
    background-position: 50% 50%;
    filter: blur(22px) saturate(240%) brightness(1.35);
    opacity: 0.70;
    animation: aurora-flow 16s ease-in-out infinite alternate;
}

.rays-overlay {
    position: absolute;
    inset: -30%;
    width: 160%;
    height: 160%;
    background-image:
        repeating-linear-gradient(
            100deg,
            rgba(7, 7, 13, 0.85) 0%,
            rgba(7, 7, 13, 0.85) 7%,
            transparent 10%,
            transparent 12%,
            rgba(7, 7, 13, 0.85) 16%
        ),
        repeating-linear-gradient(
            100deg,
            #60a5fa 10%,
            #e879f9 15%,
            #60a5fa 20%,
            #5eead4 25%,
            #818cf8 30%
        );
    background-size: 200% 130%;
    background-attachment: fixed;
    mix-blend-mode: difference;
    opacity: 0.50;
    animation: aurora-flow-reverse 22s ease-in-out infinite alternate;
}

.rays-top-glow {
    position: absolute;
    top: -20vh;
    left: 50%;
    transform: translateX(-50%);
    width: 130vw;
    height: 75vh;
    background: radial-gradient(ellipse at 50% 15%, rgba(99, 102, 241, 0.55) 0%, rgba(232, 121, 249, 0.3) 35%, transparent 70%);
    filter: blur(40px);
    pointer-events: none;
    animation: glow-breathe 7s ease-in-out infinite alternate;
}

@keyframes aurora-flow {
    0% {
        background-position: 50% 50%, 50% 50%;
        transform: rotate(0deg) scale(1);
    }
    50% {
        background-position: 180% 50%, 180% 50%;
        transform: rotate(1.5deg) scale(1.05);
    }
    100% {
        background-position: 320% 50%, 320% 50%;
        transform: rotate(-1deg) scale(1.02);
    }
}

@keyframes aurora-flow-reverse {
    0% {
        background-position: 100% 50%, 100% 50%;
        transform: rotate(0deg);
    }
    100% {
        background-position: 280% 50%, 280% 50%;
        transform: rotate(-2deg);
    }
}

@keyframes glow-breathe {
    0% { opacity: 0.65; transform: translateX(-50%) scale(0.95); }
    100% { opacity: 1.0; transform: translateX(-50%) scale(1.08); }
}

[data-testid="stAppViewContainer"] > .main { color: #F9FAFB !important; }


[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #11111a !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    cursor: pointer !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    transition: background 0.15s, color 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: rgba(99,102,241,0.18) !important;
    color: #A5B4FC !important;
    font-weight: 600 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}
/* Hide radio circle dot */
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { margin: 0; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label > div:first-child { display: none !important; }


/* Main card — glass style on dark bg */
.mn-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    margin-bottom: 16px;
}

/* Generate button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 14px 0 !important;
    box-shadow: 0 2px 20px rgba(99,102,241,0.45) !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
    box-shadow: 0 4px 28px rgba(99,102,241,0.6) !important;
}

/* Metric cards — glass */
.metric-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 14px 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.card-title { font-size: 0.72rem; text-transform: uppercase; color: rgba(255,255,255,0.55); font-weight: 600; letter-spacing: 0.5px; }
.card-value { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin-top: 2px; }

/* Status badges */
.status-badge { display:inline-block; padding:3px 10px; border-radius:12px; font-weight:600; font-size:0.78rem; }
.badge-supported    { background:rgba(34,197,94,0.18); color:#86EFAC; border:1px solid rgba(34,197,94,0.3); }
.badge-partial      { background:rgba(251,191,36,0.18); color:#FDE68A; border:1px solid rgba(251,191,36,0.3); }
.badge-unsupported  { background:rgba(239,68,68,0.18); color:#FCA5A5; border:1px solid rgba(239,68,68,0.3); }

/* Evidence box */
.evidence-box {
    background: rgba(99,102,241,0.12); border-left: 3px solid #818CF8;
    padding: 8px 12px; font-size: 0.84rem; color: #C7D2FE;
    border-radius: 0 6px 6px 0; margin-top: 6px; font-style: italic;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(139,92,246,0.5) !important;
    border-radius: 12px !important;
}


/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #C7D2FE !important;
    border-bottom-color: #6366F1 !important;
}

/* Option labels */
.opt-label {
    font-size: 0.72rem; font-weight: 600; color: #A5B4FC;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}
</style>
<div class="animated-rays-bg">
    <div class="rays-aurora"></div>
    <div class="rays-overlay"></div>
    <div class="rays-top-glow"></div>
</div>
""", unsafe_allow_html=True)



# ─── Session state ────────────────────────────────────────────────────────────
for k, v in {
    "transcript": None,
    "meeting_minutes": None,
    "verification_results": None,
    "audio_bytes": None,
    "audio_filename": None,
    "transcript_file": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:20px 4px 16px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:12px;">
        <div style="width:36px;height:36px;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.35);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;">📋</div>
        <div>
            <div style="font-size:1.05rem;font-weight:700;color:#F8FAFC;line-height:1.2;">MeetNote</div>
            <div style="font-size:0.7rem;color:#818CF8;">AI Meeting Minutes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Navigation using radio (reliable across themes)
    page = st.radio(
        "Navigation",
        ["🏠  Home", "⚡  Generate", "🕐  History", "⊞  Templates", "⚙️  Settings"],
        label_visibility="collapsed",
        index=1
    )

    # ── API Key — read silently from .env (hidden from UI) ──
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    api_key_input = env_key  # not shown in sidebar


    # ── Model & Whisper (Settings tab) ──
    if "Settings" in page:
        st.markdown("**Gemini Model**")
        selected_model = st.selectbox(
            "model", ["gemini-3.8-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash-latest"],
            label_visibility="collapsed"
        )
        st.markdown("**Whisper Model**")
        whisper_model_size = st.selectbox(
            "whisper", ["tiny", "base", "small", "medium"],
            index=1, label_visibility="collapsed"
        )
        st.markdown("**Device**")
        whisper_device = st.selectbox(
            "device", ["cpu", "cuda"],
            label_visibility="collapsed"
        )
    else:
        selected_model   = "gemini-3.8-flash"
        whisper_model_size = "base"
        whisper_device   = "cpu"





# ─── Main layout ──────────────────────────────────────────────────────────────
main_col = st.container()

with main_col:

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:#818CF8;text-transform:uppercase;margin-bottom:4px;'>Welcome Back</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:2rem;font-weight:700;color:#FFFFFF;margin-bottom:6px;'>Generate Meeting Minutes</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='font-size:0.93rem;color:#94A3B8;margin-bottom:20px;'>Upload your meeting audio or provide a transcript to get structured, accurate meeting minutes.</div>",
        unsafe_allow_html=True
    )

    # ── Input tabs ────────────────────────────────────────────────────────────
    tab_audio, tab_text = st.tabs(["⬆️  Upload Audio", "📄  Direct Transcript"])

    with tab_audio:
        uploaded_file = st.file_uploader(
            "Drop your audio file here",
            type=["mp3", "wav", "m4a"],
            label_visibility="collapsed",
            help="Supported: MP3, WAV, M4A — up to 200 MB"
        )

        if uploaded_file is not None:
            st.session_state["audio_bytes"]    = uploaded_file.getvalue()
            st.session_state["audio_filename"] = uploaded_file.name
            size_mb = round(len(uploaded_file.getvalue()) / (1024 * 1024), 1)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:10px 14px;margin-top:10px;">
                <span style="font-size:22px;">🎵</span>
                <div>
                    <div style="font-size:0.87rem;font-weight:600;color:#F8FAFC;">{uploaded_file.name}</div>
                    <div style="font-size:0.75rem;color:#94A3B8;">{size_mb} MB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.audio(uploaded_file, format=uploaded_file.type)

        elif st.session_state.get("audio_bytes"):
            fname   = st.session_state.get("audio_filename", "audio.wav")
            size_mb = round(len(st.session_state["audio_bytes"]) / (1024 * 1024), 1)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:10px;padding:10px 14px;margin-top:10px;">
                <span style="font-size:22px;">🎵</span>
                <div>
                    <div style="font-size:0.87rem;font-weight:600;color:#F8FAFC;">{fname}</div>
                    <div style="font-size:0.75rem;color:#94A3B8;">{size_mb} MB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


        st.write("")

        # Options row
        o1, o2, o3 = st.columns([1, 1, 1.4])
        with o1:
            st.markdown('<div class="opt-label">Output Language</div>', unsafe_allow_html=True)
            output_lang = st.selectbox("lang_sel", ["English", "Hindi", "Spanish", "French"],
                                       label_visibility="collapsed")
        with o2:
            st.markdown('<div class="opt-label">Meeting Type</div>', unsafe_allow_html=True)
            meeting_type = st.selectbox("type_sel", ["General", "Standup", "Review", "Brainstorm", "Planning"],
                                        label_visibility="collapsed")
        with o3:
            st.markdown('<div class="opt-label">Include Features</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                inc_summary = st.checkbox("Summary",      value=True, key="chk_sum")
                inc_actions = st.checkbox("Action Items", value=True, key="chk_act")
            with cc2:
                inc_keypts  = st.checkbox("Key Points",  value=True, key="chk_kp")
                inc_next    = st.checkbox("Next Meeting", value=False, key="chk_nm")

    with tab_text:
        pasted_transcript = st.text_area(
            "Paste transcript:",
            height=200,
            label_visibility="collapsed",
            placeholder="[00:00 - 00:15] Rahul: Good morning everyone. Let's begin with the Q3 launch review...\n[00:15 - 00:30] Arjun: I'll handle the backend deployment by October 2nd."
        )
        if st.button("✅  Use This Transcript", key="use_transcript"):
            if pasted_transcript.strip():
                segs = [
                    TranscriptSegment(start=0.0, end=0.0, text=line.strip())
                    for line in pasted_transcript.split("\n") if line.strip()
                ]
                st.session_state["transcript"] = Transcript(
                    segments=segs,
                    full_text=pasted_transcript,
                    duration_seconds=0.0
                )
                st.success("✅ Transcript ready — click **Generate Meeting Minutes** below.")
            else:
                st.warning("Please paste some transcript text first.")

    # ── Generate button ───────────────────────────────────────────────────────
    st.write("")
    start_generation = st.button("📋  Generate Meeting Minutes",
                                  type="primary", use_container_width=True,
                                  key="btn_generate")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    if start_generation:
        active_key = (api_key_input or "").strip()

        # ── Validation ──
        if not active_key or active_key == "your_gemini_api_key_here":
            st.error("⚠️ **Gemini API Key not found.** Please set `GEMINI_API_KEY=your_key` in your `.env` file and restart the app.")
            st.stop()

        has_audio      = bool(st.session_state.get("audio_bytes"))
        has_transcript = bool(st.session_state.get("transcript"))

        if not has_audio and not has_transcript:
            st.warning("⚠️ Please upload an audio file **or** paste a transcript first.")
            st.stop()

        # ── Run pipeline ──
        progress_bar = st.progress(0.0)
        status_box   = st.empty()

        try:
            # Step 1: Transcription (only if audio provided without transcript)
            if has_audio and (
                not has_transcript or
                (uploaded_file and st.session_state.get("transcript_file") != uploaded_file.name)
            ):
                status_box.info("🎙️ **Step 1/3** — Transcribing audio with faster-whisper…")

                def _tp(msg: str, frac: float):
                    progress_bar.progress(min(0.38, frac * 0.38))
                    status_box.info(f"🎙️ Transcription: {msg}")

                t = transcribe_audio(
                    audio_source=st.session_state["audio_bytes"],
                    model_size=whisper_model_size,
                    device=whisper_device,
                    progress_callback=_tp
                )
                st.session_state["transcript"] = t
                if uploaded_file:
                    st.session_state["transcript_file"] = uploaded_file.name
            else:
                progress_bar.progress(0.38)

            # Step 2: Generate minutes
            status_box.info(f"🧠 **Step 2/3** — Generating minutes with Gemini ({selected_model})…")
            progress_bar.progress(0.45)

            def _mp(msg: str, frac: float):
                progress_bar.progress(0.45 + frac * 0.35)
                status_box.info(f"🧠 Gemini: {msg}")

            client  = get_gemini_client(api_key=active_key)
            minutes = generate_meeting_minutes(
                transcript=st.session_state["transcript"],
                api_key=active_key,
                model=selected_model,
                progress_callback=_mp
            )
            st.session_state["meeting_minutes"] = minutes

            # Step 3: Faithfulness verification
            status_box.info("🛡️ **Step 3/3** — Verifying faithfulness & evidence…")
            progress_bar.progress(0.85)

            vr = verify_meeting_minutes_faithfulness(
                minutes=minutes,
                transcript=st.session_state["transcript"],
                client=client,
                model=selected_model
            )
            st.session_state["verification_results"] = vr

            progress_bar.progress(1.0)
            status_box.success("✅ Meeting Minutes generated and verified!")
            time.sleep(1.2)
            status_box.empty()
            progress_bar.empty()
            st.rerun()

        except Exception as e:
            st.error(f"❌ **Pipeline Error:** {str(e)}")
            progress_bar.empty()
            status_box.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    minutes_obj: MeetingMinutes = st.session_state.get("meeting_minutes")
    transcript_obj: Transcript  = st.session_state.get("transcript")
    verifications               = st.session_state.get("verification_results")

    if minutes_obj:
        st.divider()

        # Metric row
        total_ai    = len(minutes_obj.action_items)
        supported   = sum(1 for a in minutes_obj.action_items if a.verification_status == "SUPPORTED")
        partial     = sum(1 for a in minutes_obj.action_items if a.verification_status == "PARTIALLY_SUPPORTED")
        unsupported = sum(1 for a in minutes_obj.action_items if a.verification_status == "UNSUPPORTED")
        faith_score = int((supported / max(1, total_ai)) * 100)
        score_col   = "#166534" if faith_score >= 90 else ("#92400E" if faith_score >= 70 else "#991B1B")

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f'<div class="metric-card"><div class="card-title">Key Decisions</div><div class="card-value">{len(minutes_obj.key_decisions)}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="metric-card"><div class="card-title">Action Items</div><div class="card-value">{total_ai}</div></div>', unsafe_allow_html=True)
        with mc3:
            st.markdown(f'<div class="metric-card"><div class="card-title">Open Questions</div><div class="card-value">{len(minutes_obj.unresolved_questions)}</div></div>', unsafe_allow_html=True)
        with mc4:
            st.markdown(f'<div class="metric-card"><div class="card-title">Faithfulness</div><div class="card-value" style="color:{score_col}">{faith_score}%</div></div>', unsafe_allow_html=True)

        st.write("")
        if unsupported > 0:
            st.error(f"🚨 **{unsupported} action item(s) flagged UNSUPPORTED** — review the Faithfulness tab.")
        elif partial > 0:
            st.warning(f"⚠️ {partial} item(s) PARTIALLY SUPPORTED — owner/deadline loosely implied.")

        # Result tabs
        r1, r2, r3, r4 = st.tabs(["📋 Minutes", "🛡️ Faithfulness", "📜 Transcript", "💾 Export"])

        with r1:
            st.subheader(f"📌 {minutes_obj.meeting_title}")
            if minutes_obj.meeting_date and minutes_obj.meeting_date != "Not specified":
                st.caption(f"📅 {minutes_obj.meeting_date}")

            st.markdown("#### 1. Executive Summary")
            st.info(minutes_obj.meeting_summary)

            st.markdown("#### 2. Key Decisions")
            if minutes_obj.key_decisions:
                for d in minutes_obj.key_decisions:
                    st.markdown(f"✅ **{d}**")
            else:
                st.write("_No explicit decisions recorded._")

            st.markdown("#### 3. Action Items")
            if minutes_obj.action_items:
                for i, item in enumerate(minutes_obj.action_items, 1):
                    with st.container():
                        st.markdown(f"**{i}. {item.description}**")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"👤 **Owner:** {item.owner}")
                        with c2:
                            st.markdown(f"📅 **Deadline:** {item.deadline}")
                        with c3:
                            bmap = {
                                "SUPPORTED":          '<span class="status-badge badge-supported">🟢 SUPPORTED</span>',
                                "PARTIALLY_SUPPORTED":'<span class="status-badge badge-partial">🟡 PARTIAL</span>',
                                "UNSUPPORTED":        '<span class="status-badge badge-unsupported">🔴 UNSUPPORTED</span>',
                            }
                            st.markdown(bmap.get(item.verification_status, ""), unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="evidence-box"><strong>Evidence:</strong> "{item.evidence}"</div>',
                            unsafe_allow_html=True
                        )
                        if item.verification_explanation:
                            st.caption(f"🔍 {item.verification_explanation}")
                        st.divider()
            else:
                st.write("_No actionable items identified._")

            st.markdown("#### 4. Unresolved Questions")
            if minutes_obj.unresolved_questions:
                for q in minutes_obj.unresolved_questions:
                    st.markdown(f"❓ {q}")
            else:
                st.write("_No open questions._")

        with r2:
            st.subheader("🛡️ Faithfulness Audit")
            st.caption("Every action item is verified against the spoken transcript for grounding.")
            if verifications:
                for res in verifications:
                    with st.expander(
                        f"#{res.action_item_index + 1}: {res.action_item_description} [{res.status}]",
                        expanded=(res.status != "SUPPORTED")
                    ):
                        vc1, vc2, vc3 = st.columns(3)
                        with vc1:
                            st.write(f"**Evidence Quote:** {'✅ Found' if res.quote_found_in_transcript else '❌ Missing'}")
                        with vc2:
                            st.write(f"**Owner:** {'✅ Grounded' if res.owner_faithfully_supported else '❌ Not grounded'}")
                        with vc3:
                            st.write(f"**Deadline:** {'✅ Grounded' if res.deadline_faithfully_supported else '❌ Not grounded'}")
                        st.markdown(f"**Reasoning:** {res.explanation}")
            else:
                st.info("No verifications available.")

        with r3:
            st.subheader("📜 Transcript")
            if transcript_obj:
                st.caption(f"Language: `{transcript_obj.language}` | Duration: `{transcript_obj.duration_seconds}s` | Segments: `{len(transcript_obj.segments)}`")
                sq = st.text_input("🔍 Search segments:", placeholder="Filter…")
                segs = transcript_obj.segments
                if sq.strip():
                    segs = [s for s in segs if sq.lower() in s.text.lower()]
                body = "\n\n".join(s.format_line() for s in segs) if segs else transcript_obj.full_text
                st.text_area("Segments", body, height=400, label_visibility="collapsed")
            else:
                st.info("No transcript loaded.")

        with r4:
            st.subheader("💾 Export")
            # TXT
            txt = [f"MEETING MINUTES: {minutes_obj.meeting_title}",
                   f"Date: {minutes_obj.meeting_date}", "="*60,
                   "\n1. EXECUTIVE SUMMARY", minutes_obj.meeting_summary,
                   "\n2. KEY DECISIONS"]
            for d in minutes_obj.key_decisions: txt.append(f"- {d}")
            txt.append("\n3. ACTION ITEMS")
            for i, a in enumerate(minutes_obj.action_items, 1):
                txt.append(f"{i}. {a.description}\n   Owner: {a.owner}\n   Deadline: {a.deadline}\n   Evidence: {a.evidence}\n   Status: {a.verification_status}")
            txt.append("\n4. UNRESOLVED QUESTIONS")
            for q in minutes_obj.unresolved_questions: txt.append(f"- {q}")
            # MD
            md = [f"# {minutes_obj.meeting_title}", f"**Date:** {minutes_obj.meeting_date}\n",
                  "## Executive Summary", minutes_obj.meeting_summary, "\n## Key Decisions"]
            for d in minutes_obj.key_decisions: md.append(f"- {d}")
            md.append("\n## Action Items")
            for i, a in enumerate(minutes_obj.action_items, 1):
                md.append(f"- **{a.description}**\n  - Owner: {a.owner}\n  - Deadline: {a.deadline}\n  - Status: `{a.verification_status}`\n  - Evidence: > {a.evidence}")
            md.append("\n## Unresolved Questions")
            for q in minutes_obj.unresolved_questions: md.append(f"- {q}")

            ts = int(time.time())
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button("📄 TXT", "\n".join(txt), f"minutes_{ts}.txt", "text/plain", use_container_width=True)
            with dl2:
                st.download_button("📦 JSON", json.dumps(minutes_obj.model_dump(), indent=2), f"minutes_{ts}.json", "application/json", use_container_width=True)
            with dl3:
                st.download_button("📝 Markdown", "\n".join(md), f"minutes_{ts}.md", "text/markdown", use_container_width=True)


