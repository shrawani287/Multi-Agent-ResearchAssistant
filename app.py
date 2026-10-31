import contextlib
import inspect
import io
import math
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents import llm
from pipeline import run_research_pipeline

st.set_page_config(page_title="ResearchPilot AI – Multi-Agent Research Assistant", page_icon="🔭", layout="wide")


# ---------------------------------------------------------------------------
# Theme - injected once, up top
# ---------------------------------------------------------------------------
def _inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #0A0E17;
            --panel: #121828;
            --panel-2: #182136;
            --border: #26314A;
            --text: #E9ECF4;
            --text-dim: #8B93AD;
            --amber: #F2A93B;
            --teal: #34D6C4;
            --red: #F0554D;
        }

        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], body {
            background-color: var(--bg) !important;
            color: var(--text);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stHeader"] {
            visibility: hidden; height: 0;
        }
        .block-container { padding-top: 2.2rem; max-width: 1100px; }

        h1, h2, h3, h4,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            font-family: 'Space Grotesk', sans-serif !important; color: var(--text) !important;
        }
        p, li, span, label { color: var(--text); }
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        a { color: var(--teal); }
        code, pre, [data-testid="stCodeBlock"] * { font-family: 'JetBrains Mono', monospace !important; }
        blockquote { border-left: 3px solid var(--amber); color: var(--text-dim); padding-left: 1rem; }
        hr { border-color: var(--border); }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            font-family: 'Space Grotesk', sans-serif; text-transform: uppercase;
            font-size: 0.9rem; letter-spacing: 0.1em; color: var(--text-dim);
        }

        /* Text input */
        .stTextInput>div>div>input {
            background-color: var(--panel-2); border: 1px solid var(--border);
            color: var(--text); font-family: 'JetBrains Mono', monospace; border-radius: 6px;
        }
        .stTextInput>div>div>input:focus { border-color: var(--amber); box-shadow: 0 0 0 1px var(--amber); }
        .stTextInput label {
            font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
            letter-spacing: 0.08em; font-size: 0.72rem; color: var(--text-dim) !important;
        }

        /* Buttons */
        .stButton>button {
            background: var(--amber); color: #0A0E17; font-family: 'Space Grotesk', sans-serif;
            font-weight: 600; border: none; border-radius: 6px; padding: 0.6rem 1.7rem;
            letter-spacing: 0.03em; transition: all 0.15s ease;
        }
        .stButton>button:hover {
            background: #FFC266; box-shadow: 0 0 16px rgba(242,169,59,0.45); transform: translateY(-1px);
        }
        .stDownloadButton>button {
            background: transparent; border: 1px solid var(--teal); color: var(--teal);
            font-family: 'JetBrains Mono', monospace; border-radius: 6px;
        }
        .stDownloadButton>button:hover { background: rgba(52,214,196,0.1); }

        /* Tabs */
        [data-testid="stTabs"] button { font-family: 'Space Grotesk', sans-serif; color: var(--text-dim); }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--amber) !important; border-bottom-color: var(--amber) !important; }

        /* Sliders / checkbox labels */
        [data-testid="stSlider"] label, [data-testid="stCheckbox"] label {
            font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.06em; color: var(--text-dim);
        }

        /* Status widget */
        [data-testid="stStatusWidget"] { background-color: var(--panel-2) !important; border: 1px solid var(--border) !important; border-radius: 8px; }

        /* --- Custom components --- */
        .argus-header { display:flex; justify-content:space-between; align-items:flex-end;
            padding-bottom:1.1rem; border-bottom:1px solid var(--border); margin-bottom:1.6rem; }
        .argus-title { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.1rem;
            letter-spacing:0.14em; color:var(--text); }
        .argus-tagline { display:block; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
            color:var(--text-dim); letter-spacing:0.15em; text-transform:uppercase; margin-top:0.25rem; }
        .argus-status { font-family:'JetBrains Mono',monospace; font-size:0.75rem; letter-spacing:0.08em;
            text-transform:uppercase; display:flex; align-items:center; gap:0.5rem; padding:0.4rem 0.9rem;
            border-radius:999px; border:1px solid var(--border); color:var(--text-dim); }
        .argus-status .dot { width:8px; height:8px; border-radius:50%; background:var(--text-dim); }
        .argus-status--done { color:var(--teal); border-color:var(--teal); }
        .argus-status--done .dot { background:var(--teal); }
        .argus-status--error { color:var(--red); border-color:var(--red); }
        .argus-status--error .dot { background:var(--red); }

        .stations { position:relative; display:flex; justify-content:space-between; flex-wrap:wrap;
            margin: 0.4rem 0 2rem; }
        .stations .line { position:absolute; top:20px; left:6%; right:6%; height:1px; background:var(--border); z-index:0; }
        .stations.done .line { background:var(--teal); }
        .station { position:relative; z-index:1; display:flex; flex-direction:column; align-items:center; flex:1; min-width:80px; }
        .station .node { width:40px; height:40px; border-radius:50%; display:flex; align-items:center;
            justify-content:center; font-family:'JetBrains Mono',monospace; font-size:0.78rem;
            background:var(--panel); border:1px solid var(--border); color:var(--text-dim); }
        .station .label { margin-top:0.5rem; font-family:'Space Grotesk',sans-serif; font-size:0.72rem;
            letter-spacing:0.05em; text-transform:uppercase; color:var(--text-dim); }
        .station.done .node { border-color:var(--teal); color:var(--teal); background:rgba(52,214,196,0.08); }
        .station.done .label { color:var(--teal); }

        .gauge-wrap { position:relative; width:140px; height:140px; animation: gaugeIn 0.5s ease-out; }
        .gauge-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
        .gauge-score { font-family:'JetBrains Mono',monospace; font-size:2.1rem; font-weight:600; color:var(--text); }
        .gauge-label { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-dim); letter-spacing:0.1em; }
        @keyframes gaugeIn { from{opacity:0; transform:scale(0.85);} to{opacity:1; transform:scale(1);} }

        .chip-row { display:flex; gap:0.7rem; flex-wrap:wrap; align-items:stretch; }
        .chip { background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
            padding:0.6rem 1rem; display:flex; flex-direction:column; justify-content:center; min-width:120px; }
        .chip-label { font-family:'JetBrains Mono',monospace; font-size:0.62rem; text-transform:uppercase;
            letter-spacing:0.08em; color:var(--text-dim); }
        .chip-value { font-family:'Space Grotesk',sans-serif; font-size:1.25rem; font-weight:600; color:var(--text); margin-top:0.1rem; }

        .fault-banner { background:rgba(240,85,77,0.08); border:1px solid var(--red); color:var(--red);
            padding:0.9rem 1.1rem; border-radius:8px; margin:1rem 0; font-family:'IBM Plex Sans',sans-serif; }

        /* Chat */
        [data-testid="stChatMessage"] { background-color: var(--panel); border: 1px solid var(--border);
            border-radius: 10px; padding: 0.8rem 1rem; }
        [data-testid="stChatInput"] textarea { font-family: 'IBM Plex Sans', sans-serif; }

        @media (prefers-reduced-motion: reduce) {
            .gauge-wrap, .argus-status .dot { animation: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------
STATUS_LABELS = {"idle": "STANDBY", "done": "COMPLETE", "error": "FAULT"}


def _render_header(stage: str) -> None:
    label = STATUS_LABELS.get(stage, "STANDBY")
    st.markdown(
        f"""
        <div class="argus-header">
            <div>
                <span class="argus-title">ResearchPilot AI</span>
                <span class="argus-tagline">Multi-Agent Research Assistant</span>
            </div>
            <div class="argus-status argus-status--{stage}"><span class="dot"></span>{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_stations(stage: str) -> None:
    state_class = "done" if stage == "done" else ""
    node_class = "done" if stage == "done" else ""
    stations = [("01", "Scout"), ("02", "Reader"), ("03", "Scribe"), ("04", "Critic")]
    items = "".join(
        f'<div class="station {node_class}"><div class="node">{num}</div><div class="label">{name}</div></div>'
        for num, name in stations
    )
    st.markdown(f'<div class="stations {state_class}"><div class="line"></div>{items}</div>', unsafe_allow_html=True)


def _render_gauge(score) -> str:
    r = 54
    circumference = 2 * math.pi * r
    frac = 0.0 if score is None else max(0, min(10, score)) / 10
    dash = circumference * frac
    gap = circumference - dash
    if score is None:
        color = "var(--text-dim)"
    elif score >= 8:
        color = "var(--teal)"
    elif score >= 5:
        color = "var(--amber)"
    else:
        color = "var(--red)"
    display_score = "—" if score is None else str(score)
    return f"""
    <div class="gauge-wrap">
        <svg viewBox="0 0 128 128" width="140" height="140">
            <circle cx="64" cy="64" r="{r}" fill="none" stroke="var(--border)" stroke-width="10"/>
            <circle cx="64" cy="64" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                stroke-linecap="round" stroke-dasharray="{dash:.2f} {gap:.2f}"
                transform="rotate(-90 64 64)"/>
        </svg>
        <div class="gauge-center">
            <div class="gauge-score">{display_score}</div>
            <div class="gauge-label">SIGNAL / 10</div>
        </div>
    </div>
    """


def _chip(label: str, value: str) -> str:
    return f'<div class="chip"><span class="chip-label">{label}</span><span class="chip-value">{value}</span></div>'


# ---------------------------------------------------------------------------
# Pipeline plumbing
# ---------------------------------------------------------------------------
def _supported_params() -> set:
    try:
        return set(inspect.signature(run_research_pipeline).parameters)
    except (TypeError, ValueError):
        return set()


def _call_pipeline(topic: str, **kwargs) -> dict:
    supported = _supported_params()
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return run_research_pipeline(topic, **filtered)


# ---------------------------------------------------------------------------
# Follow-up chat (answers from the agent's own research output)
# ---------------------------------------------------------------------------
def _chat_reply(question: str, history: list, result: dict, topic: str) -> str:
    context = (
        f"REPORT:\n{result.get('report') or ''}\n\n"
        f"SEARCH RESULTS:\n{result.get('search_results') or ''}\n\n"
        f"SCRAPED CONTENT:\n{result.get('scraped_content') or ''}"
    )[:15000]

    messages = [SystemMessage(content=(
        f"You are a helpful research assistant chatting with a user about the topic: '{topic}'. "
        "Answer using the research below. If the answer is not in the research, say so briefly, "
        "then give a short general answer clearly marked as general knowledge. "
        "Keep replies concise and conversational.\n\n" + context
    ))]
    for m in history:
        messages.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                        else AIMessage(content=m["content"]))
    messages.append(HumanMessage(content=question))

    reply = llm.invoke(messages).content
    return reply if isinstance(reply, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in reply
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("result", None)
st.session_state.setdefault("logs", "")
st.session_state.setdefault("topic", "")
st.session_state.setdefault("run_time", None)
st.session_state.setdefault("error", None)
st.session_state.setdefault("stage", "idle")
st.session_state.setdefault("chat", [])

_inject_theme()

# ---------------------------------------------------------------------------
# Sidebar - control panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Control panel")

    params = _supported_params()
    extended = {"max_revisions", "score_threshold", "urls_to_read"} <= params

    max_revisions, score_threshold, urls_to_read, output_dir = 1, 8, 2, "output"
    if extended:
        max_revisions = st.slider("Max revision passes", 0, 3, 1)
        score_threshold = st.slider("Score threshold to revise below", 1, 10, 8)
        urls_to_read = st.slider("Sources to scrape in depth", 1, 5, 2)
        if "output_dir" in params:
            output_dir = st.text_input("Output folder", "output")
    else:
        st.caption("Running the base pipeline. Upgrade pipeline.py to unlock "
                   "revision-loop and multi-source controls here.")

    st.divider()
    show_logs = st.checkbox("Show process logs", value=True)

    if st.session_state.result is not None:
        st.divider()
        if st.button("Clear results"):
            st.session_state.result = None
            st.session_state.logs = ""
            st.session_state.error = None
            st.session_state.stage = "idle"
            st.session_state.chat = []
            st.rerun()

# ---------------------------------------------------------------------------
# Header + pipeline stations
# ---------------------------------------------------------------------------
_render_header(st.session_state.stage)

topic = st.text_input(
    "Research topic",
    placeholder="e.g. Impact of quantum computing on cryptography",
    value=st.session_state.topic,
)
run_clicked = st.button("Run research", type="primary")

_render_stations(st.session_state.stage)

if run_clicked:
    if not topic.strip():
        st.error("Enter a topic first.")
    else:
        st.session_state.topic = topic
        st.session_state.error = None
        st.session_state.chat = []
        log_buffer = io.StringIO()
        start = time.time()

        with st.status("Running the pipeline: search → read → write → critique...", expanded=True):
            try:
                with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                    result = _call_pipeline(
                        topic,
                        max_revisions=max_revisions,
                        score_threshold=score_threshold,
                        urls_to_read=urls_to_read,
                        output_dir=output_dir,
                    )
                st.session_state.result = result
                st.session_state.logs = log_buffer.getvalue()
                st.session_state.run_time = round(time.time() - start, 1)
                st.session_state.stage = "done"
            except Exception as e:
                st.session_state.logs = log_buffer.getvalue()
                st.session_state.error = str(e)
                st.session_state.stage = "error"
        st.rerun()

if st.session_state.error:
    st.markdown(
        f'<div class="fault-banner"><strong>System fault.</strong> {st.session_state.error}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.logs:
        with st.expander("See logs"):
            st.code(st.session_state.logs, language="text")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.result
if result:
    score = result.get("score")
    revisions = result.get("revisions_made")
    urls = result.get("urls")
    n_sources = len(urls) if isinstance(urls, list) else "—"

    gcol, ccol = st.columns([1, 2])
    with gcol:
        st.markdown(_render_gauge(score), unsafe_allow_html=True)
    with ccol:
        suffix = f" in {st.session_state.run_time}s" if st.session_state.run_time else ""
        st.markdown(f"**{st.session_state.topic}**{suffix}")
        chips = _chip("Revisions", revisions if revisions is not None else "—")
        chips += _chip("Sources", n_sources)
        if result.get("output_path"):
            chips += _chip("Saved as", os.path.basename(result["output_path"]))
        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    st.write("")
    tab_report, tab_feedback, tab_search, tab_scraped, tab_logs = st.tabs(
        ["Report", "Critique", "Search results", "Sources", "Logs"]
    )

    with tab_report:
        report_md = result.get("report") or "*No report was generated.*"
        st.markdown(report_md)
        st.download_button(
            "Download report (.md)",
            data=report_md,
            file_name=f"{st.session_state.topic.strip().replace(' ', '_')[:50] or 'report'}.md",
            mime="text/markdown",
        )

    with tab_feedback:
        feedback = result.get("feedback") or "*No feedback available.*"
        st.markdown(feedback if isinstance(feedback, str) else str(feedback))

    with tab_search:
        st.text(result.get("search_results") or "No search results captured.")

    with tab_scraped:
        st.text(result.get("scraped_content") or "No scraped content captured.")

    with tab_logs:
        if show_logs:
            st.code(st.session_state.logs or "No logs captured.", language="text")
        else:
            st.caption("Log display is turned off in the sidebar.")

    # --- Follow-up chat -------------------------------------------------
    st.divider()
    st.markdown("### Ask follow-up questions")
    st.caption("Chat about this topic - answers are based on the report and sources above.")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask something about this topic..."):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = _chat_reply(question, st.session_state.chat, result, st.session_state.topic)
                except Exception as e:
                    answer = f"Sorry, I couldn't answer that: {e}"
            st.markdown(answer)
        st.session_state.chat.append({"role": "user", "content": question})
        st.session_state.chat.append({"role": "assistant", "content": answer})
else:
    st.caption("Enter a topic above and run research to begin. Results will appear here.")