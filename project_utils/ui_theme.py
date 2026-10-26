"""Kinetic Telemetry UI theme (from the Stitch design system) and small HTML helpers."""
import html

import streamlit as st

NEAR = "#a3e635"
FAR = "#7bd0ff"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 2rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: #0a0e18; border-right: 1px solid #313540; }
.bk-hero { display:flex; align-items:center; gap:14px; margin-bottom:.25rem; }
.bk-hero .logo { font-size:2.2rem; }
.bk-hero h1 { margin:0; font-size:2rem; }
.bk-sub { color:#c2cab0; margin:0 0 1.2rem 0; }
.bk-steps { display:flex; gap:8px; margin:0 0 1.5rem 0; flex-wrap:wrap; }
.bk-step { flex:1; min-width:140px; padding:10px 14px; border-radius:8px; background:#1c1f2a;
  border:1px solid #313540; font-family:'JetBrains Mono',monospace; font-size:.78rem; color:#8b9380; }
.bk-step b { display:block; font-family:'Space Grotesk',sans-serif; font-size:.95rem; color:#c2cab0; }
.bk-step.done { border-color:#446900; color:#a3e635; }
.bk-step.done b { color:#e4e8f3; }
.bk-step.active { border-color:#a3e635; box-shadow:0 0 0 1px #a3e63555; color:#a3e635; }
.bk-step.active b { color:#fff; }
.bk-kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:.5rem 0 1.2rem; }
.bk-kpi { background:#1c1f2a; border:1px solid #313540; border-radius:8px; padding:14px 16px; }
.bk-kpi .l { font-family:'JetBrains Mono',monospace; font-size:.7rem; text-transform:uppercase;
  letter-spacing:.08em; color:#8b9380; }
.bk-kpi .v { font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:700; color:#a3e635; line-height:1.2; }
.bk-kpi .s { font-size:.78rem; color:#c2cab0; }
.bk-card { background:#1c1f2a; border:1px solid #313540; border-radius:8px; padding:14px 18px; margin-bottom:10px; }
.bk-card.near { border-left:4px solid #a3e635; }
.bk-card.far { border-left:4px solid #7bd0ff; }
.bk-card.warn { border-left:4px solid #ffb4ab; }
.bk-chip { display:inline-block; padding:2px 10px; border-radius:99px; font-size:.72rem; margin-right:6px;
  font-family:'JetBrains Mono',monospace; background:#313540; color:#e4e8f3; }
.bk-chip.near { background:#334f00; color:#d8ff8a; }
.bk-chip.far { background:#004c69; color:#c3ecff; }
.bk-empty { text-align:center; padding:2.5rem 1rem; color:#c2cab0; border:1px dashed #313540; border-radius:8px; }
.stTabs [data-baseweb="tab"] { font-family:'Space Grotesk',sans-serif; font-size:1rem; }
.stButton > button[kind="primary"] { font-weight:600; color:#213600; }
</style>
"""

STEPS = ["Upload", "Calibrate court", "Analyze", "Insights"]


def inject_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def header():
    st.markdown(
        '<div class="bk-hero"><span class="logo">🏸</span><h1>Badminton AI</h1></div>'
        '<p class="bk-sub">Singles analytics &amp; AI coaching: pose, shuttle tracking, shot recognition.</p>',
        unsafe_allow_html=True,
    )


def stepper(current):
    """current: 0-based index of the active step; earlier steps render as done."""
    items = []
    for i, name in enumerate(STEPS):
        cls = "done" if i < current else "active" if i == current else ""
        items.append(f'<div class="bk-step {cls}">STEP {i + 1}<b>{html.escape(name)}</b></div>')
    st.markdown(f'<div class="bk-steps">{"".join(items)}</div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub)."""
    cells = "".join(
        f'<div class="bk-kpi"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(str(v))}</div>'
        f'<div class="s">{html.escape(s)}</div></div>'
        for l, v, s in items
    )
    st.markdown(f'<div class="bk-kpis">{cells}</div>', unsafe_allow_html=True)


def card(text, tone=""):
    """Plain-text card; tone is '', 'near', 'far' or 'warn'."""
    st.markdown(f'<div class="bk-card {tone}">{html.escape(text)}</div>', unsafe_allow_html=True)


def chip(label, tone=""):
    return f'<span class="bk-chip {tone}">{html.escape(label)}</span>'


def empty_state(title, body):
    st.markdown(
        f'<div class="bk-empty"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )
