"""
Sentix — Multi-Domain Sentiment Analysis
Run: streamlit run app.py
"""

import os
import sys

# Ensure the app's own directory is on the path so `domains/` is always found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import html as _html
import io
import re
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from domains import run_analysis
from domains.shared import load_validation_data

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
DOMAIN_CONFIG = {
    "ppt": {
        "label":        "PPT — Professional Physical Therapy",
        "id_default":   "ID",
        "text_default": "Comments",
        "slug":         "ppt",
    },
    "hilton": {
        "label":        "Hilton — Travel & Hospitality",
        "id_default":   "ID",
        "text_default": "Comments",
        "slug":         "hilton",
    },
    "netflix": {
        "label":        "Netflix — Streaming & Entertainment",
        "id_default":   "Conversation Id",
        "text_default": "transcripts",
        "slug":         "netflix",
    },
    "spotify": {
        "label":        "Spotify — Music Streaming",
        "id_default":   "Conversation Id",
        "text_default": "Message Text (Translate/Original)",
        "slug":         "spotify",
    },
    "godaddy": {
        "label":        "GoDaddy — Domain & Hosting",
        "id_default":   "ID",
        "text_default": "Comments",
        "slug":         "godaddy",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentix",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# Apply any pending auto-detect values BEFORE widgets are instantiated
for _pk, _wk in [
    ("_pending_domain", "sb_domain"),
    ("_pending_id",     "sb_id"),
    ("_pending_text",   "sb_text"),
]:
    if _pk in st.session_state:
        st.session_state[_wk] = st.session_state.pop(_pk)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --teal:#2D5F6E; --teal-l:#3A7A8C; --slate:#6B8A99; --steel:#A8BCC8;
  --warm:#D1CFC4; --warm-l:#F0EEE8; --gold:#D4B94E;
  --bg:#F5F4F0; --card:#FFFFFF; --border:#E0DDD6;
  --text:#1E2D33; --text2:#3D5A66; --muted:#7A95A2;
  --ok:#3D7A5F; --warn:#B8963E; --err:#A04040;
}
html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
.stApp{ background:var(--bg); }
.stApp h1,.stApp h2,.stApp h3{ font-weight:600; color:var(--text); }

/* Sidebar */
section[data-testid="stSidebar"]{ background:var(--warm-l)!important; border-right:1px solid var(--border)!important; }
section[data-testid="stSidebar"] *{ color:var(--text)!important; }
section[data-testid="stSidebar"] .stButton>button{
  justify-content:flex-start!important; text-align:left!important;
  padding:10px 16px!important; border-radius:8px!important; width:100%!important;
  margin-bottom:3px!important; font-weight:500!important; font-size:13px!important;
  background:transparent!important; border:1px solid transparent!important;
  color:var(--text2)!important; transition:all .15s!important; }
section[data-testid="stSidebar"] .stButton>button:hover{
  background:var(--warm)!important; border-color:var(--teal)!important; color:var(--teal)!important; }
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:var(--teal)!important; color:#fff!important; border-color:var(--teal)!important; font-weight:600!important; }

/* Animated metric card */
@keyframes numReveal{
  from{ opacity:0; transform:translateY(8px) scale(.94) }
  to  { opacity:1; transform:translateY(0)   scale(1)   }
}
.mc-anim{
  background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:16px 12px; text-align:center; border-top-width:3px; border-top-style:solid; }
.mv{ font-size:22px; font-weight:700; margin:0; line-height:1.2;
     animation:numReveal .5s cubic-bezier(.22,1,.36,1) both; }
.ml{ font-size:10px; font-weight:600; color:var(--muted); margin:5px 0 0;
     text-transform:uppercase; letter-spacing:.6px; }

/* Executive banner */
.exec-banner{
  background:linear-gradient(135deg,#0C1A20,#162A32); border-radius:12px;
  padding:18px 22px; margin-bottom:18px; border-left:4px solid var(--gold); }
.exec-label{ font-size:10px; font-weight:700; color:var(--gold); letter-spacing:2px;
  text-transform:uppercase; margin-bottom:6px; }
.exec-text{ font-size:13px; color:#C0D8E0; line-height:1.7; }
.exec-text strong{ color:#E4E0D8; }
.exec-stats{ display:flex; gap:20px; flex-wrap:wrap; margin-top:10px; }
.exec-stat{ font-size:12px; font-weight:600; color:#8BBAC8; }
.exec-stat span{ color:#E4E0D8; }

/* Score bar */
.sbar{ display:flex; align-items:center; gap:6px; }
.sbar-t{ flex:1; height:4px; background:#E8E6DD; border-radius:999px; overflow:hidden; min-width:60px; }
.sbar-f{ height:100%; border-radius:999px; }

/* Table — lighter, less aggressive */
.pt{ width:100%; border-collapse:collapse; font-size:12px; }
.pt th{ background:#EFF0EC; color:var(--text2); font-weight:600; padding:8px 12px;
        text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.4px;
        border-bottom:2px solid var(--border); white-space:nowrap; }
.pt td{ padding:7px 12px; border-bottom:1px solid var(--border); color:var(--text);
        vertical-align:middle; max-width:300px; }
.pt td.ellip{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pt tr:hover td{ background:#EEF4F7; }

/* Sentiment chip — inline, no border */
.chip{ display:inline-block; padding:2px 9px; border-radius:4px; font-size:11px; font-weight:600; }
.c-vp{ background:#D4EAE0; color:#2A6048; }
.c-p { background:#E0EDD4; color:#3A5A28; }
.c-n { background:#E8E6DD; color:#555;    }
.c-ng{ background:#F5E8D4; color:#7A4A1A; }
.c-vn{ background:#F2D8D8; color:#7A2828; }

/* Detail card */
.detail-card{ background:var(--card); border:1px solid var(--border); border-radius:10px;
  padding:18px 20px; margin-top:12px; }
.detail-label{ font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:.8px; color:var(--muted); margin-bottom:4px; }
.detail-value{ font-size:13px; color:var(--text); line-height:1.6; }

/* Pagination */
.pg-info{ text-align:center; padding:4px 0; color:var(--muted); font-size:12px; }

/* Tab */
.stTabs [data-baseweb="tab"]{ font-weight:500; color:var(--muted); font-size:13px; }
.stTabs [aria-selected="true"]{ color:var(--teal)!important; border-bottom-color:var(--teal)!important; font-weight:600; }
.stButton>button[kind="primary"]{ background:var(--teal)!important; border-color:var(--teal)!important; color:#FFF!important; font-weight:600!important; }
.stButton>button[kind="primary"]:hover{ background:var(--teal-l)!important; }
.stProgress>div>div>div{ background:var(--teal)!important; }
footer,.stDeployButton{ display:none!important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PII REDACTION
# ─────────────────────────────────────────────────────────────────────────────
_PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"),
    "SSN":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD":  re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b"),
}

def _redact(text: str) -> str:
    if not isinstance(text, str):
        return text
    for ptype, pat in _PII_PATTERNS.items():
        text = pat.sub(f"[{ptype}]", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT FORMAT
# ─────────────────────────────────────────────────────────────────────────────
_FMT_NETFLIX  = re.compile(r'\[\d{2}:\d{2}:\d{2}\s+(CUSTOMER|AGENT|CONSUMER)\]:', re.I)
_FMT_SPOTIFY  = re.compile(r'(?:\|\s*)?\d{4}-\d{2}-\d{2}.*?Consumer:', re.I | re.DOTALL)
_FMT_SPOTIFY2 = re.compile(r'Consumer:', re.I)
_FMT_PPT_HTML = re.compile(r'<b>\d{2}:\d{2}:\d{2}', re.I)
_FMT_PPT_SMS  = re.compile(r'\d{2}:\d{2}:\d{2}\s+\w[\w ]+\s*:\s+\w')

_ID_CANDIDATES = [
    "Conversation Id", "Conversation ID", "conversation_id",
    "CS Ticket ID", "Ticket ID", "ID", "Id", "id",
]

def _detect_domain(df: pd.DataFrame):
    text_col, max_len = None, 0
    for col in df.columns:
        if df[col].dtype == object:
            avg = df[col].dropna().astype(str).str.len().mean()
            if avg and avg > max_len:
                max_len, text_col = avg, col
    if text_col is None or max_len < 15:
        return None

    non_null = df[text_col].dropna().astype(str)
    indices  = np.linspace(0, len(non_null) - 1, min(8, len(non_null)), dtype=int)
    sample   = " ".join(non_null.iloc[i] for i in indices)

    if _FMT_NETFLIX.search(sample):
        domain = "netflix"
    elif _FMT_SPOTIFY.search(sample) or ("|" in sample and _FMT_SPOTIFY2.search(sample)):
        domain = "spotify"
    elif _FMT_PPT_HTML.search(sample) or _FMT_PPT_SMS.search(sample):
        domain = "ppt"
    else:
        domain = "hilton"

    id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    if id_col is None:
        n = max(len(df), 1)
        for col in df.columns:
            if col != text_col:
                try:
                    if df[col].nunique() / n >= 0.5:
                        id_col = col
                        break
                except Exception:
                    pass
    if id_col is None:
        return None

    return domain, id_col, text_col


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_ORDER  = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
_SENT_PILL = {
    "Very Positive": "🟢 Very Positive",
    "Positive":      "🟡 Positive",
    "Neutral":       "⚪ Neutral",
    "Negative":      "🟠 Negative",
    "Very Negative": "🔴 Very Negative",
}
SENTIMENT_COLORS = {
    "Very Positive": "#3D7A5F", "Positive": "#6B9E50",
    "Neutral":       "#6B8A99", "Negative": "#C87A40", "Very Negative": "#A04040",
}
_CHIP_CLASS = {
    "Very Positive": "c-vp", "Positive": "c-p",
    "Neutral": "c-n", "Negative": "c-ng", "Very Negative": "c-vn",
}
_CHIP_ICON = {
    "Very Positive": "▲▲", "Positive": "▲",
    "Neutral": "—", "Negative": "▼", "Very Negative": "▼▼",
}

def _chip(label: str) -> str:
    cls  = _CHIP_CLASS.get(label, "c-n")
    icon = _CHIP_ICON.get(label, "")
    return f'<span class="chip {cls}">{icon} {label}</span>'

def mc_anim(label: str, value: str, color: str = "var(--teal)", delay: float = 0.0) -> str:
    return (
        f'<div class="mc-anim" style="border-top-color:{color}">'
        f'<div class="mv" style="color:{color};animation-delay:{delay:.2f}s">{value}</div>'
        f'<div class="ml">{label}</div>'
        f'</div>'
    )

def _sbar(score: float, domain: str = "") -> str:
    norm = score / 10.0 if domain == "hilton" else score
    norm = max(-1.0, min(1.0, norm))
    pct  = int((norm + 1) / 2 * 100)
    col  = "#3D7A5F" if norm >= 0.2 else "#A04040" if norm <= -0.2 else "#D4B94E"
    return (
        f'<div class="sbar">'
        f'<div class="sbar-t"><div class="sbar-f" style="width:{pct}%;background:{col}"></div></div>'
        f'<span style="color:{col};font-size:11px;font-weight:600;'
        f'font-family:\'JetBrains Mono\',monospace;min-width:46px">{score:+.3f}</span>'
        f'</div>'
    )

def _mfig(fig, h=360):
    fig.update_layout(
        font_family="DM Sans", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=10), height=h,
        xaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,.12)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,.12)", zeroline=False),
        hoverlabel=dict(bgcolor="#1E2D33", font_size=12, font_family="DM Sans", font_color="#E8E6DD"),
    )
    return fig

@st.cache_data(show_spinner=False)
def _read_file(raw, name):
    buf = io.BytesIO(raw)
    return pd.read_csv(buf) if name.endswith(".csv") else pd.read_excel(buf)


_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
    'viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="{sw}">'
    '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
)


def _logo_icon(size: int = 34, sw: float = 1.8) -> str:
    return (
        f'<div style="width:{size}px;height:{size}px;border-radius:{round(size*0.26)}px;'
        f'background:linear-gradient(135deg,#2D5F6E,#3A7A8C);flex-shrink:0;'
        f'display:flex;align-items:center;justify-content:center;'
        f'box-shadow:0 3px 10px rgba(45,95,110,.2)">'
        f'{_LOGO_SVG.format(w=round(size*0.5), h=round(size*0.5), sw=sw)}</div>'
    )


def _sentiment_buckets(dist) -> tuple[int, int, int]:
    """Return (pos_n, neu_n, neg_n) from a sentiment value_counts series/dict."""
    pos_n = int(dist.get("Very Positive", 0)) + int(dist.get("Positive", 0))
    neu_n = int(dist.get("Neutral", 0))
    neg_n = int(dist.get("Negative", 0)) + int(dist.get("Very Negative", 0))
    return pos_n, neu_n, neg_n


def _explode_keywords(series: pd.Series) -> pd.Series:
    """Split '; '-delimited keyword strings into a flat Series of individual keywords."""
    return (
        series.dropna()
        .str.split("; ").explode().str.strip()
        .loc[lambda s: s != ""]
    )


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_landing():
    st.html("""
<style>
/* Hide sidebar + header on landing */
section[data-testid="stSidebar"]{display:none!important}
header[data-testid="stHeader"]{display:none!important}
#MainMenu{display:none!important}
.stApp{background:#0C1418!important}
.block-container{padding:0!important;max-width:100%!important}

/* ── Keyframes ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(32px)}to{opacity:1;transform:translateY(0)}}
@keyframes gradSweep{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes pulse1{0%,100%{opacity:.18;transform:scale(1)}50%{opacity:.30;transform:scale(1.12)}}
@keyframes pulse2{0%,100%{opacity:.12;transform:scale(1)}50%{opacity:.22;transform:scale(1.08)}}
@keyframes pulse3{0%,100%{opacity:.10;transform:scale(1)}50%{opacity:.20;transform:scale(1.15)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes twFade{0%{opacity:0;transform:translateY(6px)}12%{opacity:1;transform:translateY(0)}88%{opacity:1;transform:translateY(0)}100%{opacity:0;transform:translateY(-6px)}}
@keyframes barG{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@keyframes countUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
@keyframes borderPulse{0%,100%{border-color:rgba(212,185,78,.2)}50%{border-color:rgba(212,185,78,.55)}}

/* ── HERO ── */
.lp-hero{position:relative;min-height:100vh;background:#0C1418;overflow:hidden;
  display:flex;flex-direction:column;align-items:center;justify-content:center;padding:72px 24px 56px;text-align:center}
.lp-orb1{position:absolute;width:600px;height:600px;border-radius:50%;
  background:radial-gradient(circle,rgba(45,95,110,.55),transparent 70%);top:-120px;left:-100px;animation:pulse1 8s ease-in-out infinite}
.lp-orb2{position:absolute;width:500px;height:500px;border-radius:50%;
  background:radial-gradient(circle,rgba(212,185,78,.28),transparent 70%);bottom:-60px;right:-80px;animation:pulse2 10s ease-in-out 2s infinite}
.lp-orb3{position:absolute;width:400px;height:400px;border-radius:50%;
  background:radial-gradient(circle,rgba(61,122,95,.22),transparent 70%);bottom:10%;left:30%;animation:pulse3 12s ease-in-out 4s infinite}
.lp-grid{position:absolute;inset:0;
  background-image:linear-gradient(rgba(168,188,200,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(168,188,200,.04) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none}

.lp-badge{display:inline-flex;align-items:center;gap:7px;background:rgba(212,185,78,.08);color:#D4B94E;
  padding:8px 22px;border-radius:24px;font-size:11px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
  border:1px solid rgba(212,185,78,.22);animation:fadeUp .7s ease-out both;margin-bottom:28px;
  animation:fadeUp .7s ease-out both,borderPulse 3s ease-in-out 1s infinite}
.lp-badge-dot{width:7px;height:7px;border-radius:50%;background:#D4B94E;animation:pulse1 2s ease-in-out infinite}

.lp-title{font-size:clamp(52px,8vw,86px);font-weight:700;line-height:1.04;letter-spacing:-2px;
  background:linear-gradient(90deg,#6B8A99,#E8E6DD 18%,#D4B94E 38%,#E8E6DD 58%,#A8BCC8 78%,#6B8A99);
  background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:fadeUp .7s ease-out .15s both,gradSweep 5s linear infinite;margin-bottom:0}
.lp-sub{font-size:clamp(14px,2vw,17px);color:#8BBAC8;font-weight:400;margin:12px 0 0;
  animation:fadeUp .7s ease-out .25s both;letter-spacing:.3px}

.lp-tagline-wrap{height:36px;overflow:hidden;margin:24px 0 40px;animation:fadeUp .7s ease-out .35s both}
.lp-tagline{font-size:clamp(15px,2.2vw,19px);color:#C0D8E0;font-weight:500;
  animation:twFade 3.5s ease-in-out infinite;display:block}

.lp-cta-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:60px;animation:fadeUp .8s ease-out .45s both}
.lp-cta-primary{background:linear-gradient(135deg,#2D5F6E,#3A7A8C);color:#fff;border:none;
  padding:16px 36px;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;
  box-shadow:0 8px 28px rgba(45,95,110,.4);transition:all .25s;font-family:'DM Sans',sans-serif;letter-spacing:.3px}
.lp-cta-primary:hover{transform:translateY(-2px);box-shadow:0 14px 36px rgba(45,95,110,.55)}
.lp-cta-ghost{background:transparent;color:#A8BCC8;border:1px solid rgba(168,188,200,.25);
  padding:16px 36px;border-radius:10px;font-size:15px;font-weight:500;cursor:pointer;
  transition:all .25s;font-family:'DM Sans',sans-serif}
.lp-cta-ghost:hover{border-color:rgba(168,188,200,.55);color:#E8E6DD;background:rgba(168,188,200,.06)}

/* ── MOCKUP WINDOW ── */
.lp-mock-wrap{width:min(780px,92vw);animation:fadeUp .9s ease-out .6s both,float 7s ease-in-out 2s infinite}
.lp-window{background:rgba(18,30,36,.55);backdrop-filter:blur(24px);border:1px solid rgba(168,188,200,.1);
  border-radius:18px;overflow:hidden;box-shadow:0 40px 100px rgba(0,0,0,.55)}
.lp-wintitle{display:flex;align-items:center;gap:8px;padding:14px 20px;
  background:rgba(10,18,22,.7);border-bottom:1px solid rgba(168,188,200,.07)}
.lp-dot{width:12px;height:12px;border-radius:50%}
.lp-winbody{padding:20px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.lp-kpi{background:rgba(255,255,255,.04);border:1px solid rgba(168,188,200,.08);border-radius:10px;
  padding:14px 12px;text-align:center;border-top:3px solid}
.lp-kpi-v{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#E8E6DD;animation:countUp .5s ease-out 1.4s both}
.lp-kpi-l{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:#6B8A99;margin-top:4px}
.lp-bars{padding:0 20px 20px;display:flex;align-items:flex-end;gap:8px;height:110px}
.lp-bar{flex:1;border-radius:6px 6px 0 0;transform-origin:bottom;animation:barG .8s cubic-bezier(.34,1.56,.64,1) both}

/* ── STATS ROW ── */
.lp-stats{display:flex;justify-content:center;gap:56px;flex-wrap:wrap;
  margin-top:56px;animation:fadeUp .7s ease-out .85s both}
.lp-stat-n{font-size:38px;font-weight:700;font-family:'JetBrains Mono',monospace;
  color:#E8E6DD;animation:countUp .6s ease-out 1.2s both;display:block}
.lp-stat-l{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#6B8A99;margin-top:4px}

/* ── DOMAINS SECTION ── */
.lp-domains{background:#F5F4F0;padding:88px 40px;text-align:center}
.lp-sec-badge{display:inline-block;background:rgba(45,95,110,.1);color:#2D5F6E;
  padding:6px 18px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:16px}
.lp-sec-title{font-size:clamp(28px,4vw,40px);font-weight:700;color:#1E2D33;margin:0 0 12px;letter-spacing:-.5px}
.lp-sec-sub{font-size:15px;color:#6B8A99;margin:0 auto 48px;max-width:520px;line-height:1.7}
.lp-domain-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:20px;max-width:1100px;margin:0 auto}
.lp-domain-card{background:#fff;border:1px solid #E0DDD6;border-radius:16px;padding:32px 26px;text-align:left;
  transition:all .35s cubic-bezier(.25,.46,.45,.94);position:relative;overflow:hidden}
.lp-domain-card::after{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#2D5F6E,#D4B94E);transform:scaleX(0);transform-origin:left;transition:transform .35s}
.lp-domain-card:hover{transform:translateY(-8px);box-shadow:0 24px 64px rgba(45,95,110,.13);border-color:#2D5F6E}
.lp-domain-card:hover::after{transform:scaleX(1)}
.lp-domain-icon{font-size:36px;margin-bottom:16px;display:block}
.lp-domain-name{font-size:16px;font-weight:700;color:#1E2D33;margin-bottom:6px}
.lp-domain-tag{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1px;
  color:#fff;background:#2D5F6E;border-radius:4px;padding:2px 8px;display:inline-block;margin-bottom:12px}
.lp-domain-desc{font-size:13px;color:#6B8A99;line-height:1.65}
.lp-domain-feats{margin-top:14px;display:flex;flex-wrap:wrap;gap:6px}
.lp-feat-chip{font-size:10px;font-weight:500;color:#3D5A66;background:#F0EEE8;
  border-radius:4px;padding:3px 9px;border:1px solid #E0DDD6}

/* ── HOW IT WORKS ── */
.lp-hiw{background:#0C1418;padding:88px 40px;text-align:center}
.lp-hiw .lp-sec-title{color:#E8E6DD}
.lp-hiw .lp-sec-sub{color:#8BBAC8}
.lp-steps{display:flex;justify-content:center;align-items:flex-start;gap:0;flex-wrap:wrap;max-width:900px;margin:0 auto}
.lp-step{flex:1;min-width:200px;padding:32px 24px;position:relative;
  background:rgba(22,36,42,.5);backdrop-filter:blur(12px);border:1px solid rgba(168,188,200,.08);
  border-radius:16px;margin:8px;transition:all .35s}
.lp-step:hover{border-color:rgba(212,185,78,.3);transform:translateY(-6px)}
.lp-step-n{width:52px;height:52px;border-radius:50%;
  background:linear-gradient(135deg,#D4B94E,#E8D97A);color:#1E2D33;font-size:22px;font-weight:700;
  display:flex;align-items:center;justify-content:center;margin:0 auto 18px;
  box-shadow:0 6px 24px rgba(212,185,78,.35);animation:float 4s ease-in-out infinite}
.lp-step-n:nth-child(1){animation-delay:0s}
.lp-step-icon{font-size:26px;margin-bottom:10px}
.lp-step-title{font-size:15px;font-weight:600;color:#E8E6DD;margin-bottom:8px}
.lp-step-desc{font-size:12px;color:#8BBAC8;line-height:1.65}

/* ── FEATURES ── */
.lp-features{background:#F5F4F0;padding:88px 40px;text-align:center}
.lp-feat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px;max-width:1100px;margin:0 auto}
.lp-feat-card{background:rgba(255,255,255,.75);backdrop-filter:blur(12px);
  border:1px solid rgba(209,207,196,.5);border-radius:14px;padding:30px 24px;text-align:left;
  transition:all .35s;position:relative;overflow:hidden}
.lp-feat-card:hover{transform:translateY(-6px);box-shadow:0 18px 52px rgba(45,95,110,.1);
  background:rgba(255,255,255,.95);border-color:#2D5F6E}
.lp-feat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#2D5F6E,#D4B94E);transform:scaleX(0);transform-origin:left;transition:transform .35s}
.lp-feat-card:hover::before{transform:scaleX(1)}
.lp-feat-ico{font-size:28px;margin-bottom:14px}
.lp-feat-title{font-size:14px;font-weight:700;color:#1E2D33;margin-bottom:8px}
.lp-feat-desc{font-size:12px;color:#6B8A99;line-height:1.65}

/* ── FOOTER ── */
.lp-footer{background:#080F12;padding:40px;text-align:center;
  border-top:1px solid rgba(168,188,200,.06)}
.lp-footer-name{font-size:18px;font-weight:700;color:#E8E6DD;margin-bottom:6px}
.lp-footer-sub{font-size:12px;color:#6B8A99}
</style>

<div class="lp-hero">
  <div class="lp-grid"></div>
  <div class="lp-orb1"></div><div class="lp-orb2"></div><div class="lp-orb3"></div>

  <div style="position:relative;z-index:2;width:100%;max-width:900px">
    <div class="lp-badge"><span class="lp-badge-dot"></span>MULTI-DOMAIN SENTIMENT INTELLIGENCE</div>
    <h1 class="lp-title">Sentix</h1>
    <p class="lp-sub">See what your customers truly feel — across every channel, every conversation.</p>
    <div class="lp-tagline-wrap">
      <span class="lp-tagline" id="lp-tag">From transcript to insight in seconds.</span>
    </div>

    <div class="lp-mock-wrap" style="margin:0 auto">
      <div class="lp-window">
        <div class="lp-wintitle">
          <div class="lp-dot" style="background:#FF5F57"></div>
          <div class="lp-dot" style="background:#FEBC2E"></div>
          <div class="lp-dot" style="background:#28C840"></div>
          <span style="margin-left:12px;font-size:12px;color:#6B8A99;font-family:'DM Sans',sans-serif">Sentix — Sentiment Analysis</span>
        </div>
        <div class="lp-winbody">
          <div class="lp-kpi" style="border-top-color:#3D7A5F">
            <div class="lp-kpi-v" style="color:#3D7A5F">68%</div>
            <div class="lp-kpi-l">Positive</div>
          </div>
          <div class="lp-kpi" style="border-top-color:#6B8A99">
            <div class="lp-kpi-v" style="color:#6B8A99">18%</div>
            <div class="lp-kpi-l">Neutral</div>
          </div>
          <div class="lp-kpi" style="border-top-color:#C87A40">
            <div class="lp-kpi-v" style="color:#C87A40">14%</div>
            <div class="lp-kpi-l">Negative</div>
          </div>
          <div class="lp-kpi" style="border-top-color:#D4B94E">
            <div class="lp-kpi-v" style="color:#D4B94E">0.94</div>
            <div class="lp-kpi-l">Confidence</div>
          </div>
        </div>
        <div class="lp-bars">
          <div class="lp-bar" style="height:82%;background:linear-gradient(180deg,#3D7A5F,#2D5F4A);animation-delay:.8s"></div>
          <div class="lp-bar" style="height:55%;background:linear-gradient(180deg,#3D7A5F,#2D5F4A);animation-delay:.9s"></div>
          <div class="lp-bar" style="height:70%;background:linear-gradient(180deg,#3D7A5F,#2D5F4A);animation-delay:1.0s"></div>
          <div class="lp-bar" style="height:38%;background:linear-gradient(180deg,#6B8A99,#4A6870);animation-delay:1.1s"></div>
          <div class="lp-bar" style="height:22%;background:linear-gradient(180deg,#C87A40,#A05828);animation-delay:1.2s"></div>
          <div class="lp-bar" style="height:60%;background:linear-gradient(180deg,#3D7A5F,#2D5F4A);animation-delay:1.3s"></div>
          <div class="lp-bar" style="height:88%;background:linear-gradient(180deg,#D4B94E,#B89830);animation-delay:1.4s"></div>
          <div class="lp-bar" style="height:44%;background:linear-gradient(180deg,#3D7A5F,#2D5F4A);animation-delay:1.5s"></div>
        </div>
      </div>
    </div>

    <div class="lp-stats">
      <div style="text-align:center"><span class="lp-stat-n">4</span><div class="lp-stat-l">Customer Domains</div></div>
      <div style="text-align:center"><span class="lp-stat-n">5</span><div class="lp-stat-l">Sentiment Levels</div></div>
      <div style="text-align:center"><span class="lp-stat-n">3</span><div class="lp-stat-l">AI Engines</div></div>
      <div style="text-align:center"><span class="lp-stat-n">1-click</span><div class="lp-stat-l">Export</div></div>
    </div>
  </div>
</div>

<!-- DOMAINS -->
<div class="lp-domains">
  <div class="lp-sec-badge">Supported Domains</div>
  <h2 class="lp-sec-title">Built for every customer channel</h2>
  <p class="lp-sec-sub">Purpose-built extraction and classification pipelines for each domain — not a one-size-fits-all model.</p>
  <div class="lp-domain-grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">
    <div class="lp-domain-card">
      <span class="lp-domain-icon">🏥</span>
      <div class="lp-domain-name">Physical Therapy</div>
      <span class="lp-domain-tag">PPT</span>
      <div class="lp-domain-desc">HTML & SMS transcript parsing with customer-speaker isolation. Detects appointment, treatment, and billing frustrations.</div>
      <div class="lp-domain-feats"><span class="lp-feat-chip">HTML transcripts</span><span class="lp-feat-chip">SMS logs</span><span class="lp-feat-chip">Speaker detection</span></div>
    </div>
    <div class="lp-domain-card">
      <span class="lp-domain-icon">🏨</span>
      <div class="lp-domain-name">Hilton Hospitality</div>
      <span class="lp-domain-tag">Hilton</span>
      <div class="lp-domain-desc">Multilingual feedback analysis with AFINN + TextBlob + VADER hybrid scoring. Auto-translates Spanish, French, and more.</div>
      <div class="lp-domain-feats"><span class="lp-feat-chip">Multilingual</span><span class="lp-feat-chip">Hybrid scoring</span><span class="lp-feat-chip">Auto-translate</span></div>
    </div>
    <div class="lp-domain-card">
      <span class="lp-domain-icon">🎬</span>
      <div class="lp-domain-name">Netflix Streaming</div>
      <span class="lp-domain-tag">Netflix</span>
      <div class="lp-domain-desc">Pipe-delimited transcript parsing with consumer-turn extraction. Rule + VADER pipeline tuned for streaming support issues.</div>
      <div class="lp-domain-feats"><span class="lp-feat-chip">Pipe-delimited</span><span class="lp-feat-chip">VADER adaptive</span><span class="lp-feat-chip">BERT correction</span></div>
    </div>
    <div class="lp-domain-card">
      <span class="lp-domain-icon">🎵</span>
      <div class="lp-domain-name">Spotify Music</div>
      <span class="lp-domain-tag">Spotify</span>
      <div class="lp-domain-desc">Consumer message extraction with language gating. DistilBERT correction for borderline scores in the ±0.1 zone.</div>
      <div class="lp-domain-feats"><span class="lp-feat-chip">Language detection</span><span class="lp-feat-chip">DistilBERT</span><span class="lp-feat-chip">Borderline tuning</span></div>
    </div>
    <div class="lp-domain-card">
      <span class="lp-domain-icon">🌐</span>
      <div class="lp-domain-name">GoDaddy</div>
      <span class="lp-domain-tag">GoDaddy</span>
      <div class="lp-domain-desc">Domain & hosting support analysis. Covers DNS issues, email setup failures, OTP/auth problems, WordPress errors, SSL, and auto-renewal disputes.</div>
      <div class="lp-domain-feats"><span class="lp-feat-chip">DNS & hosting</span><span class="lp-feat-chip">10 issue categories</span><span class="lp-feat-chip">VADER + BERT</span></div>
    </div>
  </div>
</div>

<!-- HOW IT WORKS -->
<div class="lp-hiw">
  <div class="lp-sec-badge" style="background:rgba(212,185,78,.1);color:#D4B94E;border:1px solid rgba(212,185,78,.2)">How It Works</div>
  <h2 class="lp-sec-title" style="color:#E8E6DD">Three steps to complete insight</h2>
  <p class="lp-sec-sub" style="color:#8BBAC8">No configuration required. Upload your data, select the domain, and get results.</p>
  <div class="lp-steps">
    <div class="lp-step">
      <div class="lp-step-n">1</div>
      <div class="lp-step-icon">📁</div>
      <div class="lp-step-title">Upload Your Data</div>
      <div class="lp-step-desc">Drop in an Excel or CSV with your conversation transcripts. Auto-detect identifies the domain and column mapping instantly.</div>
    </div>
    <div class="lp-step">
      <div class="lp-step-n" style="animation-delay:.5s">2</div>
      <div class="lp-step-icon">⚡</div>
      <div class="lp-step-title">Analyse at Scale</div>
      <div class="lp-step-desc">Parallel VADER scoring, rule-based overrides, and optional DistilBERT correction — all in one click. Thousands of rows in seconds.</div>
    </div>
    <div class="lp-step">
      <div class="lp-step-n" style="animation-delay:1s">3</div>
      <div class="lp-step-icon">📊</div>
      <div class="lp-step-title">Act on Insights</div>
      <div class="lp-step-desc">Drill into keyword categories, inspect individual records with full lineage, and export a 3-sheet Excel workbook for stakeholders.</div>
    </div>
  </div>
</div>

<!-- FEATURES -->
<div class="lp-features">
  <div class="lp-sec-badge">Capabilities</div>
  <h2 class="lp-sec-title">Everything you need to understand customers</h2>
  <p class="lp-sec-sub">From raw transcripts to executive-ready reports — no data science team required.</p>
  <div class="lp-feat-grid">
    <div class="lp-feat-card">
      <div class="lp-feat-ico">🧠</div>
      <div class="lp-feat-title">Adaptive AI Pipeline</div>
      <div class="lp-feat-desc">Three-layer classification: rule matching → VADER fallback → DistilBERT correction. Confidence-weighted blending for maximum accuracy.</div>
    </div>
    <div class="lp-feat-card">
      <div class="lp-feat-ico">✅</div>
      <div class="lp-feat-title">Validation Overrides</div>
      <div class="lp-feat-desc">Upload a ground-truth Excel to lock in human-verified labels. Overrides take highest priority — model predictions never touch validated rows.</div>
    </div>
    <div class="lp-feat-card">
      <div class="lp-feat-ico">🔍</div>
      <div class="lp-feat-title">Keyword Intelligence</div>
      <div class="lp-feat-desc">Category-level negative keyword breakdown with drill-down charts, sentiment mix per category, and highlighted sample comments.</div>
    </div>
    <div class="lp-feat-card">
      <div class="lp-feat-ico">🔒</div>
      <div class="lp-feat-title">PII Redaction</div>
      <div class="lp-feat-desc">Toggle to mask emails, phone numbers, SSNs, and card numbers in the display — without altering the underlying analysis or export.</div>
    </div>
    <div class="lp-feat-card">
      <div class="lp-feat-ico">📋</div>
      <div class="lp-feat-title">Audit Trail</div>
      <div class="lp-feat-desc">Full decision lineage for every record — see which rule fired, the raw VADER score, BERT correction, and final confidence step by step.</div>
    </div>
    <div class="lp-feat-card">
      <div class="lp-feat-ico">📤</div>
      <div class="lp-feat-title">One-Click Export</div>
      <div class="lp-feat-desc">Download a 3-sheet Excel workbook (Results · Summary · Keywords) or a filtered CSV — timestamp-named and ready for stakeholders.</div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<div class="lp-footer">
  <div class="lp-footer-name">Sentix</div>
  <div class="lp-footer-sub">Multi-Domain Sentiment Intelligence · VADER Adaptive · Built with Streamlit</div>
</div>

<script>
const tags = [
  "From transcript to insight in seconds.",
  "100% of conversations. Zero manual effort.",
  "Know what your customers truly feel.",
  "Four domains. One platform. Total clarity.",
];
let ti = 0;
const el = document.getElementById("lp-tag");
if(el){
  setInterval(() => {
    el.style.animation = "none";
    void el.offsetWidth;
    ti = (ti + 1) % tags.length;
    el.textContent = tags[ti];
    el.style.animation = "twFade 3.5s ease-in-out infinite";
  }, 3500);
}
</script>
""")

    # ── CTA button rendered by Streamlit (so it can trigger navigation) ────────
    _, col, _ = st.columns([2, 1, 2])
    with col:
        if st.button("🚀  Start Analysing", type="primary", key="lp_cta", use_container_width=True):
            st.session_state._page = "Upload & Analyse"
            st.rerun()


def _page_state(id_col, text_col, domain):
    """Retrieve per-page column/domain state, falling back to sidebar values."""
    return (
        st.session_state.get("_id_col",   id_col),
        st.session_state.get("_text_col", text_col),
        st.session_state.get("_domain",   domain),
    )


def _csv_download(df: pd.DataFrame, label: str, slug: str, key: str, suffix: str = "sentiment"):
    st.download_button(
        label,
        data=df.to_csv(index=False).encode(),
        file_name=f"{slug}_{suffix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", width='stretch', key=key,
    )


@st.cache_data(show_spinner=False)
def _build_cat_summary(text_values: tuple, kw_dict_items: tuple, total: int):
    s = pl.Series("t", list(text_values))
    result = []
    for cat, keywords in kw_dict_items:
        if not keywords:
            continue
        pattern = "(?i)" + "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
        mask = s.str.contains(pattern, literal=False).fill_null(False).to_list()
        cnt  = sum(mask)
        if cnt > 0:
            result.append({
                "cat": cat, "label": cat.replace("_", " ").title(),
                "records": cnt, "pct": round(cnt / total * 100, 1), "mask": mask,
            })
    result.sort(key=lambda x: x["records"], reverse=True)
    return result


def _exec_banner(dist, total, pt, val_n, domain_label):
    pos_n, neu_n, neg_n = _sentiment_buckets(dist)
    neg_pct = neg_n / total if total else 0
    pos_pct = pos_n / total if total else 0

    if pos_pct >= 0.5:
        mood = f"<strong style='color:#7FD9AC'>{pos_pct:.0%} positive</strong>"
    elif neg_pct >= 0.4:
        mood = f"<strong style='color:#F4A0A0'>⚠ {neg_pct:.0%} negative</strong>"
    else:
        mood = (f"<strong style='color:#7FD9AC'>{pos_pct:.0%} positive</strong> and "
                f"<strong style='color:#F4A0A0'>{neg_pct:.0%} negative</strong>")

    val_note = f", with <strong>{val_n:,}</strong> validation overrides" if val_n else ""
    summary  = (
        f"<strong>{total:,} {domain_label} records</strong> analysed in "
        f"<strong>{pt:.1f}s</strong>{val_note}. Sentiment: {mood}."
    )
    stats = [
        (f"{pos_pct:.0%}", "Positive"),
        (f"{neu_n / total:.0%}" if total else "0%", "Neutral"),
        (f"{neg_pct:.0%}", "Negative"),
    ]
    if val_n:
        stats.append((f"{val_n:,}", "Validated"))
    stats_html = "".join(
        f'<span class="exec-stat"><span>{v}</span> {l}</span>' for v, l in stats
    )
    st.markdown(
        f'<div class="exec-banner">'
        f'<div class="exec-label">Executive Summary</div>'
        f'<div class="exec-text">{summary}</div>'
        f'<div class="exec-stats">{stats_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _get_kw_dict(dom):
    if dom == "hilton":
        from domains.hilton import HILTON_NEGATIVE_KEYWORDS
        return HILTON_NEGATIVE_KEYWORDS
    elif dom == "netflix":
        from domains.netflix import NETFLIX_NEGATIVE_KEYWORDS
        return NETFLIX_NEGATIVE_KEYWORDS
    elif dom == "spotify":
        from domains.spotify import SPOTIFY_NEGATIVE_KEYWORDS
        return SPOTIFY_NEGATIVE_KEYWORDS
    elif dom == "godaddy":
        from domains.godaddy import GODADDY_NEGATIVE_KEYWORDS
        return GODADDY_NEGATIVE_KEYWORDS
    else:
        from domains.ppt import NEGATIVE_KEYWORDS
        return NEGATIVE_KEYWORDS


def _detail_card(row: pd.Series, text_cols: set, score_col: str,
                 sent_col: str, domain: str, redact: bool = False):
    """Render a full detail panel for one record."""
    fields = []
    for col, val in row.items():
        if col.startswith("_"):
            continue
        if col == sent_col:
            fields.append((col, _chip(str(val)), True))
        elif col == score_col:
            try:
                fields.append((col, _sbar(float(val), domain), True))
            except (ValueError, TypeError):
                fields.append((col, _html.escape(str(val)), True))
        elif col in text_cols and isinstance(val, str):
            display = _redact(val) if redact else val
            fields.append((col, _html.escape(display), True))
        elif not (pd.isna(val) if not isinstance(val, str) else False):
            fields.append((col, _html.escape(str(val)), False))

    html_parts = ['<div class="detail-card">']
    # Long text fields last
    short_fields = [(l, v, h) for l, v, h in fields if not (len(v) > 120)]
    long_fields  = [(l, v, h) for l, v, h in fields if len(v) > 120]

    # 3-column grid for short fields
    html_parts.append('<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px 24px;margin-bottom:14px">')
    for label, val, is_html in short_fields:
        display_val = val if is_html else _html.escape(str(val))
        html_parts.append(
            f'<div><div class="detail-label">{_html.escape(label)}</div>'
            f'<div class="detail-value">{display_val}</div></div>'
        )
    html_parts.append('</div>')

    # Full-width long text fields
    for label, val, _ in long_fields:
        html_parts.append(
            f'<div style="margin-top:10px"><div class="detail-label">{_html.escape(label)}</div>'
            f'<div class="detail-value" style="white-space:pre-wrap;word-break:break-word">{val}</div></div>'
        )
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _on_domain_change():
    domain = st.session_state["sb_domain"]
    cfg    = DOMAIN_CONFIG[domain]
    cols   = st.session_state.get("_file_cols")
    if cols:
        if cfg["id_default"] in cols:
            st.session_state["sb_id"] = cfg["id_default"]
        if cfg["text_default"] in cols:
            st.session_state["sb_text"] = cfg["text_default"]
    else:
        st.session_state["sb_id"]   = cfg["id_default"]
        st.session_state["sb_text"] = cfg["text_default"]

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:10px 0 14px">'
        f'{_logo_icon(34, 1.8)}'
        f'<div><span style="font-size:15px;font-weight:700;color:#1E2D33">Sentix</span><br>'
        f'<span style="font-size:11px;color:#7A95A2">Multi-Domain · VADER Adaptive</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    PAGES = ["Home", "Upload & Analyse", "Reports & Insights", "Keyword Analysis", "Audit Trail"]
    if "_page" not in st.session_state:
        st.session_state._page = "Home"
    nav_target = st.session_state.pop("_nav_target", None)
    if nav_target and nav_target in PAGES:
        st.session_state._page = nav_target

    NAV_ICONS = {
        "Home":               ":material/home:",
        "Upload & Analyse":   ":material/upload:",
        "Reports & Insights": ":material/analytics:",
        "Keyword Analysis":   ":material/search:",
        "Audit Trail":        ":material/visibility:",
    }
    for p in PAGES:
        active = st.session_state._page == p
        if st.button(p, key=f"nav_{p}", icon=NAV_ICONS[p], width='stretch',
                     type="primary" if active else "secondary"):
            st.session_state._page = p
            st.rerun()
    page = st.session_state._page
    st.divider()

    st.caption("DOMAIN")
    domain = st.selectbox(
        "Select domain",
        list(DOMAIN_CONFIG.keys()),
        format_func=lambda k: DOMAIN_CONFIG[k]["label"],
        key="sb_domain",
        on_change=_on_domain_change,
        label_visibility="collapsed",
    )
    _dcfg = DOMAIN_CONFIG[domain]
    st.divider()

    st.caption("COLUMN MAPPING")
    _file_cols = st.session_state.get("_file_cols")
    if _file_cols:
        _id_val = st.session_state.get("sb_id",   _dcfg["id_default"])
        _tx_val = st.session_state.get("sb_text", _dcfg["text_default"])
        if _id_val not in _file_cols:
            st.session_state["sb_id"]   = _file_cols[0]
        if _tx_val not in _file_cols:
            st.session_state["sb_text"] = _file_cols[min(1, len(_file_cols) - 1)]
        id_col   = st.selectbox("ID Column",           _file_cols, key="sb_id")
        text_col = st.selectbox("Conversation Column", _file_cols, key="sb_text")
    else:
        id_col   = st.text_input("ID Column",           value=_dcfg["id_default"],   key="sb_id")
        text_col = st.text_input("Conversation Column", value=_dcfg["text_default"], key="sb_text")
    st.divider()

    st.caption("VALIDATION OVERRIDE")
    val_file = st.file_uploader(
        "Validation Excel (optional)", type=["xlsx"], key="sb_val",
        help="Must have ID and 'Actual Sentiment' columns.",
        label_visibility="collapsed",
    )
    st.divider()

    st.caption("OPTIONS")
    rule_threshold = st.slider("Rule confidence threshold", 0.50, 0.99, 0.70, 0.05,
                                label_visibility="visible")
    redact_pii = st.toggle("Redact PII in display", value=False,
                           help="Masks emails, phone numbers, SSNs and card numbers.")


# ─────────────────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0 — HOME / LANDING
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":
    render_landing()
    st.stop()

st.markdown(
    f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
    f'{_logo_icon(40, 1.6)}'
    f'<div><h1 style="margin:0;font-size:22px;line-height:1.2;color:#1E2D33">Sentix</h1>'
    f'<p style="margin:0;color:#7A95A2;font-size:12px">Multi-Domain Sentiment Intelligence · VADER Adaptive</p></div>'
    f'</div>',
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — UPLOAD & ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Upload & Analyse":

    if "result" in st.session_state:
        out = st.session_state.result
        fn  = st.session_state.get("_filename", "data")
        t   = st.session_state.get("_run_time", 0)
        v   = int((out["validation_source"] == "Validation").sum())
        extra = f" · {v:,} validation overrides" if v else ""
        st.caption(f"Last run: **{_html.escape(fn)}** — {len(out):,} records in {t:.1f}s{extra}. View results in Reports ›")

    uploaded = st.file_uploader(
        "Upload conversation data (Excel or CSV)",
        type=["xlsx", "csv"], key="main_upload",
    )
    if uploaded is None:
        st.markdown(
            '<p style="color:var(--muted);font-size:13px;margin-top:8px">'
            'Upload an Excel or CSV file with an ID column and a conversation/comments column.'
            '</p>',
            unsafe_allow_html=True,
        )
        st.stop()

    raw_bytes = uploaded.read()
    file_hash = hashlib.md5(raw_bytes).hexdigest()[:10]

    with st.spinner("Reading file…"):
        df_raw = _read_file(raw_bytes, uploaded.name)

    # Store columns for sidebar selectboxes on new file
    if st.session_state.get("_last_file_hash") != file_hash:
        st.session_state["_file_cols"]      = list(df_raw.columns)
        st.session_state["_last_file_hash"] = file_hash

    # Auto-detect on first load of this file
    if st.session_state.get("_last_autodetect") != file_hash:
        detected = _detect_domain(df_raw)
        if detected:
            det_domain, det_id, det_text = detected
            st.session_state["_pending_domain"] = det_domain
            st.session_state["_pending_id"]     = det_id
            st.session_state["_pending_text"]   = det_text
        st.session_state["_last_autodetect"] = file_hash
        st.rerun()

    # File info line
    det_label = DOMAIN_CONFIG[domain]["label"].split(" — ")[0]
    st.markdown(
        f'<p style="color:var(--muted);font-size:12px;margin:4px 0 12px">'
        f'<strong style="color:var(--text)">{len(df_raw):,} rows</strong> · '
        f'{len(df_raw.columns)} columns · '
        f'Auto-detected as <strong style="color:var(--teal)">{det_label}</strong> · '
        f'ID: <code>{id_col}</code> · Text: <code>{text_col}</code>'
        f'</p>',
        unsafe_allow_html=True,
    )

    missing = [c for c in [id_col, text_col] if c not in df_raw.columns]
    if missing:
        st.error(f"Column(s) **{missing}** not found. Available: `{list(df_raw.columns)}`")
        st.stop()

    with st.expander("Preview (first 5 rows)"):
        st.dataframe(df_raw[[id_col, text_col]].head(5), width='stretch')

    validation_dict = {}
    if val_file:
        val_file.seek(0)
        validation_dict = load_validation_data(val_file)
        if validation_dict:
            st.caption(f"Validation file loaded — {len(validation_dict):,} records.")
        else:
            st.warning("Validation file loaded but no records matched expected columns.")
    st.divider()

    _, rc, _ = st.columns([1, 2, 1])
    with rc:
        _btn_label = DOMAIN_CONFIG[domain]["label"].split(" — ")[0]
        run = st.button(
            f"Run {_btn_label} Analysis  ({len(df_raw):,} records)",
            type="primary", width='stretch',
        )

    if run:
        prog = st.progress(0, text="Starting…")
        t0   = time.time()

        def update_progress(done, total):
            pct = int(done / total * 100) if total else 0
            prog.progress(pct, text=f"Processing {done:,} / {total:,} rows…")

        with st.spinner("Running analysis…"):
            result_df = run_analysis(
                df_raw, domain, id_col, text_col,
                validation_dict, update_progress, rule_threshold,
            )

        elapsed = time.time() - t0
        prog.progress(100, text=f"Done — {len(result_df):,} records in {elapsed:.1f}s")

        st.session_state.result          = result_df
        st.session_state._run_time       = elapsed
        st.session_state._filename       = uploaded.name
        st.session_state._id_col         = id_col
        st.session_state._text_col       = text_col
        st.session_state._domain         = domain
        st.session_state._rule_threshold = rule_threshold

        st.toast(f"Done — {len(result_df):,} records in {elapsed:.1f}s")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — REPORTS & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Reports & Insights":

    if "result" not in st.session_state:
        st.info("Run analysis first — go to Upload & Analyse.")
        st.stop()

    out                      = st.session_state.result
    id_col, text_col, _domain = _page_state(id_col, text_col, domain)
    total    = len(out)
    dist     = out["consumer_sentiment"].value_counts()
    pt       = st.session_state.get("_run_time", 0)
    val_n              = int((out["validation_source"] == "Validation").sum())
    pos_n, neu_n, neg_n = _sentiment_buckets(dist)

    _exec_banner(dist, total, pt, val_n, DOMAIN_CONFIG[_domain]["label"].split(" — ")[0])

    # KPI row — 5 cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(mc_anim("Total Records",   f"{total:,}",                      delay=0.00), unsafe_allow_html=True)
    k2.markdown(mc_anim("Analysis Time",   f"{pt:.1f}s", "var(--slate)",      delay=0.05), unsafe_allow_html=True)
    k3.markdown(mc_anim("Negative",        f"{neg_n:,}", "#C87A40",           delay=0.10), unsafe_allow_html=True)
    k4.markdown(mc_anim("Neutral",         f"{neu_n:,}", "var(--slate)", delay=0.15), unsafe_allow_html=True)
    k5.markdown(mc_anim("Positive",        f"{pos_n:,}", "#3D7A5F",           delay=0.20), unsafe_allow_html=True)

    st.divider()

    tab_summary, tab_evidence = st.tabs(["Summary", "Evidence"])

    # ── Summary ───────────────────────────────────────────────────────────────
    with tab_summary:
        ordered = [s for s in SENTIMENT_ORDER if s in dist.index]
        counts  = [int(dist[s]) for s in ordered]
        colors  = [SENTIMENT_COLORS[s] for s in ordered]
        pcts    = [round(c / total * 100, 1) for c in counts]

        col_donut, col_bar = st.columns([1, 2])

        with col_donut:
            bucket_labels = ["Positive", "Neutral", "Negative"]
            bucket_colors = ["#3D7A5F", "#6B8A99", "#A04040"]
            bucket_counts = [pos_n, neu_n, neg_n]
            fig_donut = go.Figure(go.Pie(
                labels=bucket_labels,
                values=bucket_counts,
                marker=dict(colors=bucket_colors, line=dict(width=0)),
                hole=0.62,
                textinfo="percent",
                textfont=dict(size=12, family="DM Sans"),
                hovertemplate="<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
            ))
            _mfig(fig_donut, 300).update_layout(
                title=dict(text="Sentiment Breakdown", font=dict(size=13, color="#1E2D33"), x=0),
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_donut, width='stretch', key="rpt_donut")

        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=ordered, y=counts,
                marker=dict(color=colors, cornerradius=5, line=dict(width=0)),
                text=[f"{p}%" for p in pcts], textposition="outside",
                textfont=dict(size=12, color="#2D5F6E", family="DM Sans"),
                hovertemplate="<b>%{x}</b><br>%{y:,} records (%{text})<extra></extra>",
            ))
            _mfig(fig_bar, 300).update_layout(
                title=dict(text="Sentiment Distribution", font=dict(size=13, color="#1E2D33"), x=0),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, width='stretch', key="rpt_bar")

        # Key stats row
        scores_norm = out["consumer_score"] / 10.0 if _domain == "hilton" else out["consumer_score"]
        valid_scores = scores_norm.dropna()
        high_conf_n  = int((out["confidence"] > 0.7).sum())
        val_pct      = round(val_n / total * 100, 1) if total else 0

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.markdown(mc_anim("Avg Score",        f"{valid_scores.mean():+.3f}", "var(--teal)",  delay=0.00), unsafe_allow_html=True)
        sk2.markdown(mc_anim("Median Score",     f"{valid_scores.median():+.3f}", "var(--slate)", delay=0.05), unsafe_allow_html=True)
        sk3.markdown(mc_anim("High Confidence",  f"{high_conf_n:,}", "#3D7A5F",  delay=0.10), unsafe_allow_html=True)
        sk4.markdown(mc_anim("Validation Override", f"{val_pct}%", "#D4B94E",   delay=0.15), unsafe_allow_html=True)

        # Top negative keywords strip
        if "Negative_Keywords" in out.columns:
            top_kws = (
                _explode_keywords(out["Negative_Keywords"])
                .value_counts().head(7).index.tolist()
            )
            if top_kws:
                kw_html = " &nbsp;·&nbsp; ".join(f"<code>{_html.escape(kw)}</code>" for kw in top_kws)
                st.markdown(
                    f'<div style="background:var(--warm-l);border-radius:8px;padding:11px 16px;'
                    f'margin-top:14px;border-left:3px solid var(--gold);font-size:13px;color:var(--text2)">'
                    f'<strong>Top negative signals:</strong> {kw_html}</div>',
                    unsafe_allow_html=True,
                )

    # ── Evidence ──────────────────────────────────────────────────────────────
    with tab_evidence:
        filt_kw = st.text_input("Search keywords / text", placeholder="e.g. wait time, billing…", key="rpt_filt_kw")
        filt_sent = st.pills("Sentiment", SENTIMENT_ORDER, selection_mode="multi", default=None, key="rpt_filt_sent")
        filt_src  = st.pills("Source", ["Validation", "Model"], selection_mode="multi", default=None, key="rpt_filt_src")

        disp = out
        if filt_sent: disp = disp[disp["consumer_sentiment"].isin(filt_sent)]
        if filt_src:  disp = disp[disp["validation_source"].isin(filt_src)]
        if filt_kw.strip():
            _kw_lower = filt_kw.strip().lower()
            _search_cols = [c for c in ["Negative_Keywords", "Text_For_Analysis", text_col] if c in disp.columns]
            _mask = pd.Series(False, index=disp.index)
            for _sc in _search_cols:
                _mask |= disp[_sc].fillna("").astype(str).str.lower().str.contains(_kw_lower, regex=False)
            disp = disp[_mask]

        ev_cols = [c for c in [id_col, "consumer_sentiment", "consumer_score", "confidence", "validation_source", "Negative_Keywords"] if c in disp.columns]
        ev_df   = disp[ev_cols].reset_index(drop=True)
        if "consumer_sentiment" in ev_df.columns:
            ev_df = ev_df.assign(consumer_sentiment=ev_df["consumer_sentiment"].map(_SENT_PILL).fillna(ev_df["consumer_sentiment"]))

        st.caption(f"{len(ev_df):,} records — click a row to inspect")
        ev_sel = st.dataframe(
            ev_df,
            on_select="rerun",
            selection_mode="single-row",
            width='stretch',
            height=360,
            key="rpt_ev_sel",
            column_config={
                "consumer_sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
                "consumer_score":     st.column_config.NumberColumn("Score", format="%.3f", width="small"),
                "confidence":         st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f", width="small"),
                "validation_source":  st.column_config.TextColumn("Source", width="small"),
                "Negative_Keywords":  st.column_config.TextColumn("Keywords", width="large"),
            },
        )
        selected_rows = ev_sel.selection.rows if hasattr(ev_sel, "selection") else []
        if selected_rows:
            orig_idx = disp.index[selected_rows[0]]
            detail_text_cols = {text_col, "CustomerOnly", "CustomerOnly_Cleaned",
                                "Text_For_Analysis", "Negative_Keywords", "Translated_Text"}
            with st.expander("Record detail", expanded=True):
                _detail_card(out.loc[orig_idx], detail_text_cols,
                             "consumer_score", "consumer_sentiment", _domain, redact_pii)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Export**")
    e1, e2 = st.columns(2)
    with e1:
        _csv_download(out, "Download CSV", _domain, key="dl_csv")
    with e2:
        xls_buf = io.BytesIO()
        with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="Results", index=False)
            summary_rows = []
            for sent in SENTIMENT_ORDER:
                cnt = int(dist.get(sent, 0))
                summary_rows.append({"Sentiment": sent, "Count": cnt,
                                     "%": round(cnt / total * 100, 1) if total else 0})
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if "Negative_Keywords" in out.columns:
                kw_all = _explode_keywords(out["Negative_Keywords"])
                if not kw_all.empty:
                    kw_df = kw_all.value_counts().reset_index()
                    kw_df.columns = ["Keyword", "Count"]
                    kw_df.to_excel(writer, sheet_name="Keywords", index=False)
        st.download_button(
            "Download Excel (3 sheets)",
            data=xls_buf.getvalue(),
            file_name=f"{_domain}_sentiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch', key="dl_excel",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — KEYWORD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Keyword Analysis":

    if "result" not in st.session_state:
        st.info("Run analysis first."); st.stop()

    out                           = st.session_state.result
    id_col, text_col, _active_domain = _page_state(id_col, text_col, domain)
    total = len(out)

    _kw_dict     = _get_kw_dict(_active_domain)
    text_src_col = "Text_For_Analysis" if "Text_For_Analysis" in out.columns else text_col
    text_series  = out[text_src_col].fillna("")

    cat_summary = _build_cat_summary(
        tuple(text_series.tolist()),
        tuple((cat, tuple(kws)) for cat, kws in _kw_dict.items()),
        total,
    )

    if not cat_summary:
        st.info("No negative keywords found."); st.stop()

    # ── Category overview bar chart ───────────────────────────────────────────
    ov_labels  = [cs["label"] for cs in cat_summary]
    ov_counts  = [cs["records"] for cs in cat_summary]
    ov_pcts    = [cs["pct"] for cs in cat_summary]
    # colour gradient: darker orange for higher counts
    _max_ov = max(ov_counts) if ov_counts else 1
    ov_colors  = [
        f"rgba({int(168 + 40 * (c / _max_ov))},{int(90 + 30 * (1 - c / _max_ov))},{int(30 + 20 * (1 - c / _max_ov))},0.85)"
        for c in ov_counts
    ]
    fig_ov = go.Figure(go.Bar(
        y=ov_labels, x=ov_counts, orientation="h",
        marker=dict(color=ov_colors, cornerradius=5, line=dict(width=0)),
        text=[f"{p}%  ({v:,})" for p, v in zip(ov_pcts, ov_counts)],
        textposition="outside",
        textfont=dict(size=11, color="#2D5F6E", family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>%{x:,} records (%{text})<extra></extra>",
    ))
    _mfig(fig_ov, max(240, len(ov_labels) * 32)).update_layout(
        yaxis={"categoryorder": "total ascending"},
        title=dict(text="Issue Categories — Record Count", font=dict(size=14, color="#1E2D33"), x=0),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,.15)"),
    )
    st.plotly_chart(fig_ov, width='stretch', key="kw_overview")

    st.divider()

    # ── Drill into selected category ──────────────────────────────────────────
    sel_label = st.selectbox("Select issue category to drill in",
                             [cs["label"] for cs in cat_summary], key="kw_cat_sel")
    cs_data   = next(c for c in cat_summary if c["label"] == sel_label)
    _mask_arr = np.array(cs_data["mask"], dtype=bool)
    cat_out   = out.loc[out.index[_mask_arr]]

    neg_in_cat  = int(cat_out["consumer_sentiment"].isin(["Negative", "Very Negative"]).sum())
    pos_in_cat  = int(cat_out["consumer_sentiment"].isin(["Positive", "Very Positive"]).sum())
    neu_in_cat  = int((cat_out["consumer_sentiment"] == "Neutral").sum())
    cat_total   = int(cs_data["records"])
    neg_pct_cat = round(neg_in_cat / cat_total * 100, 1) if cat_total else 0

    col_kw, col_donut, col_meta = st.columns([3, 2, 2])

    with col_kw:
        cat_keywords = set(_kw_dict.get(cs_data["cat"], []))
        kw_s = _explode_keywords(cat_out["Negative_Keywords"])
        if cat_keywords:
            kw_s = kw_s[kw_s.isin(cat_keywords)]
        if kw_s.empty:
            kw_s = _explode_keywords(cat_out["Negative_Keywords"])

        if not kw_s.empty:
            kw_counts = kw_s.value_counts().head(15).reset_index()
            kw_counts.columns = ["Keyword", "Count"]
            _kmax = kw_counts["Count"].max()
            kw_bar_colors = [
                f"rgba({int(160 + 48 * (v / _kmax))},{int(80 + 42 * (1 - v / _kmax))},40,0.85)"
                for v in kw_counts["Count"]
            ]
            fig_kw = go.Figure(go.Bar(
                y=kw_counts["Keyword"], x=kw_counts["Count"], orientation="h",
                marker=dict(color=kw_bar_colors, cornerradius=4, line=dict(width=0)),
                text=[f"{v:,}" for v in kw_counts["Count"]], textposition="outside",
                textfont=dict(size=11, color="#2D5F6E"),
                hovertemplate="<b>%{y}</b><br>%{x:,} hits<extra></extra>",
            ))
            _mfig(fig_kw, max(300, len(kw_counts) * 28)).update_layout(
                yaxis={"categoryorder": "total ascending"},
                title=dict(text=f"Top keywords — {sel_label}", font=dict(size=13, color="#1E2D33"), x=0),
                showlegend=False,
            )
            st.plotly_chart(fig_kw, width='stretch', key="kw_drill")
        else:
            st.caption("No keyword breakdown available for this category.")

    with col_donut:
        fig_cd = go.Figure(go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos_in_cat, neu_in_cat, neg_in_cat],
            marker=dict(colors=["#3D7A5F", "#6B8A99", "#A04040"], line=dict(width=0)),
            hole=0.60,
            textinfo="percent",
            textfont=dict(size=11, family="DM Sans"),
            hovertemplate="<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
        ))
        _mfig(fig_cd, 280).update_layout(
            title=dict(text="Sentiment mix in category", font=dict(size=13, color="#1E2D33"), x=0),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
            margin=dict(l=10, r=10, t=40, b=30),
        )
        st.plotly_chart(fig_cd, width='stretch', key="kw_cd")

    with col_meta:
        _neg_color = "#A04040" if neg_pct_cat >= 50 else "#C87A40" if neg_pct_cat >= 25 else "#6B8A99"
        st.markdown(
            f'<div class="detail-card">'
            f'<div class="detail-label">Impacted Records</div>'
            f'<div class="mv" style="color:#C87A40;font-size:28px;margin-bottom:2px">{cat_total:,}</div>'
            f'<div style="font-size:12px;color:var(--muted);margin-bottom:14px">{cs_data["pct"]}% of dataset</div>'
            f'<div class="detail-label">Negative Sentiment</div>'
            f'<div style="font-size:20px;font-weight:700;color:{_neg_color}">{neg_pct_cat}%</div>'
            f'<div style="font-size:12px;color:var(--muted);margin-bottom:14px">{neg_in_cat:,} records</div>'
            f'<div class="detail-label">Positive Sentiment</div>'
            f'<div style="font-size:16px;font-weight:600;color:#3D7A5F">'
            f'{round(pos_in_cat / cat_total * 100, 1) if cat_total else 0}%</div>'
            f'<div style="font-size:12px;color:var(--muted)">{pos_in_cat:,} records</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        samples = cat_out[text_src_col].dropna().astype(str).head(3).tolist()
        if samples:
            cat_kw_set = set(_kw_dict.get(cs_data["cat"], []))
            st.markdown(
                '<div style="margin-top:14px;font-size:10px;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.8px;color:var(--muted)">Sample Comments</div>',
                unsafe_allow_html=True,
            )
            for s in samples:
                display = _redact(s) if redact_pii else s
                short   = display[:220] + "…" if len(display) > 220 else display
                # bold-highlight matched keywords
                if cat_kw_set:
                    _hl_pat = re.compile(
                        "|".join(re.escape(k) for k in sorted(cat_kw_set, key=len, reverse=True)),
                        re.IGNORECASE,
                    )
                    highlighted = _hl_pat.sub(
                        lambda m: f'<strong style="color:#A04040">{_html.escape(m.group())}</strong>',
                        _html.escape(short),
                    )
                else:
                    highlighted = _html.escape(short)
                st.markdown(
                    f'<div style="background:var(--warm-l);border-radius:6px;padding:9px 12px;'
                    f'margin-top:7px;font-size:12px;color:var(--text2);line-height:1.6">'
                    f'{highlighted}</div>',
                    unsafe_allow_html=True,
                )



# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Audit Trail":

    if "result" not in st.session_state:
        st.info("Run analysis first."); st.stop()

    out                        = st.session_state.result
    id_col, text_col, _domain = _page_state(id_col, text_col, domain)
    total    = len(out)

    # ── Triage cards ──────────────────────────────────────────────────────────
    triage_defs = [
        ("Validation", "Validation Overrides", "#3D7A5F",
         lambda d: d[d["validation_source"] == "Validation"]),
        ("Rule",       "Rule-fired",            "#2D5F6E",
         lambda d: d[d["_rule_fired"].str.startswith("rule:", na=False)] if "_rule_fired" in d.columns else d.iloc[0:0]),
        ("VADER",      "VADER",                 "#6B8A99",
         lambda d: d[d["_rule_fired"] == "vader"] if "_rule_fired" in d.columns else d.iloc[0:0]),
        ("BERT",       "BERT Corrections",      "#D4B94E",
         lambda d: d[d["_rule_fired"].str.startswith("bert:", na=False)] if "_rule_fired" in d.columns else d.iloc[0:0]),
        ("LowConf",    "Low Confidence",        "#C87A40",
         lambda d: d[(d["confidence"] < 0.3) & (d["validation_source"] == "Model")]),
        ("Blank",      "Blank / Unscored",      "#A8BCC8",
         lambda d: d[d["_rule_fired"].isin(["blank", "non_english"])] if "_rule_fired" in d.columns else d.iloc[0:0]),
    ]

    # Compute counts once
    triage_counts = {}
    for k, lbl, col, fn in triage_defs:
        try:
            triage_counts[k] = len(fn(out))
        except Exception:
            triage_counts[k] = 0

    active_triage = [(k, lbl, col, fn) for k, lbl, col, fn in triage_defs if triage_counts[k] > 0]

    triage_cols = st.columns(len(active_triage))
    for i, (k, lbl, col, fn) in enumerate(active_triage):
        cnt = triage_counts[k]
        with triage_cols[i]:
            st.markdown(mc_anim(lbl, f"{cnt:,}", col, delay=i * 0.05), unsafe_allow_html=True)
            st.caption(f"{round(cnt / total * 100, 1)}%")

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
    _bucket_opts = [lbl for _, lbl, _, _ in active_triage]
    f_bucket = st.pills("Triage bucket", _bucket_opts, selection_mode="single", default=None, key="at_bucket")
    fc2, fc3 = st.columns(2)
    with fc2:
        f_sent = st.pills("Sentiment", SENTIMENT_ORDER, selection_mode="single", default=None, key="at_sent")
    with fc3:
        f_conf = st.pills("Confidence", ["High (>0.7)", "Medium (0.3–0.7)", "Low (<0.3)"], selection_mode="single", default=None, key="at_conf")

    ad = out
    if f_bucket:
        _, _, _, bucket_fn = next((t for t in active_triage if t[1] == f_bucket), (None, None, None, None))
        if bucket_fn:
            try:
                ad = bucket_fn(ad)
            except Exception:
                pass
    if f_sent:   ad = ad[ad["consumer_sentiment"] == f_sent]
    if f_conf:
        if "High"   in f_conf: ad = ad[ad["confidence"] > 0.7]
        elif "Medium" in f_conf: ad = ad[(ad["confidence"] >= 0.3) & (ad["confidence"] <= 0.7)]
        elif "Low"    in f_conf: ad = ad[ad["confidence"] < 0.3]

    # ── Selectable compact table ───────────────────────────────────────────────
    show_cols = [c for c in [id_col, "consumer_sentiment", "consumer_score", "confidence", "_rule_fired"] if c in ad.columns]
    at_df     = ad[show_cols].reset_index(drop=True)
    if "consumer_sentiment" in at_df.columns:
        at_df = at_df.assign(consumer_sentiment=at_df["consumer_sentiment"].map(_SENT_PILL).fillna(at_df["consumer_sentiment"]))

    st.caption(f"{len(at_df):,} records — click a row to see lineage")
    at_sel = st.dataframe(
        at_df,
        on_select="rerun",
        selection_mode="single-row",
        width='stretch',
        height=360,
        key="at_tbl_sel",
        column_config={
            "consumer_sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
            "consumer_score":     st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "confidence":         st.column_config.NumberColumn("Conf", format="%.2f", width="small"),
            "_rule_fired":        st.column_config.TextColumn("Method", width="medium"),
        },
    )

    # ── Record lineage panel ───────────────────────────────────────────────────
    selected = at_sel.selection.rows if hasattr(at_sel, "selection") else []
    if selected:
        orig_idx = ad.index[selected[0]]
        row      = out.loc[orig_idx]

        st.divider()
        st.markdown("**Record lineage**")

        steps = []
        raw_text = row.get(text_col, "")
        if isinstance(raw_text, str) and raw_text.strip():
            display = _redact(raw_text[:300]) if redact_pii else raw_text[:300]
            steps.append(("Source text", display + ("…" if len(str(raw_text)) > 300 else ""), "#6B8A99"))

        for ecol in ["CustomerOnly", "ConsumerOnly", "Text_For_Analysis"]:
            if ecol in row.index and isinstance(row[ecol], str) and row[ecol].strip() and row[ecol] != raw_text:
                display = _redact(row[ecol][:300]) if redact_pii else row[ecol][:300]
                steps.append((ecol.replace("_", " "), display + ("…" if len(row[ecol]) > 300 else ""), "#3A7A8C"))
                break

        steps.append(("Classification method", str(row.get("_rule_fired", "—")), "#D4B94E"))

        raw_vader = row.get("_raw_vader", None)
        if raw_vader is not None and not pd.isna(raw_vader):
            steps.append(("VADER raw score", f"{float(raw_vader):+.4f}", "#6B8A99"))

        score = row.get("consumer_score", 0) or 0
        conf  = row.get("confidence", 0) or 0
        sent  = str(row.get("consumer_sentiment", "—"))
        steps.append(("Final score", f"{float(score):+.4f}", "#2D5F6E"))
        steps.append(("Sentiment · Confidence", f"{sent} · {float(conf):.2f}",
                       SENTIMENT_COLORS.get(sent, "#6B8A99")))

        if row.get("validation_source") == "Validation":
            steps.append(("Override", "Validation file — model output replaced", "#3D7A5F"))

        html_parts = ['<div style="display:flex;flex-direction:column;gap:0">']
        for i, (label, value, color) in enumerate(steps):
            connector = (
                f'<div style="width:2px;height:10px;background:{color};opacity:.35;margin-left:10px"></div>'
                if i < len(steps) - 1 else ""
            )
            html_parts.append(
                f'<div style="display:flex;align-items:flex-start;gap:10px">'
                f'<div style="width:20px;height:20px;border-radius:50%;background:{color};flex-shrink:0;'
                f'display:flex;align-items:center;justify-content:center;font-size:9px;color:#fff;font-weight:700">'
                f'{i+1}</div>'
                f'<div style="flex:1;padding-bottom:4px">'
                f'<div class="detail-label">{_html.escape(label)}</div>'
                f'<div class="detail-value" style="white-space:pre-wrap;word-break:break-word">'
                f'{_html.escape(value)}</div></div></div>{connector}'
            )
        html_parts.append('</div>')
        st.markdown("".join(html_parts), unsafe_allow_html=True)

        kw = row.get("Negative_Keywords", "")
        if isinstance(kw, str) and kw.strip():
            st.markdown(
                f'<div style="margin-top:10px;padding:8px 12px;background:var(--warm-l);'
                f'border-radius:6px;font-size:12px">'
                f'<span class="detail-label">Negative keywords: </span>'
                f'<span style="color:var(--text2)">{_html.escape(kw)}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()
    _csv_download(ad, "Export filtered CSV", _domain, key="at_dl", suffix="audit")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 12px;color:var(--muted);font-size:11px">
  Sentix · Multi-Domain Sentiment Intelligence · Streamlit
</div>
""", unsafe_allow_html=True)
