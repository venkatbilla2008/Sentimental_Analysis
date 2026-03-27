"""
SentimentHub — PPT Vader Adaptive
Self-contained Streamlit app with all PPT logic embedded.
Run: streamlit run app.py
"""

import html as _html_mod
import io
import re
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Optional multilingual / hybrid-model packages (Hilton domain) ─────────────
_AFINN_OK = _TEXTBLOB_OK = _LANGDETECT_OK = _TRANSLATOR_OK = False
try:
    from afinn import Afinn as _Afinn
    _af = _Afinn()
    _AFINN_OK = True
except ImportError:
    pass

try:
    from textblob import TextBlob as _TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    pass

try:
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    _LANGDETECT_OK = True
except ImportError:
    pass

try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _TRANSLATOR_OK = True
except ImportError:
    pass

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
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentHub — PPT",
    layout="wide",
    page_icon="💬",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (teal / slate / gold palette)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --teal:#2D5F6E; --teal-l:#3A7A8C; --slate:#6B8A99; --steel:#A8BCC8;
  --warm:#D1CFC4; --warm-l:#E8E6DD; --gold:#D4B94E;
  --bg:#F5F4F0; --card:#FFFFFF; --border:#D1CFC4;
  --text:#1E2D33; --text2:#3D5A66; --muted:#6B8A99;
  --success:#3D7A5F; --warn:#B8963E; --err:#A04040;
}
.stApp { font-family:'DM Sans',sans-serif; background:var(--bg); }
.stApp h1,.stApp h2,.stApp h3 { font-weight:600; color:var(--text); }

/* Sidebar */
section[data-testid="stSidebar"] { background:var(--warm-l)!important; border-right:1px solid var(--warm)!important; }
section[data-testid="stSidebar"] * { color:var(--text)!important; }
section[data-testid="stSidebar"] .stButton>button { justify-content:flex-start!important; padding-left:16px!important; }
section[data-testid="stSidebar"] .stButton>button[kind="primary"] { background:var(--teal)!important; color:#E8D97A!important; font-weight:600!important; }
section[data-testid="stSidebar"] .stButton>button[kind="secondary"] { background:transparent!important; border-color:var(--warm)!important; }
section[data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover { background:var(--warm-l)!important; border-color:var(--teal)!important; }

/* Metric cards */
.mc { background:var(--card); border:1px solid var(--border); border-radius:10px;
      padding:18px 16px; text-align:center; border-top:3px solid var(--teal);
      box-shadow:0 1px 4px rgba(45,95,110,.06); transition:all .2s; }
.mc:hover { box-shadow:0 4px 16px rgba(45,95,110,.1); transform:translateY(-1px); }
.mv { font-size:22px; font-weight:700; color:var(--text); margin:0; }
.ml { font-size:10px; font-weight:600; color:var(--muted); margin:5px 0 0;
      text-transform:uppercase; letter-spacing:.7px; }

/* Section headers */
.sh { display:flex; align-items:center; gap:8px; margin:24px 0 12px;
      font-size:15px; font-weight:600; color:var(--text);
      padding-bottom:8px; border-bottom:2px solid var(--warm); }

/* Badges */
.badge { display:inline-flex; align-items:center; gap:4px; padding:4px 12px;
         border-radius:5px; font-size:12px; font-weight:600; }
.b-ok   { background:#D4E8DC; color:var(--success); }
.b-warn { background:#F0E6C8; color:#7A6620; }
.b-info { background:#D6E8EE; color:var(--teal); }
.b-err  { background:#F2D6D6; color:var(--err); }

/* Sentiment row chips */
.chip-vp  { background:#D4E8DC; color:#2A5C40; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.chip-pos { background:#E8F4D4; color:#3D6B20; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.chip-neu { background:#E8E6DD; color:#5A5A5A; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.chip-neg { background:#F2D6D6; color:#7A2020; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }
.chip-vn  { background:#EEC4C4; color:#5A1010; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600; }

/* Buttons */
.stButton>button[kind="primary"] { background:var(--teal)!important; border-color:var(--teal)!important; color:#FFF!important; }
.stButton>button[kind="primary"]:hover { background:var(--teal-l)!important; }
.stProgress>div>div>div { background:var(--teal)!important; }
footer, .stDeployButton { display:none!important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════  PPT LOGIC (from notebook)  ══════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── VADER ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ── PT Content (domain terms to strip) ───────────────────────────────────────
PT_CONTENT_SET = {
    "professional physical therapy", "professional pt", "physical therapy",
    "pt session", "therapy session", "treatment session",
}

REMOVE_PATTERNS = [
    r"chat with an agent", r"chat with agent", r"speak to agent",
    r"talk to agent", r"connect me to agent", r"agent chat",
    r"chat to agent", r"agent help", r"help chat",
]
REMOVE_PATTERN = re.compile("|".join(REMOVE_PATTERNS), re.IGNORECASE)


# ── Speaker extraction ────────────────────────────────────────────────────────
def extract_customer_messages(text):
    """Extract customer-only messages from HTML or SMS transcript formats."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # ---- HTML format: <b>HH:MM:SS Speaker:</b> message <br/>
    html_pattern = r'<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)'
    html_matches = re.findall(html_pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if html_matches:
        speaker_messages = {}
        for timestamp, speaker, message in html_matches:
            spk = speaker.strip().lower()
            if spk == "system":
                continue
            speaker_messages.setdefault(spk, []).append(message.strip())

        if not speaker_messages:
            return ""

        speaker_counts = {s: len(m) for s, m in speaker_messages.items()}
        if len(speaker_counts) == 1:
            customer_name = list(speaker_counts.keys())[0]
        else:
            sorted_speakers = sorted(speaker_counts.items(), key=lambda x: x[1])
            first_speaker = None
            for _, spk, _ in html_matches:
                if spk.strip().lower() != "system":
                    first_speaker = spk.strip().lower()
                    break
            if first_speaker and speaker_counts[first_speaker] <= sorted_speakers[0][1]:
                customer_name = first_speaker
            else:
                customer_name = sorted_speakers[0][0]

        msgs = speaker_messages[customer_name]
        return " ".join(re.sub(r"\s+", " ", m).strip() for m in msgs if m.strip())

    # ---- SMS format: HH:MM:SS Speaker: message
    sms_pattern = r'(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)'
    sms_matches = re.findall(sms_pattern, text, flags=re.DOTALL)

    if sms_matches:
        speaker_messages = {}
        for timestamp, speaker, message in sms_matches:
            spk = speaker.strip().lower()
            if spk == "system":
                continue
            msg = re.sub(r'\d{4}-\d{2}-\d{2}T[\d:.]+Z\w+$', '', message)
            msg = re.sub(r'Looks up Phone Number.*?digits-\d+', '', msg)
            msg = re.sub(r'Looks up SSN number.*?digits-\d+', '', msg)
            msg = re.sub(r'Phone Numbers rule for Chat', '', msg)
            msg = re.sub(r'SSN rule for Chat', '', msg)
            msg = msg.strip()
            if msg:
                speaker_messages.setdefault(spk, []).append(msg)

        if not speaker_messages:
            return ""

        phone_spks = [s for s in speaker_messages if re.match(r'^\d+$', s)]
        if phone_spks:
            customer_name = phone_spks[0]
        elif len(speaker_messages) == 1:
            customer_name = list(speaker_messages.keys())[0]
        else:
            speaker_counts = {s: len(m) for s, m in speaker_messages.items()}
            customer_name = sorted(speaker_counts.items(), key=lambda x: x[1])[0][0]

        msgs = speaker_messages[customer_name]
        return " ".join(re.sub(r"\s+", " ", m).strip() for m in msgs if m.strip())

    return ""


# ── Text cleaning ─────────────────────────────────────────────────────────────
def aggressive_clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = REMOVE_PATTERN.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 5 or len(text.split()) < 2:
        return ""
    return text


def remove_pt_content_names(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text_lower = text.lower()
    cleaned = text
    for content in PT_CONTENT_SET:
        if content in text_lower:
            cleaned = re.compile(re.escape(content), re.IGNORECASE).sub("", cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ── Negative keyword dictionary ───────────────────────────────────────────────
NEGATIVE_KEYWORDS = {
    "payment_issues": [
        "money is debited", "money was debited", "money was withdrawn",
        "charged double", "being charged double", "double charged",
        "unauthorized charge", "charged without permission",
        "payment not going through", "unable to do payment",
        "payment is pending", "payment failed", "payment issue",
        "billing issue", "billing problem", "wrong amount charged",
        "overcharged", "charged incorrectly", "incorrect charge",
        "refund not received", "still waiting for refund",
    ],
    "access_issues": [
        "cannot access", "cant access", "can't access", "unable to access",
        "not working properly", "doesn't work", "does not work",
        "not working", "stopped working", "no longer works",
        "hacked my account", "account hacked", "compromised account",
        "locked out", "account locked", "suspended account",
        "login issue", "login problem", "can't login", "cannot login",
        "portal not working", "website down", "app not working",
    ],
    "delivery_issues": [
        "still have not received", "have not received", "not received",
        "did not receive", "never received", "haven't received",
        "waiting for", "still waiting", "no response",
        "not delivered", "delivery failed", "never got",
    ],
    "unwanted_actions": [
        "without my permission", "without permission", "without consent",
        "i do not want this", "did not want to renew",
        "did not want to restart", "didn't want to renew",
        "auto renewed", "automatically renewed", "renewed without asking",
        "charged after cancellation", "cancelled but charged",
        "didn't authorize", "unauthorized",
    ],
    "technical_issues": [
        "notifications are getting stuck", "getting stuck", "stuck",
        "error message", "error occurred", "system error",
        "technical issue", "technical problem", "glitch",
        "buffering issue", "streaming problem", "won't load",
        "keeps crashing", "app crashes", "freezing",
        "not loading", "page not loading",
    ],
    "frustration": [
        "frustrating", "frustrated", "annoyed", "angry",
        "disappointed", "upset", "irritated", "furious",
        "unacceptable", "ridiculous", "terrible", "horrible",
        "worst", "pathetic", "useless", "awful",
        "disgusted", "fed up", "sick of", "tired of",
        "not happy", "unhappy", "dissatisfied",
    ],
    "poor_service": [
        "thanks for nothing", "no help", "not helpful", "unhelpful",
        "you dont want to help", "you don't want to help",
        "waste of time", "wasting my time", "poor service",
        "bad service", "terrible service", "worst service",
        "no support", "poor support", "terrible support",
        "unprofessional", "rude", "disrespectful",
    ],
    "resolution_issues": [
        "not resolved", "still not resolved", "unresolved",
        "no solution", "not fixed", "still broken",
        "problem persists", "issue continues", "still having issues",
        "unforeseen circumstances", "unexpected problem",
        "still experiencing", "ongoing issue",
    ],
    "appointment_issues": [
        "missed appointment", "late appointment", "wrong appointment",
        "appointment not scheduled", "double booked", "no appointment available",
        "long wait time", "waiting too long", "waited forever",
        "no show", "didn't show up", "never showed",
        "cancelled my appointment", "appointment was cancelled",
        "rescheduled without notice", "changed without telling me",
        "therapist didn't show", "no therapist available",
        "running late", "always late", "never on time",
    ],
    "treatment_issues": [
        "treatment didn't work", "no improvement", "getting worse",
        "pain increased", "not getting better", "ineffective treatment",
        "wrong diagnosis", "misdiagnosed", "incorrect treatment",
        "not helping", "doesn't help", "no progress",
        "condition worsened", "more pain", "hurts more",
    ],
    "communication_issues": [
        "no response", "didn't respond", "no callback", "never called back",
        "ignored my request", "no follow up", "poor communication",
        "never heard back", "no one contacted me",
        "didn't call me back", "no one called", "unreachable",
        "can't reach anyone", "no answer", "not responding",
    ],
    "staff_issues": [
        "rude staff", "unprofessional staff", "incompetent",
        "didn't listen", "ignored me", "dismissive",
        "not trained", "inexperienced", "careless",
        "rushed through", "didn't care", "indifferent",
    ],
    "billing_insurance": [
        "insurance not accepted", "insurance issue", "insurance problem",
        "claim denied", "not covered", "out of network",
        "surprise bill", "unexpected charge", "hidden fee",
        "wrong insurance", "billing error",
    ],
    "general_negative": [
        "problem", "issue", "concern", "complaint",
        "trouble", "difficulty", "struggle",
        "wrong", "incorrect", "mistake", "error",
        "broken", "damaged", "defective",
        "lost", "missing", "disappeared",
        "failed", "failure", "unsuccessful",
    ],
}

ALL_NEGATIVE_KEYWORDS_SORTED = sorted(
    [kw for kws in NEGATIVE_KEYWORDS.values() for kw in kws],
    key=len, reverse=True
)
negative_keyword_pattern = re.compile(
    "|".join(re.escape(k) for k in ALL_NEGATIVE_KEYWORDS_SORTED),
    re.IGNORECASE
)


# ── Sentiment trigger patterns ────────────────────────────────────────────────
VERY_NEGATIVE_TRIGGERS = [
    "worst experience", "absolutely terrible", "completely unacceptable",
    "never coming back", "will never use again", "horrible experience",
    "disgusted", "furious", "outraged",
    "scam", "fraud", "stealing my money",
    "sue", "lawyer", "legal action",
    "file a complaint", "report to", "better business bureau",
]
NEGATIVE_TRIGGERS = [
    "still have not received", "have not received", "not received",
    "money is debited", "money was debited", "money was withdrawn",
    "without my permission", "without permission", "without consent",
    "i do not want this", "did not want to renew",
    "thanks for nothing",
    "getting stuck", "notifications are getting stuck",
    "unable to do payment", "payment is pending", "payment failed",
    "frustrating", "frustrated", "annoyed", "angry", "upset",
    "disappointed", "terrible", "horrible", "worst",
    "unacceptable", "ridiculous", "pathetic", "useless",
    "not working properly", "doesn't work", "cant access", "cannot access",
    "not working", "stopped working",
    "hacked my account", "compromised account",
    "charged double", "overcharged", "incorrect charge",
    "unforeseen circumstances",
    "you dont want to help", "you don't want to help",
    "no help", "not helpful", "unhelpful",
    "waste of time", "wasting my time",
    "long wait", "waiting too long", "waited forever",
    "no improvement", "getting worse", "pain increased",
    "treatment didn't work", "not getting better",
    "no show", "didn't show up", "never showed",
    "rude", "unprofessional", "dismissive",
    "no response", "never called back", "ignored",
    "not resolved", "still not resolved", "problem persists",
]
NEUTRAL_TRIGGERS = [
    "cancel appointment", "reschedule", "change appointment",
    "need to cancel", "please cancel", "want to cancel",
    "want to reschedule", "need to reschedule",
    "move my appointment", "change my appointment time",
    "change email", "change my email", "update email",
    "change the email address", "need to change my email",
    "change my card", "change card details", "update payment",
    "remove payment method", "update my information",
    "verify my account", "need to update",
    "want my refund", "please refund", "need a refund",
    "issue a refund", "can you refund", "requesting refund",
    "how to", "how can i", "how do i",
    "can you help", "need help", "help me",
    "can i get", "is there any", "is it possible",
    "questions about", "just wondering",
    "what is", "where can i", "when will",
    "change plan", "switch plan", "upgrade", "downgrade",
    "change from", "switch to",
    "just canceled", "just cancelled",
    "finding current password", "reset password",
    "need to add", "want to add",
]
POSITIVE_TRIGGERS = [
    "thank you", "thanks", "thank u", "ty",
    "appreciate", "grateful", "thankful",
    "perfect", "great", "excellent", "awesome",
    "helpful", "very helpful", "super helpful",
    "wonderful", "fantastic", "amazing",
    "thanks for the quick reply", "thanks for your help",
    "appreciate you looking into", "appreciate your help",
    "thanks for clarifying", "thanks for explaining",
    "that works", "sounds good", "that's fine",
    "all set", "all good", "we're good",
]
VERY_POSITIVE_TRIGGERS = [
    "outstanding", "exceeded expectations",
    "extremely helpful", "incredibly helpful",
    "absolutely perfect", "best service",
    "highly recommend", "couldn't be better",
    "amazing service", "excellent service",
    "love it", "love this", "absolutely love",
]

# Compile all trigger patterns once
very_neg_pattern  = re.compile("|".join(re.escape(p) for p in VERY_NEGATIVE_TRIGGERS),  re.IGNORECASE)
neg_pattern       = re.compile("|".join(re.escape(p) for p in NEGATIVE_TRIGGERS),        re.IGNORECASE)
neutral_pattern   = re.compile("|".join(re.escape(p) for p in NEUTRAL_TRIGGERS),          re.IGNORECASE)
positive_pattern  = re.compile("|".join(re.escape(p) for p in POSITIVE_TRIGGERS),         re.IGNORECASE)
very_pos_pattern  = re.compile("|".join(re.escape(p) for p in VERY_POSITIVE_TRIGGERS),    re.IGNORECASE)

STRONG_NEGATIVE_PHRASES = [
    "without my permission", "without permission",
    "do not want this", "did not want",
    "thanks for nothing",
    "unable to do payment", "payment failed",
    "still have not received", "never received",
    "long wait", "waiting too long", "waited forever",
    "no improvement", "getting worse", "pain increased",
    "treatment didn't work",
    "no show", "didn't show up",
    "rude", "unprofessional", "dismissive",
    "waste of time", "wasting my time",
    "not helpful", "no help",
    "not resolved", "problem persists",
]


# ── Context helpers ───────────────────────────────────────────────────────────
def has_resolution_or_thanks(text):
    tl = text.lower()
    indicators = ["thanks","thank you","thank u","ty","okay","ok","ok thanks","perfect",
                  "got it","understood","appreciate","that will be all","sounds good",
                  "that works","all set","all good","we're good"]
    return any(i in tl for i in indicators)

def is_polite_request(text):
    tl = text.lower()
    polite = ["please","can you","could you","would you","i want to","i need to",
              "i would like","help me","can i","may i","how to","how can","how do i"]
    frustration = ["frustrated","angry","annoyed","furious","terrible","horrible",
                   "worst","pathetic","useless","awful","unacceptable","ridiculous"]
    return any(p in tl for p in polite) and not any(f in tl for f in frustration)

def has_simple_cancellation(text):
    tl = text.lower()
    simple = ["cancel my appointment","cancel appointment","need to cancel",
              "want to cancel","please cancel","can i cancel","reschedule"]
    complaint = ["frustrated","angry","terrible","horrible","long wait",
                 "no show","late","problem","issue","complaint"]
    return any(s in tl for s in simple) and not any(c in tl for c in complaint)


# ── Rule-based classification (direct from notebook) ─────────────────────────
def classify_by_rules(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0

    tl = text.lower()

    if very_neg_pattern.search(text):
        return -0.85, 0.95

    if very_pos_pattern.search(text):
        return 0.85, 0.95

    if positive_pattern.search(text):
        pos_count = len(re.findall(
            r'\b(thank|thanks|appreciate|perfect|helpful|great|excellent|awesome)\b', tl))
        return (0.75, 0.90) if pos_count >= 2 else (0.35, 0.80)

    if neg_pattern.search(text):
        if has_resolution_or_thanks(text):
            neg_matches = len(neg_pattern.findall(text))
            if neg_matches == 1 and any(w in tl[-100:] for w in ["thank","thanks","appreciate"]):
                return 0.0, 0.75
        if any(phrase in tl for phrase in STRONG_NEGATIVE_PHRASES):
            return -0.70, 0.95
        return -0.55, 0.80

    if neutral_pattern.search(text):
        if has_simple_cancellation(text):
            return 0.0, 0.90
        if is_polite_request(text):
            return 0.0, 0.90
        if "refund" in tl and any(p in tl for p in ["please","want","can you","need","request"]):
            return 0.0, 0.85
        return 0.0, 0.80

    return None, 0.0


# ── Sentiment label from score ────────────────────────────────────────────────
def classify_sentiment(score):
    if pd.isna(score):
        return "Neutral"
    if score >= 0.60:   return "Very Positive"
    if score >= 0.20:   return "Positive"
    if score >= -0.20:  return "Neutral"
    if score >= -0.60:  return "Negative"
    return "Very Negative"


# ── VADER chunked scoring ─────────────────────────────────────────────────────
def get_vader_compound(text, analyzer):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= 300:
            cur += " " + s
        else:
            if cur: chunks.append(cur.strip())
            cur = s
    if cur: chunks.append(cur.strip())
    if not chunks: return 0.0
    return float(np.mean([analyzer.polarity_scores(c)['compound'] for c in chunks]))


# ── Negative keyword extraction ───────────────────────────────────────────────
def extract_negative_keywords(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    matches = negative_keyword_pattern.findall(text)
    if not matches:
        return ""
    seen, found = set(), []
    for m in matches:
        ml = m.lower()
        if ml not in seen:
            seen.add(ml)
            found.append(m)
    return "; ".join(found)


# ── Validation data loader ────────────────────────────────────────────────────
def load_validation_data(uploaded_file):
    validation_dict = {}
    if uploaded_file is None:
        return validation_dict
    try:
        df_val = pd.read_excel(io.BytesIO(uploaded_file.read()))
        id_col = None
        for col in df_val.columns:
            if col.strip().upper() in ['ID', 'PPTLEADS_COMM_ID', 'TICKET ID', 'TICKET_ID']:
                id_col = col
                break
        sent_col = None
        for col in df_val.columns:
            if col.lower().strip() == 'actual sentiment':
                sent_col = col
                break
        if not id_col or not sent_col:
            st.warning(f"⚠️ Validation file columns not matched. Found: {df_val.columns.tolist()}")
            return validation_dict
        for _, row in df_val.iterrows():
            tid = row.get(id_col)
            snt = row.get(sent_col)
            if pd.isna(tid) or pd.isna(snt) or not str(snt).strip():
                continue
            validation_dict[str(tid).strip().replace(" ", "")] = str(snt).strip()
        return validation_dict
    except Exception as e:
        st.warning(f"Could not load validation file: {e}")
        return {}


# ── Core processing function ──────────────────────────────────────────────────
SENTIMENT_TO_SCORE = {
    "Very Positive": 0.85, "Positive": 0.35,
    "Neutral": 0.0, "Negative": -0.55, "Very Negative": -0.85,
}

def run_ppt_analysis(df, id_col, text_col, validation_dict, progress_cb=None):
    analyzer = load_vader()
    df = df.copy()

    # Normalize IDs
    df["_id_norm"] = df[id_col].astype(str).str.strip().str.replace(" ", "")

    # Extract + clean customer text
    df["CustomerOnly"] = df[text_col].apply(extract_customer_messages)
    df["CustomerOnly_Cleaned"] = df["CustomerOnly"].apply(aggressive_clean_text)
    df["CustomerOnly_Cleaned"] = df["CustomerOnly_Cleaned"].apply(remove_pt_content_names)
    df["Text_For_Analysis"] = df["CustomerOnly_Cleaned"]

    n = len(df)
    compounds, confidences, sentiments, sources = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_cb and i % max(1, n // 100) == 0:
            progress_cb(i, n)

        norm_id = row["_id_norm"]
        text = row["Text_For_Analysis"]

        # Priority 1 — validation override
        if norm_id in validation_dict:
            label = validation_dict[norm_id]
            score = SENTIMENT_TO_SCORE.get(label, 0.0)
            compounds.append(score)
            confidences.append(1.0)
            sentiments.append(label)
            sources.append("Validation")
            continue

        sources.append("Model")

        # Empty text
        if not isinstance(text, str) or not text.strip():
            compounds.append(0.0)
            confidences.append(0.0)
            sentiments.append("Neutral")
            continue

        # Priority 2 — rule-based
        rule_score, confidence = classify_by_rules(text)

        if rule_score is not None and confidence > 0.7:
            compounds.append(round(rule_score, 3))
            confidences.append(confidence)
            sentiments.append(classify_sentiment(rule_score))
        else:
            # Priority 3 — VADER
            vader_score = get_vader_compound(text, analyzer)
            if abs(vader_score) < 0.05:
                vader_score = 0.0
            if rule_score is not None:
                final = (rule_score * confidence) + (vader_score * (1 - confidence))
            else:
                final = vader_score
            compounds.append(round(final, 3))
            confidences.append(0.5)
            sentiments.append(classify_sentiment(final))

    if progress_cb:
        progress_cb(n, n)

    df["consumer_score"]     = compounds
    df["confidence"]         = confidences
    df["consumer_sentiment"] = sentiments
    df["validation_source"]  = sources

    # Negative keyword extraction
    df["Negative_Keywords"] = df["Text_For_Analysis"].apply(extract_negative_keywords)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════  HILTON (TRAVEL) DOMAIN LOGIC  ═══════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── Hilton keyword dictionaries ───────────────────────────────────────────────
HILTON_KEYWORDS = {
    "very_negative": [
        "bad", "poor", "terrible", "awful", "worst", "horrible", "pathetic",
        "disappointing", "useless", "waste", "boring", "dull", "confusing",
        "frustrating", "annoying", "unpleasant", "uncomfortable", "disgusting",
        "hate", "hated", "disaster", "nightmare", "rubbish", "garbage",
    ],
    "moderate_negative": [
        "cold", "asleep", "feeling asleep", "sleepy", "tired", "exhausting",
        "no equipment", "no tools", "lack of", "missing", "insufficient",
        "had to write", "cramped", "difficult", "challenging",
        "issue", "problem", "concern",
    ],
    "negative_phrases": [
        "room was cold", "cold room", "cold board room", "feeling asleep",
        "no equipment", "had to write everything", "lack of equipment",
        "waste of time", "complete waste", "not worth", "didn't help",
    ],
    "very_positive": [
        "excellent", "amazing", "outstanding", "brilliant", "superb",
        "fantastic", "awesome", "wonderful", "perfect", "exceptional",
        "mind blowing", "mind-blowing", "loved it", "absolutely love",
    ],
    "positive": [
        "very good", "engaging", "all good", "good going", "overall nice",
        "good", "interactive", "service", "experience", "product",
        "overall good", "everything good", "informative", "helpful", "great",
        "impressive", "valuable", "useful", "beneficial", "effective",
        "satisfactory", "satisfied", "pleased", "happy", "enjoy", "enjoyed",
        "nice session", "informative session", "good session", "great session",
        "well organized", "well done", "keep it up", "thank you", "thanks",
        "appreciate", "friendly", "professional", "approachable", "fun",
        "insightful", "cool", "looking forward", "excited", "like", "liked",
    ],
    "neutral_phrases": [
        "nothing else", "no additional comments", "no comments", "nothing",
        "nothing to add", "no comment", "none so far", "no other comment",
        "nothing more", "no more comments", "nothing specific",
        "nothing in particular", "okay", "fine",
    ],
    "meaningless_patterns": [
        r"^[a-zA-Z]$", r"^[0-9]+$",
        r"^(na|n/a|n\.a|n\|a|n\\a|n\?a|ma|n\./a|n-a)$",
        r"^(nil|none|non|nope)$",
        r"^(ok|okay|yes|no|y|n)$",
        r"^[^\w\s]+$",
        r"^(.)\1+$",
    ],
}

# Flat negative dict for Keyword Analysis page (mirrors PPT structure)
HILTON_NEGATIVE_KEYWORDS = {
    "very_negative":     HILTON_KEYWORDS["very_negative"],
    "moderate_negative": HILTON_KEYWORDS["moderate_negative"],
    "negative_phrases":  HILTON_KEYWORDS["negative_phrases"],
}

HILTON_KEYWORD_SCORES = {
    "very_negative":     -9.0,
    "moderate_negative": -5.0,
    "negative_phrases":  -7.0,
    "very_positive":      9.0,
    "positive":           7.0,
    "neutral_phrases":    0.0,
}

HILTON_SENTIMENT_THRESHOLDS = {
    "very_positive":  5,
    "positive":       1,
    "neutral_low":   -2,
    "negative":      -3,
}

# Compiled constants
_URL_RE           = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE         = re.compile(r"\S+@\S+\.\S+")
_HTML_TAG_RE      = re.compile(r"<[^>]+>")
_MULTI_WS_RE      = re.compile(r"\s+")
_REPEATED_PUNCT   = re.compile(r"([!?.,])\1{1,}")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_EMOJI_RE         = re.compile(
    "["
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "]+", flags=re.UNICODE
)


def hilton_clean_text(raw):
    """Enhanced text cleaning for Hilton domain (URL, HTML, encoding normalisation)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    # Fix common encoding artefacts
    s = s.replace("Ã±", "ñ").replace("Ã¡", "á").replace("Ã©", "é")
    s = s.replace("Ã­", "í").replace("Ã³", "ó").replace("Ãº", "ú")
    s = _html_mod.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = _URL_RE.sub(" ", s)
    s = _EMAIL_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _NON_PRINTABLE_RE.sub(" ", s)
    s = _REPEATED_PUNCT.sub(lambda m: m.group(1), s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    s = re.sub(r"^[^\w']+|[^\w']+$", "", s)
    return s.strip()


def hilton_is_meaningless(text):
    """Detect NA/nil/empty/repeated-word/mostly-numeric text. Returns (bool, reason)."""
    if not text or not text.strip():
        return True, "empty"
    t_lower = text.strip().lower()
    t_norm = re.sub(r"[/\\|.\-?_ ]", "", t_lower)
    if t_norm in {"na", "ma", "nil", "none", "non", "ok", "yes", "no"}:
        return True, f"normalised={t_norm}"
    for pat in HILTON_KEYWORDS["meaningless_patterns"]:
        if re.match(pat, t_lower, re.IGNORECASE):
            return True, f"pattern={pat}"
    if len(text) == 1 and text.lower() not in ("i", "a"):
        return True, "single_char"
    if not any(c.isalnum() for c in text):
        return True, "no_alnum"
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not words:
        return True, "no_words"
    if len(words) > 1 and len(set(words)) == 1:
        return True, "repeated_word"
    num_count = sum(c.isdigit() for c in text)
    if len(text) > 0 and num_count / len(text) > 0.5:
        return True, "mostly_numbers"
    return False, "ok"


_SPANISH_INDICATORS = {
    "tengo", "nada", "mas", "que", "agregar", "muy", "esta", "dia",
    "gracias", "como", "mejorar", "estoy", "trabajo", "comentarios",
}
_ENGLISH_INDICATORS = {
    "the", "is", "are", "was", "were", "have", "has", "had", "will",
    "i", "you", "he", "she", "it", "we", "they", "not", "all",
}


def hilton_detect_language(text):
    """Returns (lang_code, confidence). Falls back to 'en' if langdetect not available."""
    if not text or len(text) < 3:
        return "en", 1.0
    t_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", t_lower))
    spanish_chars = any(c in text for c in "áéíóúñüÃ")
    if spanish_chars or len(words & _SPANISH_INDICATORS) >= 2:
        return "es", 0.95
    if len(words & _ENGLISH_INDICATORS) >= 3:
        return "en", 0.99
    if not _LANGDETECT_OK:
        return "en", 0.80
    try:
        probs = detect_langs(text)
        top = probs[0]
        return top.lang, top.prob
    except Exception:
        return "unknown", 0.0


_TRANSLATION_CONFIDENCE_THRESHOLD = 0.7
_MIN_TRANSLATION_LENGTH = 3


def hilton_smart_translate(text, lang_code, confidence):
    """Translate to English if needed. Returns (translated_text, was_translated)."""
    if lang_code == "en":
        return text, False
    if not _TRANSLATOR_OK:
        return text, False
    if lang_code in ("es", "fr", "pt", "it"):
        if confidence < 0.5:
            return text, False
    else:
        if confidence < _TRANSLATION_CONFIDENCE_THRESHOLD:
            return text, False
    if len(text) < _MIN_TRANSLATION_LENGTH:
        return text, False
    try:
        translated = _GoogleTranslator(source="auto", target="en").translate(text)
        return translated, True
    except Exception:
        return text, False


def _hilton_has_positive_override(text):
    t = text.lower().strip()
    patterns = [
        r"nothing.*(?:happy|good|great|excellent|awesome|perfect|love|amazing)",
        r"no.*(?:complain|issue|problem).*(?:good|great|excellent)",
    ]
    return any(re.search(p, t) for p in patterns)


def _hilton_keyword_score(text):
    """Returns (score, total_matches) on -10…+10 scale."""
    if not text:
        return 0.0, 0
    t_lower = text.lower().strip()
    score, total = 0.0, 0
    for cat, words in [
        ("very_positive",     HILTON_KEYWORDS["very_positive"]),
        ("positive",          HILTON_KEYWORDS["positive"]),
        ("neutral_phrases",   HILTON_KEYWORDS["neutral_phrases"]),
        ("negative_phrases",  HILTON_KEYWORDS["negative_phrases"]),
        ("very_negative",     HILTON_KEYWORDS["very_negative"]),
        ("moderate_negative", HILTON_KEYWORDS["moderate_negative"]),
    ]:
        for kw in words:
            if kw.lower() in t_lower:
                score += HILTON_KEYWORD_SCORES[cat]
                total += 1
    if _hilton_has_positive_override(text):
        score += 6.0
        total += 1
    if total > 0:
        score /= total
    return score, total


def _hilton_hybrid_score(text):
    """VADER(40%) + AFINN(20%) + TextBlob(20%) + Keywords(20%), adaptive weights.
    Returns (score_-10_to_+10, confidence, method_scores_dict)."""
    if not text or not text.strip():
        return 0.0, 0.0, {}

    method_scores = {}
    analyzer = load_vader()

    # VADER
    try:
        vs = analyzer.polarity_scores(text)
        vader_scaled = vs["compound"] * 10
        vader_conf = max(vs["pos"], vs["neg"], vs["neu"])
    except Exception:
        vader_scaled, vader_conf = 0.0, 0.0
    method_scores["vader"] = vader_scaled

    # AFINN
    if _AFINN_OK:
        try:
            afinn_scaled = max(-10.0, min(10.0, _af.score(text) * 2.0))
        except Exception:
            afinn_scaled = 0.0
    else:
        afinn_scaled = vader_scaled * 0.5   # rough fallback
    method_scores["afinn"] = afinn_scaled

    # TextBlob
    if _TEXTBLOB_OK:
        try:
            tb_scaled = _TextBlob(text).sentiment.polarity * 10
        except Exception:
            tb_scaled = 0.0
    else:
        tb_scaled = vader_scaled * 0.5
    method_scores["textblob"] = tb_scaled

    # Keywords
    kw_score, kw_total = _hilton_keyword_score(text)
    method_scores["keywords"] = kw_score

    # Adaptive weights
    n_words = len(text.split())
    has_caps = any(c.isupper() for c in text)
    has_excl = "!" in text
    has_emoji = bool(_EMOJI_RE.search(text))

    w = {"vader": 0.40, "afinn": 0.20, "textblob": 0.20, "keywords": 0.20}
    if has_caps or has_excl or has_emoji:
        w["vader"] += 0.15; w["textblob"] -= 0.05; w["afinn"] -= 0.05; w["keywords"] -= 0.05
    if kw_total >= 3:
        w["keywords"] += 0.15; w["vader"] -= 0.05; w["afinn"] -= 0.05; w["textblob"] -= 0.05
    elif kw_total >= 1:
        w["keywords"] += 0.10; w["vader"] -= 0.05; w["afinn"] -= 0.025; w["textblob"] -= 0.025
    if n_words <= 5:
        w["keywords"] += 0.10; w["afinn"] += 0.05; w["vader"] -= 0.10; w["textblob"] -= 0.05
    if n_words > 20:
        w["vader"] += 0.10; w["textblob"] += 0.05; w["keywords"] -= 0.10; w["afinn"] -= 0.05
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    combined = max(-10.0, min(10.0,
        w["vader"]    * vader_scaled +
        w["afinn"]    * afinn_scaled +
        w["textblob"] * tb_scaled    +
        w["keywords"] * kw_score
    ))

    # Confidence
    scores = [vader_scaled, afinn_scaled, tb_scaled, kw_score]
    nz = [s for s in scores if abs(s) > 0.1]
    agree_conf = max(0.0, 1.0 - (np.std(nz) / 10.0)) if len(nz) >= 2 else 0.5
    conf = vader_conf * 0.6 + agree_conf * 0.4
    if kw_total >= 2:
        conf = min(1.0, conf + 0.15)
    conf = max(0.3, min(0.95, conf))

    return combined, conf, method_scores


def classify_sentiment_hilton(score):
    """Map -10…+10 score to 5-class label (matching PPT label names for UI consistency)."""
    if pd.isna(score):
        return "Neutral"
    if score >= HILTON_SENTIMENT_THRESHOLDS["very_positive"]:
        return "Very Positive"
    if score >= HILTON_SENTIMENT_THRESHOLDS["positive"]:
        return "Positive"
    if score >= HILTON_SENTIMENT_THRESHOLDS["neutral_low"]:
        return "Neutral"
    if score >= HILTON_SENTIMENT_THRESHOLDS["negative"]:
        return "Negative"
    return "Very Negative"


def _extract_hilton_negative_keywords(text):
    """Extract negative keyword matches for Hilton domain."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t_lower = text.lower()
    found, seen = [], set()
    all_kws = sorted(
        [kw for kws in HILTON_NEGATIVE_KEYWORDS.values() for kw in kws],
        key=len, reverse=True
    )
    for kw in all_kws:
        if kw.lower() in t_lower and kw.lower() not in seen:
            seen.add(kw.lower())
            found.append(kw)
    return "; ".join(found)


def run_hilton_analysis(df, id_col, text_col, validation_dict, progress_cb=None):
    """Full Hilton domain pipeline: clean → meaningless-filter → lang-detect → translate → hybrid score."""
    analyzer_missing = []
    if not _AFINN_OK:      analyzer_missing.append("afinn")
    if not _TEXTBLOB_OK:   analyzer_missing.append("textblob")
    if not _LANGDETECT_OK: analyzer_missing.append("langdetect")
    if not _TRANSLATOR_OK: analyzer_missing.append("deep-translator")
    if analyzer_missing:
        st.info(f"ℹ️ Optional packages not found: `{', '.join(analyzer_missing)}` — running with available engines only.")

    df = df.copy()
    df["_id_norm"] = df[id_col].astype(str).str.strip().str.replace(" ", "")
    df["Cleaned_Comments"] = df[text_col].apply(hilton_clean_text)

    n = len(df)
    compounds, confidences, sentiments, sources, translated_texts, methods_log = [], [], [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        if progress_cb and i % max(1, n // 100) == 0:
            progress_cb(i, n)

        norm_id = row["_id_norm"]
        cleaned = row["Cleaned_Comments"]

        # Validation override
        if norm_id in validation_dict:
            label = validation_dict[norm_id]
            score = SENTIMENT_TO_SCORE.get(label, 0.0)
            # Re-scale to -10…+10 for Hilton; keep raw labels
            compounds.append(score * 10)
            confidences.append(1.0)
            sentiments.append(label)
            sources.append("Validation")
            translated_texts.append("")
            methods_log.append("validation")
            continue

        sources.append("Model")

        # Meaningless check
        is_mless, _ = hilton_is_meaningless(cleaned)
        if is_mless or not cleaned:
            compounds.append(0.0)
            confidences.append(0.0)
            sentiments.append("Neutral")
            translated_texts.append("")
            methods_log.append("blank")
            continue

        # Language detection + translation
        lang, lang_conf = hilton_detect_language(cleaned)
        text_for_scoring = cleaned
        trans = ""
        if lang != "en":
            translated, did_translate = hilton_smart_translate(cleaned, lang, lang_conf)
            if did_translate:
                text_for_scoring = translated
                trans = translated
        translated_texts.append(trans)

        # Hybrid scoring
        score, conf, _ = _hilton_hybrid_score(text_for_scoring)
        compounds.append(round(score, 3))
        confidences.append(round(conf, 3))
        sentiments.append(classify_sentiment_hilton(score))
        methods_log.append("hybrid")

    if progress_cb:
        progress_cb(n, n)

    df["Translated_Text"]    = translated_texts
    df["Text_For_Analysis"]  = [
        (t if t else c) for t, c in zip(translated_texts, df["Cleaned_Comments"])
    ]
    df["consumer_score"]     = compounds
    df["confidence"]         = confidences
    df["consumer_sentiment"] = sentiments
    df["validation_source"]  = sources
    df["Negative_Keywords"]  = df["Text_For_Analysis"].apply(_extract_hilton_negative_keywords)
    return df


# ── Unified dispatcher ────────────────────────────────────────────────────────
def run_analysis(df, domain, id_col, text_col, validation_dict, progress_cb=None):
    """Route to domain-specific analysis pipeline."""
    if domain == "hilton":
        return run_hilton_analysis(df, id_col, text_col, validation_dict, progress_cb)
    else:
        return run_ppt_analysis(df, id_col, text_col, validation_dict, progress_cb)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════  UI HELPERS  ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

SENTIMENT_ORDER  = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]
SENTIMENT_COLORS = {
    "Very Positive": "#3D7A5F", "Positive": "#6B9E50",
    "Neutral": "#6B8A99", "Negative": "#C87A40", "Very Negative": "#A04040",
}

def mcard(label, value, color="var(--teal)"):
    return (f'<div class="mc" style="border-top-color:{color}">'
            f'<p class="mv">{value}</p><p class="ml">{label}</p></div>')

def shdr(text, icon="📊"):
    st.markdown(f'<div class="sh">{icon}&nbsp;{text}</div>', unsafe_allow_html=True)

def _mfig(fig, h=380):
    fig.update_layout(
        font_family="DM Sans", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=10), height=h,
        xaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,.15)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(168,188,200,.15)", zeroline=False),
        hoverlabel=dict(bgcolor="#1E2D33", font_size=12, font_family="DM Sans", font_color="#E8E6DD"),
    )
    return fig

@st.cache_data(show_spinner=False)
def _read_file(raw, name):
    buf = io.BytesIO(raw)
    return pd.read_csv(buf) if name.endswith(".csv") else pd.read_excel(buf)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════  SIDEBAR  ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 0 14px">
      <div style="width:36px;height:36px;border-radius:10px;
           background:linear-gradient(135deg,#2D5F6E,#3A7A8C);
           display:flex;align-items:center;justify-content:center">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
             viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
      </div>
      <div>
        <span style="font-size:16px;font-weight:700;color:#1E2D33">SentimentHub</span><br>
        <span style="font-size:11px;color:#6B8A99;font-style:italic">Multi-Domain · Vader Adaptive</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Navigation
    PAGES = ["Upload & Analyse", "Reports & Insights", "Keyword Analysis", "Audit Trail"]
    if "_page" not in st.session_state:
        st.session_state._page = "Upload & Analyse"
    nav_target = st.session_state.pop("_nav_target", None)
    if nav_target and nav_target in PAGES:
        st.session_state._page = nav_target

    st.markdown('<div style="font-size:11px;font-weight:600;color:#6B8A99;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Navigation</div>',
                unsafe_allow_html=True)
    NAV_ICONS = {
        "Upload & Analyse":  ":material/upload:",
        "Reports & Insights":":material/analytics:",
        "Keyword Analysis":  ":material/search:",
        "Audit Trail":       ":material/visibility:",
    }
    for p in PAGES:
        active = st.session_state._page == p
        if st.button(p, key=f"nav_{p}", icon=NAV_ICONS[p],
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state._page = p
            st.rerun()
    page = st.session_state._page
    st.divider()

    # Domain selector
    st.markdown("**🌐 Domain**")
    domain = st.selectbox(
        "Select domain",
        list(DOMAIN_CONFIG.keys()),
        format_func=lambda k: DOMAIN_CONFIG[k]["label"],
        key="sb_domain",
    )
    _dcfg = DOMAIN_CONFIG[domain]
    st.divider()

    # Column mapping (defaults update with domain)
    st.markdown("**⚙️ Column Mapping**")
    id_col   = st.text_input("ID Column",           value=_dcfg["id_default"],   key="sb_id")
    text_col = st.text_input("Conversation Column", value=_dcfg["text_default"], key="sb_text")
    st.divider()

    # Validation file
    st.markdown("**🛡️ Validation Override**")
    val_file = st.file_uploader(
        "Validation Excel (optional)", type=["xlsx"], key="sb_val",
        help="Must have ID and 'Actual Sentiment' columns."
    )
    st.divider()

    # Threshold (PPT only — shown for completeness)
    st.markdown("**🎛️ Options**")
    rule_threshold = st.slider("Rule confidence threshold", 0.50, 0.99, 0.70, 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
  <div style="width:44px;height:44px;border-radius:12px;
       background:linear-gradient(135deg,#2D5F6E,#3A7A8C);
       display:flex;align-items:center;justify-content:center;
       flex-shrink:0;box-shadow:0 4px 12px rgba(45,95,110,.25)">
    <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22"
         viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.5">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>
  </div>
  <div>
    <h1 style="margin:0;font-size:24px;line-height:1.2;color:#1E2D33">SentimentHub</h1>
    <p style="margin:0;color:#6B8A99;font-size:12px" id="app-domain-label">
      Multi-Domain Sentiment Analysis · VADER Adaptive
    </p>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — UPLOAD & ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Upload & Analyse":
    shdr("Upload & Analyse", "📤")

    # Show last-run summary
    if "result" in st.session_state:
        out = st.session_state.result
        n   = len(out)
        t   = st.session_state.get("_run_time", 0)
        fn  = st.session_state.get("_filename", "data")
        v   = int((out["validation_source"] == "Validation").sum())
        st.markdown(
            f'<div style="background:var(--card);border:1px solid var(--border);'
            f'border-radius:10px;padding:12px 16px;margin-bottom:14px;'
            f'border-left:3px solid var(--success)">'
            f'<span style="color:var(--success);font-weight:600">✓ {fn}</span>'
            f'<span style="color:var(--muted);font-size:13px;margin-left:8px">'
            f'{n:,} records analysed in {t:.1f}s'
            f'{f" · {v:,} validation overrides" if v else ""}'
            f' — see Reports ›</span></div>',
            unsafe_allow_html=True)

    # File upload
    uploaded = st.file_uploader(
        "Upload conversation data (Excel or CSV)",
        type=["xlsx", "csv"], label_visibility="visible", key="main_upload"
    )
    if uploaded is None:
        st.info("⬆️ Upload a file to get started. The file must contain an ID column and a conversation/comments column.")
        st.stop()

    with st.spinner("Reading file…"):
        df_raw = _read_file(uploaded.read(), uploaded.name)

    st.success(f"Loaded **{len(df_raw):,}** rows × {len(df_raw.columns)} columns from `{uploaded.name}`")

    # Column check
    missing = [c for c in [id_col, text_col] if c not in df_raw.columns]
    if missing:
        st.error(f"Column(s) **{missing}** not found. Available columns: `{list(df_raw.columns)}`")
        st.stop()

    with st.expander("Preview raw data (first 5 rows)"):
        st.dataframe(df_raw[[id_col, text_col]].head(5), use_container_width=True)

    # Validation summary
    validation_dict = {}
    if val_file:
        val_file.seek(0)
        validation_dict = load_validation_data(val_file)
        if validation_dict:
            st.markdown(
                f'<span class="badge b-ok">✓ {len(validation_dict):,} validation records loaded</span>',
                unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge b-warn">⚠️ Validation file loaded but no records matched expected columns</span>',
                        unsafe_allow_html=True)
    st.divider()

    # Domain badge
    st.markdown(
        f'<span class="badge b-info">🌐 {DOMAIN_CONFIG[domain]["label"]}</span>',
        unsafe_allow_html=True)

    # Run button
    _, rc, _ = st.columns([1, 2, 1])
    with rc:
        _btn_label = DOMAIN_CONFIG[domain]["label"].split(" — ")[0]
        run = st.button(
            f"🚀 Run {_btn_label} Sentiment Analysis  ({len(df_raw):,} records)",
            type="primary", use_container_width=True
        )

    if run:
        prog = st.progress(0, text="Starting…")
        t0   = time.time()

        def update_progress(done, total):
            pct = int(done / total * 100) if total else 0
            prog.progress(pct, text=f"Processing {done:,} / {total:,} rows…")

        with st.spinner("Running analysis…"):
            result_df = run_analysis(
                df_raw, domain, id_col, text_col, validation_dict, update_progress
            )

        elapsed = time.time() - t0
        prog.progress(100, text=f"✓ Done — {len(result_df):,} records in {elapsed:.1f}s")

        st.session_state.result      = result_df
        st.session_state._run_time   = elapsed
        st.session_state._filename   = uploaded.name
        st.session_state._id_col     = id_col
        st.session_state._text_col   = text_col
        st.session_state._domain     = domain

        st.toast(f"Analysis complete: {len(result_df):,} records classified in {elapsed:.1f}s")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — REPORTS & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Reports & Insights":
    shdr("Reports & Insights", "📊")

    if "result" not in st.session_state:
        st.info("No results yet. Go to **Upload & Analyse** first.")
        st.stop()

    out   = st.session_state.result
    id_col   = st.session_state.get("_id_col", id_col)
    text_col = st.session_state.get("_text_col", text_col)
    total = len(out)
    dist  = out["consumer_sentiment"].value_counts()
    pt    = st.session_state.get("_run_time", 0)
    src   = out["validation_source"].value_counts()

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(mcard("Total Records",    f"{total:,}"),                                            unsafe_allow_html=True)
    k2.markdown(mcard("Analysis Time",    f"{pt:.1f}s",   "var(--slate)"),                          unsafe_allow_html=True)
    k3.markdown(mcard("Very Negative",    str(dist.get("Very Negative", 0)), "#A04040"),             unsafe_allow_html=True)
    k4.markdown(mcard("Negative",         str(dist.get("Negative", 0)),      "#C87A40"),             unsafe_allow_html=True)
    k5.markdown(mcard("Neutral",          str(dist.get("Neutral", 0)),        "var(--slate)"),       unsafe_allow_html=True)
    k6.markdown(mcard("Positive / Very+", str(dist.get("Positive", 0) + dist.get("Very Positive", 0)), "var(--success)"),
                unsafe_allow_html=True)

    # Source badges
    val_n   = int(src.get("Validation", 0))
    model_n = int(src.get("Model", 0))

    # Count rule vs vader inside Model rows
    rule_n  = int((out[(out["validation_source"] == "Model") & (out["confidence"] > 0.5)].shape[0]))
    vader_n = int((out[(out["validation_source"] == "Model") & (out["confidence"] <= 0.5)].shape[0]))

    st.markdown(
        f'<div style="margin:8px 0 4px">'
        f'<span class="badge b-info">Validation override: {val_n:,}</span> &nbsp;'
        f'<span class="badge b-ok">Rule-based: {rule_n:,}</span> &nbsp;'
        f'<span class="badge b-warn">VADER fallback: {vader_n:,}</span>'
        f'</div>',
        unsafe_allow_html=True)
    st.divider()

    tab_dist, tab_score, tab_data = st.tabs(["Distribution", "Score Analysis", "Full Results"])

    # ── Distribution tab ─────────────────────────────────────────────────────
    with tab_dist:
        ordered = [s for s in SENTIMENT_ORDER if s in dist.index]
        counts  = [int(dist[s]) for s in ordered]
        colors  = [SENTIMENT_COLORS[s] for s in ordered]
        pcts    = [round(c / total * 100, 1) for c in counts]

        # Bar chart
        fig_bar = go.Figure(go.Bar(
            x=ordered, y=counts,
            marker=dict(color=colors, cornerradius=6, line=dict(width=0)),
            text=[f"{p}%" for p in pcts],
            textposition="outside",
            textfont=dict(size=13, color="#2D5F6E", family="DM Sans"),
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<br>Share: %{text}<extra></extra>",
        ))
        _mfig(fig_bar, 380).update_layout(
            title=dict(text="Sentiment Distribution", font=dict(size=15, color="#1E2D33"), x=0),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="rpt_bar")

        c1, c2 = st.columns(2)
        with c1:
            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=ordered, values=counts,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent", hole=0.4,
                hovertemplate="<b>%{label}</b><br>%{value:,} records (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                font_family="DM Sans", paper_bgcolor="rgba(0,0,0,0)",
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Sentiment Share", font=dict(size=14, color="#1E2D33"), x=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="rpt_pie")

        with c2:
            # Source breakdown donut
            src_labels  = ["Validation Override", "Rule-based", "VADER Fallback"]
            src_values  = [val_n, rule_n, vader_n]
            src_colors  = ["#3D7A5F", "#2D5F6E", "#D4B94E"]
            fig_src = go.Figure(go.Pie(
                labels=src_labels, values=src_values,
                marker=dict(colors=src_colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent", hole=0.4,
                hovertemplate="<b>%{label}</b><br>%{value:,} records<extra></extra>",
            ))
            fig_src.update_layout(
                font_family="DM Sans", paper_bgcolor="rgba(0,0,0,0)",
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text="Classification Source", font=dict(size=14, color="#1E2D33"), x=0),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_src, use_container_width=True, key="rpt_src")

    # ── Score Analysis tab ────────────────────────────────────────────────────
    with tab_score:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = go.Figure(go.Histogram(
                x=out["consumer_score"], nbinsx=40,
                marker=dict(
                    color=out["consumer_score"].apply(
                        lambda s: "#A04040" if s < -0.2 else "#3D7A5F" if s > 0.2 else "#6B8A99"
                    ),
                    line=dict(width=0),
                ),
                hovertemplate="Score: %{x:.2f}<br>Count: %{y:,}<extra></extra>",
            ))
            _mfig(fig_hist, 340).update_layout(
                title=dict(text="Compound Score Distribution", font=dict(size=14, color="#1E2D33"), x=0),
                bargap=0.05,
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="rpt_hist")

        with c2:
            conf_bins = pd.cut(
                out["confidence"],
                bins=[0, 0.3, 0.6, 0.8, 1.01],
                labels=["Low\n(<0.3)", "Medium\n(0.3–0.6)", "High\n(0.6–0.8)", "Very High\n(>0.8)"]
            )
            conf_dist = conf_bins.value_counts().sort_index()
            fig_conf = go.Figure(go.Bar(
                x=conf_dist.index.tolist(), y=conf_dist.values,
                marker=dict(
                    color=["#A04040", "#C87A40", "#3A7A8C", "#3D7A5F"],
                    cornerradius=6, line=dict(width=0)
                ),
                text=[f"{v:,}" for v in conf_dist.values],
                textposition="outside",
                textfont=dict(size=12, color="#2D5F6E"),
            ))
            _mfig(fig_conf, 340).update_layout(
                title=dict(text="Confidence Band Distribution", font=dict(size=14, color="#1E2D33"), x=0),
                showlegend=False,
            )
            st.plotly_chart(fig_conf, use_container_width=True, key="rpt_conf")

        # Sentiment by source stacked bar
        shdr("Sentiment by Classification Source", "📈")
        src_sent = out.groupby(["validation_source", "consumer_sentiment"]).size().reset_index(name="Count")
        fig_ss = go.Figure()
        for sent in SENTIMENT_ORDER:
            sub = src_sent[src_sent["consumer_sentiment"] == sent]
            fig_ss.add_trace(go.Bar(
                x=sub["validation_source"], y=sub["Count"],
                name=sent,
                marker=dict(color=SENTIMENT_COLORS[sent], cornerradius=4),
                hovertemplate=f"<b>{sent}</b><br>Source: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
            ))
        _mfig(fig_ss, 360).update_layout(
            barmode="stack", title=dict(text="Sentiment by Source", font=dict(size=14, color="#1E2D33"), x=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        )
        st.plotly_chart(fig_ss, use_container_width=True, key="rpt_ss")

    # ── Full Results tab ──────────────────────────────────────────────────────
    with tab_data:
        f1, f2 = st.columns(2)
        with f1:
            filt_sent = st.multiselect("Filter by sentiment", SENTIMENT_ORDER, default=[], key="rpt_filt_sent")
        with f2:
            filt_src = st.multiselect("Filter by source", ["Validation", "Model"], default=[], key="rpt_filt_src")

        disp = out.copy()
        if filt_sent: disp = disp[disp["consumer_sentiment"].isin(filt_sent)]
        if filt_src:  disp = disp[disp["validation_source"].isin(filt_src)]

        show_cols = [c for c in [
            id_col, text_col, "CustomerOnly", "consumer_sentiment",
            "consumer_score", "confidence", "validation_source", "Negative_Keywords"
        ] if c in disp.columns]
        st.markdown(f'<span class="badge b-info">{len(disp):,} rows shown</span>', unsafe_allow_html=True)
        st.dataframe(disp[show_cols].reset_index(drop=True), use_container_width=True, height=420)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    shdr("Export Results", "⬇️")
    _dl_domain = st.session_state.get("_domain", "sentiment")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "⬇️ Download CSV",
            data=out.to_csv(index=False).encode(),
            file_name=f"{_dl_domain}_sentiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True, key="dl_csv",
        )
    with e2:
        xls_buf = io.BytesIO()
        with pd.ExcelWriter(xls_buf, engine="openpyxl") as writer:
            # Sheet 1 — Full results
            out.to_excel(writer, sheet_name="Results", index=False)
            # Sheet 2 — Summary
            summary_rows = []
            for sent in SENTIMENT_ORDER:
                cnt = int(dist.get(sent, 0))
                summary_rows.append({
                    "Sentiment": sent,
                    "Count": cnt,
                    "%": round(cnt / total * 100, 1) if total else 0,
                })
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            # Sheet 3 — Negative keywords
            kw_all = (out["Negative_Keywords"].dropna()
                      .str.split("; ").explode().str.strip()
                      .loc[lambda s: s != ""])
            if not kw_all.empty:
                kw_df = kw_all.value_counts().reset_index()
                kw_df.columns = ["Keyword", "Count"]
                kw_df.to_excel(writer, sheet_name="Keywords", index=False)
        st.download_button(
            "⬇️ Download Excel (3 sheets)",
            data=xls_buf.getvalue(),
            file_name=f"{_dl_domain}_sentiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, key="dl_excel",
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — KEYWORD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Keyword Analysis":
    shdr("Negative Keyword Analysis", "🔍")

    if "result" not in st.session_state:
        st.info("Run analysis first."); st.stop()

    out = st.session_state.result

    kw_series = (
        out["Negative_Keywords"].dropna()
        .str.split("; ").explode().str.strip()
        .loc[lambda s: s != ""]
    )

    if kw_series.empty:
        st.markdown('<span class="badge b-ok">No negative keywords found in dataset.</span>',
                    unsafe_allow_html=True)
        st.stop()

    # ── Top keywords chart ────────────────────────────────────────────────────
    top_n = st.slider("Top N keywords to display", 10, 60, 30, key="kw_topn")
    kw_counts = kw_series.value_counts().head(top_n).reset_index()
    kw_counts.columns = ["Keyword", "Count"]

    fig_kw = go.Figure(go.Bar(
        y=kw_counts["Keyword"], x=kw_counts["Count"],
        orientation="h",
        marker=dict(
            color=kw_counts["Count"],
            colorscale=[[0, "#F0E6C8"], [0.5, "#C87A40"], [1, "#A04040"]],
            cornerradius=4, line=dict(width=0),
        ),
        text=[f"{v:,}" for v in kw_counts["Count"]],
        textposition="outside",
        textfont=dict(size=11, color="#2D5F6E", family="DM Sans"),
        hovertemplate="<b>%{y}</b><br>Occurrences: %{x:,}<extra></extra>",
    ))
    _mfig(fig_kw, max(380, top_n * 26)).update_layout(
        yaxis={"categoryorder": "total ascending"},
        title=dict(text=f"Top {top_n} Negative Keywords / Phrases", font=dict(size=15, color="#1E2D33"), x=0),
        showlegend=False, coloraxis_showscale=False,
    )
    st.plotly_chart(fig_kw, use_container_width=True, key="kw_main")

    # ── By category ───────────────────────────────────────────────────────────
    st.divider()
    shdr("Keywords by Issue Category", "📋")
    _active_domain = st.session_state.get("_domain", "ppt")
    _kw_dict = HILTON_NEGATIVE_KEYWORDS if _active_domain == "hilton" else NEGATIVE_KEYWORDS
    _dict_label = "Hilton" if _active_domain == "hilton" else "PPT"
    st.markdown(f"Each category reflects the issue group from the **{_dict_label}** keyword dictionary.")

    cat_data = []
    for cat, keywords in _kw_dict.items():
        cat_pattern = re.compile(
            "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True)),
            re.IGNORECASE
        )
        total_hits = out["Text_For_Analysis"].dropna().apply(
            lambda t: bool(cat_pattern.search(str(t)))
        ).sum()
        if total_hits > 0:
            cat_data.append({"Category": cat.replace("_", " ").title(), "Records with hits": int(total_hits)})

    if cat_data:
        cat_df = pd.DataFrame(cat_data).sort_values("Records with hits", ascending=False)
        fig_cat = go.Figure(go.Bar(
            y=cat_df["Category"], x=cat_df["Records with hits"],
            orientation="h",
            marker=dict(
                color=cat_df["Records with hits"],
                colorscale=[[0, "#D6E8EE"], [0.5, "#3A7A8C"], [1, "#1E3A44"]],
                cornerradius=4, line=dict(width=0),
            ),
            text=[f"{v:,}" for v in cat_df["Records with hits"]],
            textposition="outside",
            textfont=dict(size=11, color="#2D5F6E"),
            hovertemplate="<b>%{y}</b><br>Records: %{x:,}<extra></extra>",
        ))
        _mfig(fig_cat, max(300, len(cat_df) * 32)).update_layout(
            yaxis={"categoryorder": "total ascending"},
            title=dict(text="Issue Category Frequency", font=dict(size=14, color="#1E2D33"), x=0),
            showlegend=False, coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="kw_cat")

    # ── Filter by sentiment ───────────────────────────────────────────────────
    st.divider()
    shdr("Filter Keywords by Sentiment", "🎯")
    sent_pick = st.selectbox("Show keywords for", ["All"] + SENTIMENT_ORDER, key="kw_sent_pick")
    filtered = out if sent_pick == "All" else out[out["consumer_sentiment"] == sent_pick]
    kw2 = (filtered["Negative_Keywords"].dropna()
           .str.split("; ").explode().str.strip()
           .loc[lambda s: s != ""])
    if not kw2.empty:
        kw2_df = kw2.value_counts().head(25).reset_index()
        kw2_df.columns = ["Keyword", "Count"]
        st.dataframe(kw2_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No keywords found for {sent_pick}.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — AUDIT TRAIL
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Audit Trail":
    shdr("Audit Trail — Row-Level Traceability", "🔎")

    if "result" not in st.session_state:
        st.info("Run analysis first."); st.stop()

    out      = st.session_state.result
    id_col   = st.session_state.get("_id_col", id_col)
    text_col = st.session_state.get("_text_col", text_col)

    a1, a2, a3 = st.columns(3)
    with a1:
        f_sent = st.selectbox("Sentiment",   ["All"] + SENTIMENT_ORDER, key="at_sent")
    with a2:
        f_src  = st.selectbox("Source",      ["All", "Validation", "Model"],              key="at_src")
    with a3:
        f_conf = st.selectbox("Confidence",  ["All", "High (>0.7)", "Medium (0.3–0.7)", "Low (<0.3)"], key="at_conf")

    ad = out.copy()
    if f_sent != "All": ad = ad[ad["consumer_sentiment"] == f_sent]
    if f_src  != "All": ad = ad[ad["validation_source"]  == f_src]
    if "High"   in f_conf: ad = ad[ad["confidence"] > 0.7]
    elif "Medium" in f_conf: ad = ad[(ad["confidence"] >= 0.3) & (ad["confidence"] <= 0.7)]
    elif "Low"    in f_conf: ad = ad[ad["confidence"] < 0.3]

    st.markdown(f'<span class="badge b-info">{len(ad):,} rows match filters</span>', unsafe_allow_html=True)

    show = [c for c in [
        id_col, text_col,
        "CustomerOnly", "Cleaned_Comments", "Translated_Text", "Text_For_Analysis",
        "consumer_sentiment", "consumer_score",
        "confidence", "validation_source", "Negative_Keywords"
    ] if c in ad.columns]
    st.dataframe(ad[show].reset_index(drop=True), use_container_width=True, height=480)

    # Export filtered
    _audit_domain = st.session_state.get("_domain", "sentiment")
    st.download_button(
        "⬇️ Export filtered audit CSV",
        data=ad.to_csv(index=False).encode(),
        file_name=f"{_audit_domain}_audit_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv", use_container_width=True, key="at_dl",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center;color:#6B8A99;font-size:11px;padding:20px 0 4px">'
    'SentimentHub &nbsp;|  Multi-Domain · VADER Adaptive · Hybrid Model · Validation Override'
    '</div>',
    unsafe_allow_html=True
)
