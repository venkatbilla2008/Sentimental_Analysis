"""
Shared utilities, constants, and keyword dictionaries used across all domains.
"""
import io
import re

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

# ── Sentiment score mapping ───────────────────────────────────────────────────
SENTIMENT_TO_SCORE = {
    "Very Positive":  0.85,
    "Positive":       0.35,
    "Neutral":        0.0,
    "Negative":      -0.55,
    "Very Negative": -0.85,
}

# ── Shared negative keyword dictionary (PPT / Netflix / Spotify all extend this)
SHARED_NEGATIVE_KEYWORDS = {
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
    "general_negative": [
        "problem", "issue", "concern", "complaint",
        "trouble", "difficulty", "struggle",
        "wrong", "incorrect", "mistake", "error",
        "broken", "damaged", "defective",
        "lost", "missing", "disappeared",
        "failed", "failure", "unsuccessful",
    ],
}


# ── VADER / BERT loaders (cached at app level) ────────────────────────────────
@st.cache_resource
def load_vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


@st.cache_resource
def _load_bert_optional():
    """DistilBERT for borderline Spotify corrections. Returns None if unavailable."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        return hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )
    except Exception:
        return None


# ── Sentiment classification ──────────────────────────────────────────────────
def classify_sentiment(score):
    if pd.isna(score):
        return "Neutral"
    if score >= 0.60:   return "Very Positive"
    if score >= 0.20:   return "Positive"
    if score >= -0.20:  return "Neutral"
    if score >= -0.60:  return "Negative"
    return "Very Negative"


def classify_sentiment_expr(col: str = "consumer_score") -> pl.Expr:
    """Vectorized Polars expression equivalent of classify_sentiment."""
    return (
        pl.when(pl.col(col).is_null()).then(pl.lit("Neutral"))
        .when(pl.col(col) >= 0.60).then(pl.lit("Very Positive"))
        .when(pl.col(col) >= 0.20).then(pl.lit("Positive"))
        .when(pl.col(col) >= -0.20).then(pl.lit("Neutral"))
        .when(pl.col(col) >= -0.60).then(pl.lit("Negative"))
        .otherwise(pl.lit("Very Negative"))
    )


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
            if cur:
                chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    if not chunks:
        return 0.0
    return float(np.mean([analyzer.polarity_scores(c)['compound'] for c in chunks]))


# ── Keyword pattern builder + extractor (shared across all domains) ───────────
def build_neg_kw_pattern(keyword_dict):
    """Compile a single regex from all keywords in a dict, longest-match first."""
    all_kws = sorted(
        [kw for kws in keyword_dict.values() for kw in kws],
        key=len, reverse=True,
    )
    return re.compile("|".join(re.escape(k) for k in all_kws), re.IGNORECASE)


def extract_negative_keywords(text, pattern):
    if not isinstance(text, str) or not text.strip():
        return ""
    matches = pattern.findall(text)
    if not matches:
        return ""
    seen, found = set(), []
    for m in matches:
        ml = m.lower()
        if ml not in seen:
            seen.add(ml)
            found.append(m)
    return "; ".join(found)


# ── Parallel VADER scorer (used by PPT and Netflix) ──────────────────────────
def run_vader_parallel(texts_list, needs_vader, analyzer, max_workers=6):
    """Run VADER in parallel threads for rows marked True in needs_vader.
    Returns a float64 numpy array (NaN for rows that weren't scored)."""
    from concurrent.futures import ThreadPoolExecutor
    vader_inputs = [
        (i, texts_list[i]) for i, need in enumerate(needs_vader)
        if need and texts_list[i].strip()
    ]
    vader_raw = np.full(len(texts_list), np.nan)
    if not vader_inputs:
        return vader_raw

    def _score(args):
        i, text = args
        return i, get_vader_compound(text, analyzer)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, score in ex.map(_score, vader_inputs):
            vader_raw[i] = score
    return vader_raw


# ── Context helpers shared by PPT and Netflix ─────────────────────────────────
def has_resolution_or_thanks(text):
    tl = text.lower()
    indicators = [
        "thanks", "thank you", "thank u", "ty", "okay", "ok", "ok thanks",
        "perfect", "got it", "understood", "appreciate", "that will be all",
        "sounds good", "that works", "all set", "all good", "we're good",
    ]
    return any(i in tl for i in indicators)


def is_polite_request(text):
    tl = text.lower()
    polite = ["please", "can you", "could you", "would you", "i want to", "i need to",
              "i would like", "help me", "can i", "may i", "how to", "how can", "how do i"]
    frustration = ["frustrated", "angry", "annoyed", "furious", "terrible", "horrible",
                   "worst", "pathetic", "useless", "awful", "unacceptable", "ridiculous"]
    return any(p in tl for p in polite) and not any(f in tl for f in frustration)


def has_simple_cancellation(text):
    tl = text.lower()
    simple = ["cancel my appointment", "cancel appointment", "need to cancel",
              "want to cancel", "please cancel", "can i cancel", "reschedule"]
    complaint = ["frustrated", "angry", "terrible", "horrible", "long wait",
                 "no show", "late", "problem", "issue", "complaint"]
    return any(s in tl for s in simple) and not any(c in tl for c in complaint)


# ── Validation data loader ────────────────────────────────────────────────────
def load_validation_data(uploaded_file):
    validation_dict = {}
    if uploaded_file is None:
        return validation_dict
    try:
        plv_raw  = pl.read_excel(io.BytesIO(uploaded_file.read()))
        id_col = next(
            (col for col in plv_raw.columns
             if col.strip().upper() in ['ID', 'PPTLEADS_COMM_ID', 'TICKET ID', 'TICKET_ID']),
            None,
        )
        sent_col = next(
            (col for col in plv_raw.columns if col.lower().strip() == 'actual sentiment'),
            None,
        )
        if not id_col or not sent_col:
            st.warning(f"⚠️ Validation file columns not matched. Found: {plv_raw.columns}")
            return validation_dict
        plv = plv_raw.select([id_col, sent_col]).filter(pl.col(sent_col).is_not_null())
        plv = plv.with_columns(
            pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_k"),
            pl.col(sent_col).cast(pl.String).str.strip_chars().alias("_v"),
        ).filter(pl.col("_v").str.len_bytes() > 0)
        validation_dict = dict(zip(plv["_k"].to_list(), plv["_v"].to_list()))
        return validation_dict
    except Exception as e:
        st.warning(f"Could not load validation file: {e}")
        return {}
