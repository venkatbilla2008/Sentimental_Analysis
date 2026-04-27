"""
Hilton (Travel & Hospitality) domain logic.
"""
import html as _html_mod
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

from .shared import (
    SENTIMENT_TO_SCORE, load_vader,
    build_neg_kw_pattern, extract_negative_keywords, classify_sentiment_expr,
)

# ── Optional multilingual packages ────────────────────────────────────────────
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

HILTON_NEGATIVE_KEYWORDS = {
    "very_negative":     HILTON_KEYWORDS["very_negative"],
    "moderate_negative": HILTON_KEYWORDS["moderate_negative"],
    "negative_phrases":  HILTON_KEYWORDS["negative_phrases"],
    "account_merge": [
        "merge account", "merge accounts", "duplicate account",
        "account consolidation", "account i mistakenly created",
        "two accounts", "combine accounts", "link accounts",
        "separate accounts", "merge my profile",
    ],
    "room_promise_violation": [
        "connecting rooms not available", "room not available",
        "promised room not honored", "confirmed room type unavailable",
        "room was not ready", "wrong room type", "room downgrade",
        "booked confirmed", "reservation not honored",
        "room change without notice", "different room than booked",
        "not the room i reserved", "suite not available",
    ],
    "loyalty_pricing": [
        "member cost more than online", "loyalty price higher",
        "rewards member more expensive", "member rate not applied",
        "honors price wrong", "hilton honors discount not applied",
        "points not credited", "points missing", "points not added",
        "status not recognized", "rewards not applied",
        "member rate not honored", "discount not applied",
    ],
    "partial_cancellation": [
        "partial cancellation", "cancel one night", "remove one night",
        "cancel specific night", "cancel the fourth night",
        "entire booking cancelled", "full booking cancelled when",
        "just wanted to remove", "cancel second night",
        "modify booking", "shorten stay", "reduce nights",
    ],
    "language_barrier": [
        "language barrier", "hard to understand", "couldn't understand",
        "not us based", "offshore support", "communication difficulty",
        "accent issue", "language issue", "couldn't follow",
        "repeat multiple times", "had to repeat myself",
        "did not understand", "basic question not understood",
    ],
    "sales_tactics": [
        "sales pitch", "bait and switch", "bait-and-switch",
        "pushy promotion", "unwanted upsell", "pushy sales",
        "tried to sell me", "felt pressured", "promotional pitch",
        "just a sales call", "upsell attempt", "cross sell",
        "aggressive promotion", "forced to listen to promotion",
    ],
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

_neg_kw_pattern = build_neg_kw_pattern(HILTON_NEGATIVE_KEYWORDS)

# ── Compiled text-cleaning regexes ────────────────────────────────────────────
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
    "]+", flags=re.UNICODE,
)

_SPANISH_INDICATORS = {
    "tengo", "nada", "mas", "que", "agregar", "muy", "esta", "dia",
    "gracias", "como", "mejorar", "estoy", "trabajo", "comentarios",
}
_ENGLISH_INDICATORS = {
    "the", "is", "are", "was", "were", "have", "has", "had", "will",
    "i", "you", "he", "she", "it", "we", "they", "not", "all",
}

_TRANSLATION_CONFIDENCE_THRESHOLD = 0.7
_MIN_TRANSLATION_LENGTH = 3


def hilton_clean_text(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
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


def hilton_detect_language(text):
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


def hilton_smart_translate(text, lang_code, confidence):
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


def _hilton_hybrid_score(text, analyzer):
    """VADER(40%) + AFINN(20%) + TextBlob(20%) + Keywords(20%), adaptive weights."""
    if not text or not text.strip():
        return 0.0, 0.0, {}

    try:
        vs = analyzer.polarity_scores(text)
        vader_scaled = vs["compound"] * 10
        vader_conf = max(vs["pos"], vs["neg"], vs["neu"])
    except Exception:
        vader_scaled, vader_conf = 0.0, 0.0

    if _AFINN_OK:
        try:
            afinn_scaled = max(-10.0, min(10.0, _af.score(text) * 2.0))
        except Exception:
            afinn_scaled = 0.0
    else:
        afinn_scaled = vader_scaled * 0.5

    if _TEXTBLOB_OK:
        try:
            tb_scaled = _TextBlob(text).sentiment.polarity * 10
        except Exception:
            tb_scaled = 0.0
    else:
        tb_scaled = vader_scaled * 0.5

    kw_score, kw_total = _hilton_keyword_score(text)

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

    scores = [vader_scaled, afinn_scaled, tb_scaled, kw_score]
    nz = [s for s in scores if abs(s) > 0.1]
    agree_conf = max(0.0, 1.0 - (np.std(nz) / 10.0)) if len(nz) >= 2 else 0.5
    conf = vader_conf * 0.6 + agree_conf * 0.4
    if kw_total >= 2:
        conf = min(1.0, conf + 0.15)
    conf = max(0.3, min(0.95, conf))

    return combined, conf, {}


def classify_sentiment_hilton(score):
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


def _process_hilton_row(args):
    """Process a single Hilton row — safe to run in a thread."""
    i, norm_id, text, validation_dict, analyzer = args
    result = {
        "i": i, "compound": 0.0, "confidence": 0.0,
        "sentiment": "Neutral", "source": "Model",
        "translated": "", "rule": "blank",
    }

    if norm_id in validation_dict:
        label = validation_dict[norm_id]
        result.update({
            "compound":   SENTIMENT_TO_SCORE.get(label, 0.0) * 10,
            "confidence": 1.0,
            "sentiment":  label,
            "source":     "Validation",
            "rule":       "validation",
        })
        return result

    is_mless, _ = hilton_is_meaningless(text)
    if is_mless or not text:
        return result

    lang, lang_conf = hilton_detect_language(text)
    text_for_scoring = text
    if lang != "en":
        translated, did_translate = hilton_smart_translate(text, lang, lang_conf)
        if did_translate:
            text_for_scoring = translated
            result["translated"] = translated

    score, conf, _ = _hilton_hybrid_score(text_for_scoring, analyzer)
    result.update({
        "compound":   round(score, 3),
        "confidence": round(conf, 3),
        "sentiment":  classify_sentiment_hilton(score),
        "rule":       "hybrid",
    })
    return result


def run_hilton_analysis(df, id_col, text_col, validation_dict, progress_cb=None):
    missing_pkgs = [
        name for name, ok in [
            ("afinn", _AFINN_OK), ("textblob", _TEXTBLOB_OK),
            ("langdetect", _LANGDETECT_OK), ("deep-translator", _TRANSLATOR_OK),
        ] if not ok
    ]
    if missing_pkgs:
        st.info(f"ℹ️ Optional packages not found: `{', '.join(missing_pkgs)}` — running with available engines only.")

    # ── Step 1: Polars preprocessing ─────────────────────────────────────────
    pldf = pl.from_pandas(df)
    pldf = pldf.with_columns(
        pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_id_norm"),
    ).with_columns(
        pl.col(text_col).map_elements(hilton_clean_text, return_dtype=pl.String).alias("Cleaned_Comments"),
    )

    n        = len(pldf)
    id_norms = pldf["_id_norm"].to_list()
    cleaned  = pldf["Cleaned_Comments"].to_list()

    # ── Step 2: ThreadPoolExecutor for hybrid scoring (CPU-bound) ─────────────
    compounds        = np.zeros(n, dtype=np.float64)
    confidences      = np.zeros(n, dtype=np.float64)
    sentiments       = np.full(n, "Neutral", dtype=object)
    sources          = np.full(n, "Model",   dtype=object)
    translated_texts = np.full(n, "",        dtype=object)
    rules_fired      = np.full(n, "blank",   dtype=object)

    analyzer   = load_vader()  # load on main thread — avoids ScriptRunContext warning in workers
    args_list  = [(i, nid, text, validation_dict, analyzer)
                  for i, (nid, text) in enumerate(zip(id_norms, cleaned))]
    done_count = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_process_hilton_row, args): args[0]
                   for args in args_list}
        for future in as_completed(futures):
            res = future.result()
            i = res["i"]
            compounds[i]        = res["compound"]
            confidences[i]      = res["confidence"]
            sentiments[i]       = res["sentiment"]
            sources[i]          = res["source"]
            translated_texts[i] = res["translated"]
            rules_fired[i]      = res["rule"]
            done_count += 1
            if progress_cb and done_count % max(1, n // 100) == 0:
                progress_cb(done_count, n)

    if progress_cb:
        progress_cb(n, n)

    # ── Step 3: Vectorized post-processing in Polars ──────────────────────────
    text_for_analysis = [t if t else c for t, c in zip(translated_texts, cleaned)]

    result_pldf = pldf.with_columns([
        pl.Series("Translated_Text",   translated_texts.tolist()),
        pl.Series("Text_For_Analysis", text_for_analysis),
        pl.Series("consumer_score",    compounds),
        pl.Series("confidence",        confidences),
        pl.Series("consumer_sentiment", sentiments.tolist()),
        pl.Series("validation_source", sources.tolist()),
        pl.Series("_rule_fired",       rules_fired.tolist()),
        pl.lit(float("nan")).alias("_raw_vader"),
    ])

    # Keyword extraction (parallel map_elements)
    neg_kws = pl.Series(text_for_analysis).map_elements(
        lambda t: extract_negative_keywords(t, _neg_kw_pattern), return_dtype=pl.String
    ).to_list()

    result_df = result_pldf.to_pandas()
    result_df["Negative_Keywords"] = neg_kws
    return result_df
