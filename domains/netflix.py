"""
Netflix (Streaming & Entertainment) domain logic.
"""
import re

import numpy as np
import pandas as pd
import polars as pl

from .shared import (
    SHARED_NEGATIVE_KEYWORDS, SENTIMENT_TO_SCORE,
    load_vader, _load_bert_optional, classify_sentiment, classify_sentiment_expr,
    get_vader_compound, run_vader_parallel,
    build_neg_kw_pattern, extract_negative_keywords,
    has_resolution_or_thanks, is_polite_request,
)

# ── Netflix-specific keyword extensions on top of shared ─────────────────────
_NETFLIX_EXTRA_KEYWORDS = {
    "unwanted_actions": [
        "i do not want this membership",
    ],
    "account_verification": [
        "cannot verify account", "can't verify account",
        "otp not received", "code not received", "didn't receive code",
        "email link not received", "verification email not received",
        "mfa not working", "two factor not working", "2fa issue",
        "change email", "update email", "change my email",
        "email not working", "cannot change email",
        "phone number not accepted", "number not verified",
    ],
    "geographic_access": [
        "moved to", "relocated to", "can't access from",
        "service not available in", "region locked", "not available in my country",
        "geo restricted", "country not supported", "blocked in my region",
        "different country", "international access", "vpn required",
    ],
    "security_block": [
        "multiple attempts to make changes", "account locked for security",
        "unable to process modifications", "security hold",
        "flagged for security", "suspicious activity",
        "account under review", "blocked for security",
        "security verification failed", "locked due to",
    ],
    "plan_change": [
        "change plan", "switch plan", "change to premium",
        "change to standard", "tier upgrade failed",
        "upgrade failed", "downgrade failed", "plan not updated",
        "why can't i subscribe", "subscription not changing",
        "plan change not applied", "still on old plan",
    ],
}

NETFLIX_NEGATIVE_KEYWORDS = {
    cat: list(kws) + _NETFLIX_EXTRA_KEYWORDS.get(cat, [])
    for cat, kws in SHARED_NEGATIVE_KEYWORDS.items()
}
# Add Netflix-only categories not in shared
for _cat, _kws in _NETFLIX_EXTRA_KEYWORDS.items():
    if _cat not in NETFLIX_NEGATIVE_KEYWORDS:
        NETFLIX_NEGATIVE_KEYWORDS[_cat] = _kws
_neg_kw_pattern = build_neg_kw_pattern(NETFLIX_NEGATIVE_KEYWORDS)

# ── Netflix content titles to strip ──────────────────────────────────────────
NETFLIX_CONTENT_SET = {
    "stranger things", "the crown", "bridgerton", "squid game", "money heist",
    "ozark", "the witcher", "narcos", "breaking bad", "better call saul",
    "black mirror", "the umbrella academy", "cobra kai", "prison break",
    "you", "wednesday", "outer banks", "lucifer", "peaky blinders", "dark",
    "mindhunter", "the sinner", "how to get away with murder", "criminal minds",
    "the blacklist", "dexter", "bloodline", "kissing booth",
    "bon appetite", "bon appetit",
}

# Precompile all content-removal patterns once
_CONTENT_PATTERNS = [
    (c.lower(), re.compile(re.escape(c), re.IGNORECASE))
    for c in NETFLIX_CONTENT_SET
]

_REMOVE_RE = re.compile(
    r"chat with an agent|chat with agent|speak to agent|talk to agent|"
    r"connect me to agent|agent chat|chat to agent|agent help|help chat",
    re.IGNORECASE,
)

# ── Trigger patterns ──────────────────────────────────────────────────────────
_NEG_TRIGGERS = [
    "still have not received", "have not received", "not received",
    "money is debited", "money was debited", "money was withdrawn",
    "without my permission", "without permission",
    "i do not want this membership", "did not want to renew",
    "did not want to restart",
    "thanks for nothing", "notifications are getting stuck", "getting stuck",
    "unable to do payment", "payment is pending",
    "frustrating", "frustrated", "annoyed", "angry",
    "not working properly", "doesn't work", "cant access", "cannot access",
    "hacked my account", "compromised account",
    "charged double", "being charged double",
    "unforeseen circumstances",
    "you dont want to help", "you don't want to help",
]
_NEU_TRIGGERS = [
    "want my refund back", "please refund", "need a refund",
    "issue a refund", "can you refund",
    "change email", "change my email", "update email", "switch to a different email",
    "change the email address", "need to change my email",
    "change my card", "change card details", "update payment",
    "remove payment method", "verify my account with a credit card",
    "change from premium to standard", "change plan", "switch plan",
    "automatically changed", "change from standard to premium",
    "how to", "how can i", "can i get", "is there any discount",
    "questions about", "just wondering", "need to add",
    "how to change", "help finding", "unable to access",
    "device is not part", "can not access from",
    "just canceled", "want to cancel", "can you cancel",
    "verify my account", "finding current password",
]
_POS_TRIGGERS = [
    "thank you", "thanks", "appreciate", "grateful",
    "perfect", "helpful", "great", "excellent", "awesome",
    "super helpful", "very helpful", "wonderful",
    "thanks for the quick reply", "appreciate you looking into",
    "thanks for clarifying", "thanks for explaining",
]
_VERY_POS_TRIGGERS = [
    "bon appetite", "bon appetit",
    "amazing", "outstanding", "exceeded expectations",
    "extremely helpful", "incredibly helpful", "absolutely perfect",
]
_STRONG_NEG_PHRASES = [
    "without my permission", "do not want this membership",
    "did not want to renew", "thanks for nothing",
    "unable to do payment", "still have not received",
]


# Polars-compatible pattern strings (case-insensitive prefix)
_NF_NEG_RE   = "(?i)" + "|".join(re.escape(p) for p in _NEG_TRIGGERS)
_NF_NEU_RE   = "(?i)" + "|".join(re.escape(p) for p in _NEU_TRIGGERS)
_NF_POS_RE   = "(?i)" + "|".join(re.escape(p) for p in _POS_TRIGGERS)
_NF_VPOS_RE  = "(?i)" + "|".join(re.escape(p) for p in _VERY_POS_TRIGGERS)
_NF_STRONG_RE = "(?i)" + "|".join(re.escape(p) for p in _STRONG_NEG_PHRASES)
_NF_RES_RE   = (
    r"(?i)thanks|thank you|thank u|\bty\b|okay|\bok\b|perfect|"
    r"got it|understood|appreciate|sounds good|all set|all good"
)
_NF_POS_WORD_RE = r"(?i)\b(thank|thanks|appreciate|perfect|helpful)\b"


def _extract_customer(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    segs = re.findall(
        r"\[\d{2}:\d{2}:\d{2}\s*CUSTOMER\]:\s*(.*?)(?=\[\d{2}:\d{2}:\d{2}\s*\w+\]:|$)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    return " ".join(re.sub(r"\s+", " ", s).strip() for s in segs if s.strip())


def _clean(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _REMOVE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 5 or len(text.split()) < 2:
        return ""
    return text


def _remove_content_names(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text_lower = text.lower()
    cleaned = text
    for content_lower, pattern in _CONTENT_PATTERNS:
        if content_lower in text_lower:
            cleaned = pattern.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_by_rules(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, "blank"
    tl = text.lower()
    if _vpos_trig.search(text) or _pos_trig.search(text):
        pc = len(re.findall(r"\b(thank|thanks|appreciate|perfect|helpful)\b", tl))
        if pc >= 2 or _vpos_trig.search(text):
            return 0.85, 0.9, "very_positive"
        return 0.40, 0.8, "positive"
    if _neg_trig.search(text):
        if has_resolution_or_thanks(text):
            if len(_neg_trig.findall(text)) == 1 and any(w in tl[-50:] for w in ["thank", "thanks"]):
                return 0.0, 0.75, "neutral_resolved"
        if any(p in tl for p in _STRONG_NEG_PHRASES):
            return -0.65, 0.95, "strong_negative"
        return -0.60, 0.80, "negative"
    if _neu_trig.search(text):
        if is_polite_request(text):
            return 0.0, 0.90, "neutral_polite"
        if "refund" in tl and any(p in tl for p in ["please", "want", "can you", "need"]):
            return 0.0, 0.85, "neutral_refund"
        return 0.0, 0.80, "neutral"
    return None, 0.0, None


def run_netflix_analysis(df, id_col, text_col, validation_dict,
                         progress_cb=None, rule_threshold=0.7):
    analyzer = load_vader()

    # ── Step 1: Polars preprocessing ─────────────────────────────────────────
    pldf = pl.from_pandas(df)

    pldf = pldf.with_columns(
        pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_id_norm"),
    ).with_columns(
        pl.col(text_col).map_elements(_extract_customer, return_dtype=pl.String).alias("CustomerOnly"),
    ).with_columns(
        pl.col("CustomerOnly").map_elements(
            lambda t: _remove_content_names(_clean(t)), return_dtype=pl.String
        ).alias("Text_For_Analysis"),
    )

    # ── Step 2: Validation lookup (vectorized) ────────────────────────────────
    pldf = pldf.with_columns(
        pl.col("_id_norm").map_elements(
            lambda x: validation_dict.get(x), return_dtype=pl.String
        ).alias("_val_label")
    )

    if progress_cb:
        progress_cb(len(pldf) // 4, len(pldf))

    # ── Step 3: Pattern-match columns (parallel in Polars) ────────────────────
    tc = pl.col("Text_For_Analysis")
    pldf = pldf.with_columns([
        (tc.str.len_bytes() == 0).alias("_blank"),
        tc.str.contains(_NF_VPOS_RE).alias("_m_vpos"),
        tc.str.contains(_NF_POS_RE).alias("_m_pos"),
        tc.str.contains(_NF_NEG_RE).alias("_m_neg"),
        tc.str.contains(_NF_NEU_RE).alias("_m_neu"),
        tc.str.contains(_NF_STRONG_RE).alias("_m_strong_neg"),
        tc.str.contains(_NF_RES_RE).alias("_m_resolution"),
        tc.str.count_matches(_NF_POS_WORD_RE).alias("_pos_word_count"),
    ])

    # ── Step 4: Rule score/confidence/label (vectorized when/then) ────────────
    is_val = pl.col("_val_label").is_not_null()
    blank  = pl.col("_blank")

    pldf = pldf.with_columns([
        pl.when(is_val).then(
            pl.col("_val_label").map_elements(
                lambda x: SENTIMENT_TO_SCORE.get(x, 0.0), return_dtype=pl.Float64
            )
        )
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_vpos")).then(pl.lit(0.85))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit(0.85))
        .when(pl.col("_m_pos")).then(pl.lit(0.40))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(-0.65))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.0))
        .when(pl.col("_m_neg")).then(pl.lit(-0.60))
        .when(pl.col("_m_neu")).then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .cast(pl.Float64).alias("_rule_score"),

        pl.when(is_val).then(pl.lit(1.0))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_vpos")).then(pl.lit(0.90))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit(0.90))
        .when(pl.col("_m_pos")).then(pl.lit(0.80))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(0.95))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.75))
        .when(pl.col("_m_neg")).then(pl.lit(0.80))
        .when(pl.col("_m_neu")).then(pl.lit(0.80))
        .otherwise(pl.lit(0.5))
        .cast(pl.Float64).alias("_rule_conf"),

        pl.when(is_val).then(pl.lit("validation"))
        .when(blank).then(pl.lit("blank"))
        .when(pl.col("_m_vpos")).then(pl.lit("rule:very_positive"))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit("rule:very_positive"))
        .when(pl.col("_m_pos")).then(pl.lit("rule:positive"))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit("rule:strong_negative"))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit("rule:neutral_resolved"))
        .when(pl.col("_m_neg")).then(pl.lit("rule:negative"))
        .when(pl.col("_m_neu")).then(pl.lit("rule:neutral"))
        .otherwise(pl.lit("vader"))
        .alias("_rule_fired"),

        pl.when(is_val).then(pl.lit("Validation")).otherwise(pl.lit("Model")).alias("_source"),
    ])

    if progress_cb:
        progress_cb(len(pldf) // 2, len(pldf))

    # ── Step 5: Batch VADER only for rows that need it ────────────────────────
    texts_list  = pldf["Text_For_Analysis"].fill_null("").to_list()
    needs_vader = (
        pldf["_rule_score"].is_null() |
        ((pldf["_rule_conf"] <= rule_threshold) & pldf["_val_label"].is_null() & ~pldf["_blank"])
    ).to_list()

    vader_raw = run_vader_parallel(texts_list, needs_vader, analyzer)

    pldf = pldf.with_columns(pl.Series("_vader_raw", vader_raw))
    pldf = pldf.with_columns(
        pl.when(pl.col("_vader_raw").abs() < 0.05).then(pl.lit(0.0))
        .otherwise(pl.col("_vader_raw")).alias("_vader_clamped")
    )

    # ── Step 6: Final score/sentiment (vectorized) ────────────────────────────
    pldf = pldf.with_columns(
        pl.when(is_val).then(pl.col("_rule_score"))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_rule_score").is_not_null() & (pl.col("_rule_conf") > rule_threshold))
            .then(pl.col("_rule_score"))
        .when(pl.col("_rule_score").is_not_null())
            .then(pl.col("_rule_score") * pl.col("_rule_conf") +
                  pl.col("_vader_clamped") * (1.0 - pl.col("_rule_conf")))
        .otherwise(pl.col("_vader_clamped"))
        .round(3).alias("consumer_score")
    ).with_columns(
        pl.when(is_val).then(pl.col("_rule_conf"))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_rule_score").is_not_null() & (pl.col("_rule_conf") > rule_threshold))
            .then(pl.col("_rule_conf"))
        .otherwise(pl.lit(0.5))
        .alias("confidence")
    ).with_columns(
        classify_sentiment_expr("consumer_score").alias("consumer_sentiment")
    )

    # ── Step 7: BERT correction ───────────────────────────────────────────────
    result_df = pldf.to_pandas()
    _bert = _load_bert_optional()
    if _bert is not None:
        low_mask = (result_df["confidence"] < 0.6) & (result_df["_source"] == "Model")
        idxs = [i for i in result_df.index[low_mask] if str(texts_list[i]).strip()]
        if idxs:
            try:
                bert_results = _bert([texts_list[i] for i in idxs], batch_size=16, truncation=True)
                lbls   = np.array([r["label"].lower() for r in bert_results])
                bconfs = np.array([r["score"]         for r in bert_results])
                high   = bconfs > 0.75
                if high.any():
                    hi_idx  = [idxs[j] for j in np.where(high)[0]]
                    hi_lbls = lbls[high]
                    hi_conf = bconfs[high]
                    result_df.loc[hi_idx, "consumer_score"]     = np.where(hi_lbls == "positive", 0.35, -0.35)
                    result_df.loc[hi_idx, "consumer_sentiment"] = np.where(hi_lbls == "positive", "Positive", "Negative")
                    result_df.loc[hi_idx, "confidence"]         = hi_conf.round(3)
                    result_df.loc[hi_idx, "_rule_fired"]        = ["bert:" + l for l in hi_lbls]
            except Exception:
                pass

    if progress_cb:
        progress_cb(len(pldf), len(pldf))

    # ── Step 8: Keyword extraction (parallel) ─────────────────────────────────
    neg_kws = pl.Series(texts_list).map_elements(
        lambda t: extract_negative_keywords(t, _neg_kw_pattern), return_dtype=pl.String
    ).to_list()

    result_df["CustomerOnly"]         = pldf["CustomerOnly"].to_list()
    result_df["CustomerOnly_Cleaned"] = pldf["Text_For_Analysis"].to_list()
    result_df["Text_For_Analysis"]    = pldf["Text_For_Analysis"].to_list()
    result_df["_id_norm"]             = pldf["_id_norm"].to_list()
    result_df["validation_source"]    = pldf["_source"].to_list()
    result_df["_rule_fired"]          = result_df.get("_rule_fired", pldf["_rule_fired"].to_list())
    result_df["_raw_vader"]           = vader_raw
    result_df["Negative_Keywords"]    = neg_kws

    drop_cols = [c for c in result_df.columns if c.startswith("_m_") or c in
                 ("_blank", "_pos_word_count", "_rule_score", "_rule_conf",
                  "_val_label", "_vader_raw", "_vader_clamped", "_source")]
    result_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    return result_df
