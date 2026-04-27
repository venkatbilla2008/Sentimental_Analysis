"""
Spotify (Music Streaming) domain logic.
"""
import re

import numpy as np
import pandas as pd
import polars as pl

from .shared import (
    SHARED_NEGATIVE_KEYWORDS, SENTIMENT_TO_SCORE,
    load_vader, _load_bert_optional, classify_sentiment, classify_sentiment_expr,
    get_vader_compound,
    build_neg_kw_pattern, extract_negative_keywords,
)

# ── Spotify negative keywords (subset of shared, with service_issues rename) ─
SPOTIFY_NEGATIVE_KEYWORDS = {
    "payment_issues": [
        "can't make payment", "cannot make payment", "refund not",
        "billing", "overcharged", "charged incorrectly",
    ],
    "access_issues": [
        "not working", "doesn't work", "login", "sign in",
        "locked out", "account locked",
    ],
    "frustration": SHARED_NEGATIVE_KEYWORDS["frustration"],
    "service_issues": [
        "customer service", "no help", "not resolved",
        "complaint", "delay", "waiting too long",
        "difficult", "unwilling", "policies aren't",
    ],
    "general_negative": [
        "problem", "issue", "cancel", "still can't",
        "how much longer",
    ],
    "student_plan": [
        "student account", "student plan", "student pricing",
        "education discount", "student discount",
        "price increased from", "student verification failed",
        "student rate not applied", "no longer student rate",
        "student membership", "academic plan",
    ],
    "artist_account": [
        "artist page claim", "can't claim artist page",
        "claim my artist profile", "artist verification",
        "music distribution", "getting music on spotify",
        "submit music", "artist access denied",
        "spotify for artists", "artist dashboard",
        "my artist profile", "artist account issue",
    ],
    "payment_method_switch": [
        "charge different card", "change payment method",
        "update card details", "change my card",
        "switch payment", "different credit card",
        "add new card", "remove old card", "payment card update",
        "change billing card", "update billing method",
    ],
    "cancellation_timing": [
        "cancel premium", "when does cancellation take effect",
        "downgrade confusion", "cancellation not applied",
        "still being charged after cancel", "cancel effective date",
        "when will cancellation apply", "cancellation pending",
        "still subscribed after cancel", "downgrade to free",
    ],
}
_neg_kw_pattern = build_neg_kw_pattern(SPOTIFY_NEGATIVE_KEYWORDS)

# ── Trigger patterns ──────────────────────────────────────────────────────────
_NEG_TRIGGERS = [
    "not happy", "difficult", "still can't", "can't make payment",
    "cannot make payment", "how much longer", "unwilling", "problem",
    "issue", "refund not", "angry", "complaint",
    "bad", "poor", "horrible", "terrible", "customer service", "no help",
    "not satisfied", "disappointed", "policies aren't", "not resolved",
    "not working", "doesn't work", "worse", "delay", "waiting too long",
    "frustrated", "cancel", "unacceptable",
]
_NEU_TRIGGERS = [
    "want to talk", "continue conversation", "verify my", "reach out",
    "need my spotify", "staff was friendly", "my account is reverted",
    "I love Spotify", "label user", "no decline", "no billing",
    "ad is coming", "i want a refund", "refund", "switch", "changed",
    "changed my card", "i want to reach", "check my account",
    "contact support", "login", "sign in", "update account",
    "different subscription",
]


_SP_NEG_RE = "(?i)" + "|".join(re.escape(p) for p in _NEG_TRIGGERS)
_SP_NEU_RE = "(?i)" + "|".join(re.escape(p) for p in _NEU_TRIGGERS)

_LANGDETECT_OK = False
try:
    from langdetect import detect as _detect
    _LANGDETECT_OK = True
except ImportError:
    pass


def _extract_consumer(text):
    if isinstance(text, str):
        parts = text.split("|")
        msgs = [p.split("Consumer:")[1].strip() for p in parts if "Consumer:" in p]
        return " ".join(msgs).strip()
    return ""


def _detect_language(text):
    if not isinstance(text, str) or not text.strip():
        return "non-en"
    if not _LANGDETECT_OK:
        return "en"
    try:
        return "en" if _detect(text) == "en" else "non-en"
    except Exception:
        return "non-en"


def run_spotify_analysis(df, id_col, text_col, validation_dict, progress_cb=None):
    """
    1. Extract Consumer messages from pipe-delimited transcripts
    2. Language detection (English only scored)
    3. VADER scoring with trigger overrides
    4. BERT correction for borderline scores (-0.1 to +0.1)
    5. Final trigger re-enforcement after BERT (vectorized, non-validated rows only)
    """
    analyzer = load_vader()

    # ── Step 1: Polars preprocessing ─────────────────────────────────────────
    pldf = pl.from_pandas(df)

    pldf = pldf.with_columns(
        pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_id_norm"),
    ).with_columns(
        pl.col(text_col).map_elements(_extract_consumer, return_dtype=pl.String).alias("ConsumerOnly"),
    ).with_columns(
        pl.col("ConsumerOnly").map_elements(_detect_language, return_dtype=pl.String).alias("Language"),
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
    tc = pl.col("ConsumerOnly")
    pldf = pldf.with_columns([
        (tc.str.len_bytes() == 0).alias("_blank"),
        pl.col("Language").eq("en").alias("_is_en"),
        tc.str.contains(_SP_NEG_RE).alias("_m_neg"),
        tc.str.contains(_SP_NEU_RE).alias("_m_neu"),
    ])

    # ── Step 4: Rule assignment (vectorized) ──────────────────────────────────
    is_val      = pl.col("_val_label").is_not_null()
    skipped     = pl.col("_blank") | ~pl.col("_is_en")

    pldf = pldf.with_columns([
        pl.when(is_val).then(
            pl.col("_val_label").map_elements(
                lambda x: SENTIMENT_TO_SCORE.get(x, 0.0), return_dtype=pl.Float64
            )
        )
        .when(skipped).then(pl.lit(None))  # null → empty sentiment for non-English/blank
        .when(pl.col("_m_neg")).then(pl.lit(-0.6))
        .when(pl.col("_m_neu")).then(pl.lit(0.0))
        .otherwise(pl.lit(None))  # VADER needed
        .cast(pl.Float64).alias("_rule_score"),

        pl.when(is_val).then(pl.lit("validation"))
        .when(pl.col("_blank")).then(pl.lit("blank"))
        .when(~pl.col("_is_en")).then(pl.lit("non_english"))
        .when(pl.col("_m_neg")).then(pl.lit("trigger:negative"))
        .when(pl.col("_m_neu")).then(pl.lit("trigger:neutral"))
        .otherwise(pl.lit("vader"))
        .alias("_rule_fired"),

        pl.when(is_val).then(pl.lit("Validation")).otherwise(pl.lit("Model")).alias("_source"),
    ])

    if progress_cb:
        progress_cb(len(pldf) // 2, len(pldf))

    # ── Step 5: Batch VADER only for rows that need it ────────────────────────
    texts_list  = pldf["ConsumerOnly"].fill_null("").to_list()
    # needs_vader = rule_score is null AND the row is NOT skipped (skipped rows stay null)
    is_skipped  = (pldf["_blank"] | ~pldf["_is_en"]).to_list()
    needs_vader = [
        rs is None and not sk
        for rs, sk in zip(pldf["_rule_score"].to_list(), is_skipped)
    ]

    # Use Python list with None so Polars stores actual nulls (not float NaN)
    vader_raw_list: list = [None] * len(pldf)
    for i, (need, text) in enumerate(zip(needs_vader, texts_list)):
        if need and text.strip():
            v = get_vader_compound(text, analyzer)
            vader_raw_list[i] = 0.0 if abs(v) < 0.05 else round(v, 3)

    pldf = pldf.with_columns(
        pl.Series("_vader_raw", vader_raw_list, dtype=pl.Float64)
    )

    # ── Step 6: Final score (vectorized) ─────────────────────────────────────
    # skipped rows (non-English/blank) keep null → pandas NaN → displayed as empty
    pldf = pldf.with_columns(
        pl.when(pl.col("_rule_score").is_not_null())
            .then(pl.col("_rule_score"))
        .when(pl.col("_vader_raw").is_not_null())
            .then(pl.col("_vader_raw"))
        .otherwise(pl.lit(None).cast(pl.Float64))
        .round(3).alias("consumer_score")
    ).with_columns(
        pl.when(is_val).then(pl.lit(1.0))
        .when(skipped).then(pl.lit(0.0))
        .otherwise(pl.lit(0.7))
        .alias("confidence")
    ).with_columns(
        # null score → empty string (matches Colab behaviour for non-English rows)
        pl.when(pl.col("consumer_score").is_null()).then(pl.lit(""))
        .otherwise(classify_sentiment_expr("consumer_score"))
        .alias("consumer_sentiment")
    )

    # ── Step 7: BERT correction (borderline, non-validated English) ───────────
    result_df = pldf.to_pandas()
    _bert = _load_bert_optional()
    if _bert is not None:
        border_mask = (
            result_df["consumer_score"].between(-0.1, 0.1)
            & result_df["Language"].eq("en")
            & result_df["_source"].eq("Model")
        )
        if border_mask.any():
            border_idxs  = list(result_df.index[border_mask])
            texts_border = [texts_list[i] for i in border_idxs]
            try:
                bert_results = _bert(texts_border, batch_size=16, truncation=True)
                lbls   = np.array([r["label"].lower() for r in bert_results])
                bconfs = np.array([r["score"]         for r in bert_results])
                result_df.loc[border_idxs, "consumer_sentiment"] = np.where(lbls == "positive", "Positive", "Negative")
                result_df.loc[border_idxs, "consumer_score"]     = np.where(lbls == "positive", bconfs, -bconfs).round(3)
                result_df.loc[border_idxs, "confidence"]         = bconfs.round(3)
                result_df.loc[border_idxs, "_rule_fired"]        = ["bert:" + l for l in lbls]
            except Exception:
                pass

    # ── Step 8: Final trigger re-enforcement (vectorized, non-validated) ──────
    non_val   = (result_df["_source"] == "Model") & result_df["Language"].eq("en")
    _texts_pl = pl.Series(texts_list)
    _neg_hits = _texts_pl.str.contains(_SP_NEG_RE).to_numpy()
    _neu_hits = _texts_pl.str.contains(_SP_NEU_RE).to_numpy()
    neg_mask  = non_val & _neg_hits
    neu_mask  = non_val & ~_neg_hits & _neu_hits
    result_df.loc[neg_mask, "consumer_score"]     = -0.6
    result_df.loc[neg_mask, "consumer_sentiment"] = "Negative"
    result_df.loc[neg_mask, "_rule_fired"]        = "trigger:negative"
    result_df.loc[neu_mask, "consumer_score"]     = 0.0
    result_df.loc[neu_mask, "consumer_sentiment"] = "Neutral"
    result_df.loc[neu_mask, "_rule_fired"]        = "trigger:neutral"

    if progress_cb:
        progress_cb(len(pldf), len(pldf))

    # ── Step 9: Keyword extraction (parallel) ─────────────────────────────────
    neg_kws = pl.Series(texts_list).map_elements(
        lambda t: extract_negative_keywords(t, _neg_kw_pattern), return_dtype=pl.String
    ).to_list()

    result_df["Text_For_Analysis"] = texts_list
    result_df["_id_norm"]          = pldf["_id_norm"].to_list()
    result_df["validation_source"] = pldf["_source"].to_list()
    result_df["_raw_vader"]        = vader_raw_list
    result_df["Negative_Keywords"] = neg_kws

    drop_cols = [c for c in result_df.columns if c.startswith("_m_") or c in
                 ("_blank", "_is_en", "_rule_score", "_val_label", "_vader_raw", "_source")]
    result_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    return result_df
