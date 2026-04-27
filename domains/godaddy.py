"""
GoDaddy (Domain & Hosting) domain logic.
"""
import re

import numpy as np
import polars as pl

from .shared import (
    SHARED_NEGATIVE_KEYWORDS, SENTIMENT_TO_SCORE,
    load_vader, _load_bert_optional, classify_sentiment_expr,
    get_vader_compound, run_vader_parallel,
    build_neg_kw_pattern, extract_negative_keywords,
    has_resolution_or_thanks, is_polite_request,
)

# ── GoDaddy-specific keyword categories ───────────────────────────────────────
_GODADDY_EXTRA_KEYWORDS = {
    "dns_configuration": [
        "dns locked", "locked dns", "can't change dns", "cannot change dns",
        "dns settings locked", "a record locked", "locked a records",
        "nameserver", "name server", "dns record", "multiple a records",
        "a record issue", "cname issue", "mx record", "dns not updating",
        "dns propagation", "dns not working", "dns configuration",
        "change nameserver", "point domain", "domain pointing",
        "dns entry", "remove a record", "dns setup failed",
    ],
    "domain_visibility": [
        "domain not showing", "site not visible", "domain not working",
        "wrong page showing", "domain not displaying", "website not showing",
        "domain not loading", "site not loading on domain",
        "domain not connected", "site is not visible",
        "i paid for domain but not work", "domain not live",
        "website not live", "page not showing", "domain not resolving",
        "404 on my domain", "domain not pointed correctly",
        "connected domain but not showing",
    ],
    "email_setup": [
        "cannot send email", "cannot receive email", "can't send email",
        "can't receive email", "email not working", "email setup failed",
        "sending mail failing", "receiving mail failing",
        "email not delivering", "email bouncing", "smtp issue",
        "imap issue", "email configuration", "setup email",
        "professional email not working", "workspace email issue",
        "email account not set up", "microsoft 365 email",
        "email forwarding not working",
    ],
    "auto_renewal": [
        "auto renew", "auto-renewal", "domain renewal charge",
        "unexpected renewal", "refund renewal", "renewal charge",
        "auto renewed without permission", "auto renew was off",
        "charged for renewal", "renewal surprise",
        "didn't want to renew domain", "auto renewal disabled but charged",
        "renewal refund", "cancel renewal",
    ],
    "account_blocked": [
        "account locked", "account blocked", "blocked from using",
        "locked method", "account suspended", "access denied",
        "can't log in to account", "login blocked",
        "account disabled", "restricted account",
        "temporarily locked", "account flagged",
    ],
    "otp_auth_failure": [
        "otp issue", "otp not received", "code not received",
        "didn't receive code", "verification code not received",
        "authenticator not working", "sms code issue",
        "two factor issue", "2fa not working", "authentication failed",
        "can't get verification code", "code expired",
        "backup code not working", "authenticator app issue",
        "not receiving sms", "phone verification failed",
    ],
    "delegate_access": [
        "delegate access", "delegate page", "delegate access not working",
        "can't access delegate page", "delegate access denied",
        "delegation issue", "account delegation",
        "delegate login", "manage delegate",
    ],
    "wordpress_builder": [
        "wordpress", "wordpress media upload", "media upload failing",
        "wordpress installation", "install wordpress",
        "wordpress error", "airo website", "airo builder",
        "website builder error", "builder not working",
        "managed wordpress", "wp admin issue",
        "wordpress plugin", "wp site not loading",
        "website builder issue", "drag and drop not working",
    ],
    "ssl_certificate": [
        "ssl not working", "ssl certificate", "https not working",
        "certificate expired", "ssl error", "not secure",
        "certificate issue", "ssl installation failed",
        "mixed content", "ssl renewal", "https redirect not working",
    ],
    "domain_transfer": [
        "transfer domain", "domain transfer failed",
        "transfer out", "transfer in", "domain locked for transfer",
        "transfer authorization", "epp code", "auth code",
        "transfer rejected", "domain not transferred",
        "transfer pending", "unlock domain for transfer",
    ],
}

GODADDY_NEGATIVE_KEYWORDS = {**SHARED_NEGATIVE_KEYWORDS, **_GODADDY_EXTRA_KEYWORDS}
_neg_kw_pattern = build_neg_kw_pattern(GODADDY_NEGATIVE_KEYWORDS)

# ── Trigger patterns ──────────────────────────────────────────────────────────
_NEG_TRIGGERS = [
    "dns locked", "locked a records", "can't change dns", "domain not showing",
    "domain not working", "site not visible", "cannot send email",
    "cannot receive email", "email not working", "auto renew",
    "unexpected renewal", "account locked", "account blocked",
    "otp not received", "code not received", "authenticator not working",
    "wordpress error", "media upload failing", "ssl not working",
    "certificate expired", "transfer domain", "transfer failed",
    "not working", "doesn't work", "problem", "issue",
    "frustrated", "angry", "terrible", "worst", "horrible",
    "no help", "not resolved", "waste of time",
]
_NEU_TRIGGERS = [
    "how do i", "how can i", "can you help", "need help",
    "want to transfer", "want to cancel", "want to renew",
    "check my account", "verify my account", "update my",
    "change my", "set up", "configure", "connect my domain",
    "point my domain", "install", "how to setup",
]

_GD_NEG_RE = "(?i)" + "|".join(re.escape(p) for p in _NEG_TRIGGERS)
_GD_NEU_RE = "(?i)" + "|".join(re.escape(p) for p in _NEU_TRIGGERS)
_GD_STRONG_RE = "(?i)" + "|".join(re.escape(p) for p in [
    "dns locked", "account locked", "account blocked",
    "cannot receive email", "cannot send email",
    "domain not working", "ssl not working", "transfer failed",
])
_GD_RES_RE = (
    r"(?i)thanks|thank you|thank u|\bty\b|okay|\bok\b|perfect|"
    r"got it|understood|appreciate|sounds good|all set|all good"
)


def _extract_customer(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # GoDaddy chats are pipe-delimited like Spotify/Netflix
    if "|" in text and ("Consumer:" in text or "Customer:" in text):
        parts = text.split("|")
        msgs = []
        for p in parts:
            for marker in ("Consumer:", "Customer:"):
                if marker in p:
                    msgs.append(p.split(marker)[1].strip())
                    break
        return " ".join(msgs).strip()
    # Plain text fallback
    return text.strip()


def _clean(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 5 or len(text.split()) < 2:
        return ""
    return text


def run_godaddy_analysis(df, id_col, text_col, validation_dict,
                         progress_cb=None, rule_threshold=0.7):
    analyzer = load_vader()

    # ── Step 1: Polars preprocessing ─────────────────────────────────────────
    pldf = pl.from_pandas(df)

    pldf = pldf.with_columns(
        pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_id_norm"),
    ).with_columns(
        pl.col(text_col).map_elements(_extract_customer, return_dtype=pl.String).alias("CustomerOnly"),
    ).with_columns(
        pl.col("CustomerOnly").map_elements(_clean, return_dtype=pl.String).alias("Text_For_Analysis"),
    )

    # ── Step 2: Validation lookup ─────────────────────────────────────────────
    pldf = pldf.with_columns(
        pl.col("_id_norm").map_elements(
            lambda x: validation_dict.get(x), return_dtype=pl.String
        ).alias("_val_label")
    )

    if progress_cb:
        progress_cb(len(pldf) // 4, len(pldf))

    # ── Step 3: Pattern-match columns ────────────────────────────────────────
    tc = pl.col("Text_For_Analysis")
    pldf = pldf.with_columns([
        (tc.str.len_bytes() == 0).alias("_blank"),
        tc.str.contains(_GD_NEG_RE).alias("_m_neg"),
        tc.str.contains(_GD_NEU_RE).alias("_m_neu"),
        tc.str.contains(_GD_STRONG_RE).alias("_m_strong_neg"),
        tc.str.contains(_GD_RES_RE).alias("_m_resolution"),
    ])

    # ── Step 4: Rule score/confidence/label ───────────────────────────────────
    is_val = pl.col("_val_label").is_not_null()
    blank  = pl.col("_blank")

    pldf = pldf.with_columns([
        pl.when(is_val).then(
            pl.col("_val_label").map_elements(
                lambda x: SENTIMENT_TO_SCORE.get(x, 0.0), return_dtype=pl.Float64
            )
        )
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(-0.70))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.0))
        .when(pl.col("_m_neg")).then(pl.lit(-0.55))
        .when(pl.col("_m_neu")).then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .cast(pl.Float64).alias("_rule_score"),

        pl.when(is_val).then(pl.lit(1.0))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(0.95))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.75))
        .when(pl.col("_m_neg")).then(pl.lit(0.80))
        .when(pl.col("_m_neu")).then(pl.lit(0.80))
        .otherwise(pl.lit(0.5))
        .cast(pl.Float64).alias("_rule_conf"),

        pl.when(is_val).then(pl.lit("validation"))
        .when(blank).then(pl.lit("blank"))
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

    # ── Step 5: Parallel VADER for rows that need it ──────────────────────────
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

    # ── Step 6: Final score/sentiment ─────────────────────────────────────────
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

    # ── Step 8: Keyword extraction ────────────────────────────────────────────
    neg_kws = pl.Series(texts_list).map_elements(
        lambda t: extract_negative_keywords(t, _neg_kw_pattern), return_dtype=pl.String
    ).to_list()

    result_df["CustomerOnly"]      = pldf["CustomerOnly"].to_list()
    result_df["Text_For_Analysis"] = pldf["Text_For_Analysis"].to_list()
    result_df["_id_norm"]          = pldf["_id_norm"].to_list()
    result_df["validation_source"] = pldf["_source"].to_list()
    result_df["_raw_vader"]        = vader_raw
    result_df["Negative_Keywords"] = neg_kws

    drop_cols = [c for c in result_df.columns if c.startswith("_m_") or c in
                 ("_blank", "_rule_score", "_rule_conf", "_val_label", "_vader_raw", "_vader_clamped", "_source")]
    result_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    return result_df
