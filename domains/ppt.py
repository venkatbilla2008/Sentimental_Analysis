"""
PPT (Professional Physical Therapy) domain logic.
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
    has_resolution_or_thanks, is_polite_request, has_simple_cancellation,
)

# ── PPT-specific keyword extensions ──────────────────────────────────────────
_PPT_EXTRA_KEYWORDS = {
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
    "insurance_verification": [
        "are you in-network", "out-of-network", "in network",
        "do you accept", "accept my insurance", "verify my insurance",
        "insurance verification", "check my coverage", "coverage verification",
        "do i have visits left", "visits remaining", "how many visits",
        "insurance eligibility", "benefits check", "prior authorization",
        "pre-authorization", "referral required", "need a referral",
    ],
    "new_patient_intake": [
        "new patient", "first appointment", "first visit", "first time",
        "how do i sign up", "register as a patient", "intake form",
        "patient registration", "pricing of services", "how much does it cost",
        "i have a referral", "referral from my doctor",
        "what do i need to bring", "initial evaluation",
        "onboarding", "new client",
    ],
    "location_availability": [
        "nearest location", "closest office", "which office",
        "location not available", "no office near me", "too far",
        "branch not available", "no therapist at this location",
        "office hours", "location hours", "is there a clinic",
        "which clinic", "closest clinic",
    ],
    "condition_specific": [
        "torn meniscus", "lower back pain", "back pain",
        "myofascial pain", "knee pain", "shoulder pain",
        "hip pain", "neck pain", "spine", "herniated disc",
        "rotator cuff", "tendonitis", "sciatica", "post surgery",
        "post-surgery", "after surgery", "surgery recovery",
        "chronic pain", "sports injury", "accident injury",
        "physical injury", "muscle strain", "joint pain",
    ],
}

NEGATIVE_KEYWORDS = {**SHARED_NEGATIVE_KEYWORDS, **_PPT_EXTRA_KEYWORDS}
_neg_kw_pattern = build_neg_kw_pattern(NEGATIVE_KEYWORDS)

# ── Content/noise removal ─────────────────────────────────────────────────────
PT_CONTENT_SET = {
    "professional physical therapy", "professional pt", "physical therapy",
    "pt session", "therapy session", "treatment session",
}

_REMOVE_PATTERN = re.compile(
    r"chat with an agent|chat with agent|speak to agent|talk to agent|"
    r"connect me to agent|agent chat|chat to agent|agent help|help chat",
    re.IGNORECASE,
)

# ── Agent-name detection (improvement #13) ───────────────────────────────────
_AGENT_NAME_RE = re.compile(
    r'\b(agent|support|care|service|team|rep|representative|advisor|specialist|'
    r'associate|staff|bot|system|virtual|assistant|operator|helpdesk|help desk)\b',
    re.IGNORECASE,
)


def _is_agent_name(name):
    return bool(_AGENT_NAME_RE.search(name))


def _pick_customer(speaker_messages, ordered_speakers):
    """
    Return the customer speaker name from a dict of {speaker: [messages]}.
    Prefers speakers whose name does NOT look like an agent.
    Falls back to the speaker with fewest messages (tie-broken by first appearance).
    """
    non_agents = {s: msgs for s, msgs in speaker_messages.items() if not _is_agent_name(s)}
    pool = non_agents if non_agents else speaker_messages

    # Among candidates, prefer the first speaker by order of appearance
    ordered_candidates = [s for s in ordered_speakers if s in pool]
    if not ordered_candidates:
        ordered_candidates = list(pool.keys())

    # Pick the candidate with fewest messages; break ties by first appearance
    return min(ordered_candidates, key=lambda s: len(pool[s]))


def extract_customer_messages(text):
    """Extract customer-only messages from HTML or SMS transcript formats."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # ---- HTML format: <b>HH:MM:SS Speaker:</b> message <br/>
    html_pattern = r'<b>(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*</b>([^<]+?)(?:<br\s*/?>|$)'
    html_matches = re.findall(html_pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if html_matches:
        speaker_messages = {}
        ordered_speakers = []
        for _ts, speaker, message in html_matches:
            spk = speaker.strip().lower()
            if spk == "system":
                continue
            if spk not in speaker_messages:
                speaker_messages[spk] = []
                ordered_speakers.append(spk)
            speaker_messages[spk].append(message.strip())

        if not speaker_messages:
            return ""

        customer_name = _pick_customer(speaker_messages, ordered_speakers)
        msgs = speaker_messages[customer_name]
        return " ".join(re.sub(r"\s+", " ", m).strip() for m in msgs if m.strip())

    # ---- SMS format: HH:MM:SS Speaker: message
    sms_pattern = r'(\d{2}:\d{2}:\d{2})\s+([^:]+?)\s*:\s*(.+?)(?=\d{2}:\d{2}:\d{2}\s+|$)'
    sms_matches = re.findall(sms_pattern, text, flags=re.DOTALL)

    if sms_matches:
        speaker_messages = {}
        ordered_speakers = []
        for _ts, speaker, message in sms_matches:
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
                if spk not in speaker_messages:
                    speaker_messages[spk] = []
                    ordered_speakers.append(spk)
                speaker_messages[spk].append(msg)

        if not speaker_messages:
            return ""

        # Numeric names are almost always the customer (phone-number identifiers)
        phone_spks = [s for s in speaker_messages if re.match(r'^\d+$', s)]
        if phone_spks:
            customer_name = phone_spks[0]
        else:
            customer_name = _pick_customer(speaker_messages, ordered_speakers)

        msgs = speaker_messages[customer_name]
        return " ".join(re.sub(r"\s+", " ", m).strip() for m in msgs if m.strip())

    return ""


# ── Text cleaning ─────────────────────────────────────────────────────────────
def aggressive_clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _REMOVE_PATTERN.sub("", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 5 or len(text.split()) < 2:
        return ""
    return text


# Precompile per-content-item patterns once instead of inside the loop
_PT_CONTENT_PATTERNS = [
    re.compile(re.escape(c), re.IGNORECASE) for c in PT_CONTENT_SET
]
_PT_CONTENT_LOWER = [c.lower() for c in PT_CONTENT_SET]


def remove_pt_content_names(text):
    if not isinstance(text, str) or not text.strip():
        return text
    text_lower = text.lower()
    cleaned = text
    for content_lower, pattern in zip(_PT_CONTENT_LOWER, _PT_CONTENT_PATTERNS):
        if content_lower in text_lower:
            cleaned = pattern.sub("", cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ── Trigger patterns ──────────────────────────────────────────────────────────
_VERY_NEG_TRIGGERS = [
    "worst experience", "absolutely terrible", "completely unacceptable",
    "never coming back", "will never use again", "horrible experience",
    "disgusted", "furious", "outraged",
    "scam", "fraud", "stealing my money",
    "sue", "lawyer", "legal action",
    "file a complaint", "report to", "better business bureau",
]
_NEG_TRIGGERS = [
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
_NEUTRAL_TRIGGERS = [
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
_POS_TRIGGERS = [
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
_VERY_POS_TRIGGERS = [
    "outstanding", "exceeded expectations",
    "extremely helpful", "incredibly helpful",
    "absolutely perfect", "best service",
    "highly recommend", "couldn't be better",
    "amazing service", "excellent service",
    "love it", "love this", "absolutely love",
]
_STRONG_NEG_PHRASES = [
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

very_neg_pattern = re.compile("|".join(re.escape(p) for p in _VERY_NEG_TRIGGERS),  re.IGNORECASE)
neg_pattern      = re.compile("|".join(re.escape(p) for p in _NEG_TRIGGERS),        re.IGNORECASE)
neutral_pattern  = re.compile("|".join(re.escape(p) for p in _NEUTRAL_TRIGGERS),    re.IGNORECASE)
positive_pattern = re.compile("|".join(re.escape(p) for p in _POS_TRIGGERS),        re.IGNORECASE)
very_pos_pattern = re.compile("|".join(re.escape(p) for p in _VERY_POS_TRIGGERS),   re.IGNORECASE)


# ── Rule-based classifier — returns (score, confidence, rule_name) ─────────
def classify_by_rules(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, "blank"

    tl = text.lower()

    if very_neg_pattern.search(text):
        return -0.85, 0.95, "very_negative"

    if very_pos_pattern.search(text):
        return 0.85, 0.95, "very_positive"

    if positive_pattern.search(text):
        pos_count = len(re.findall(
            r'\b(thank|thanks|appreciate|perfect|helpful|great|excellent|awesome)\b', tl))
        if pos_count >= 2:
            return 0.75, 0.90, "positive_strong"
        return 0.35, 0.80, "positive"

    if neg_pattern.search(text):
        if has_resolution_or_thanks(text):
            neg_matches = len(neg_pattern.findall(text))
            if neg_matches == 1 and any(w in tl[-100:] for w in ["thank", "thanks", "appreciate"]):
                return 0.0, 0.75, "neutral_resolved"
        if any(phrase in tl for phrase in _STRONG_NEG_PHRASES):
            return -0.70, 0.95, "strong_negative"
        return -0.55, 0.80, "negative"

    if neutral_pattern.search(text):
        if has_simple_cancellation(text):
            return 0.0, 0.90, "neutral_cancel"
        if is_polite_request(text):
            return 0.0, 0.90, "neutral_polite"
        if "refund" in tl and any(p in tl for p in ["please", "want", "can you", "need", "request"]):
            return 0.0, 0.85, "neutral_refund"
        return 0.0, 0.80, "neutral"

    return None, 0.0, None


# ── Pre-built pattern strings for Polars str.contains (case-insensitive) ──────
_STRONG_NEG_RE = "(?i)" + "|".join(re.escape(p) for p in _STRONG_NEG_PHRASES)
_RESOLUTION_RE = (
    r"(?i)thanks|thank you|thank u|\bty\b|okay|\bok\b|perfect|"
    r"got it|understood|appreciate|sounds good|all set|all good|we're good"
)
_POS_WORD_RE   = r"(?i)\b(thank|thanks|appreciate|perfect|helpful|great|excellent|awesome)\b"


# ── Main analysis pipeline ────────────────────────────────────────────────────
def run_ppt_analysis(df, id_col, text_col, validation_dict,
                     progress_cb=None, rule_threshold=0.7):
    analyzer = load_vader()

    # ── Step 1: Polars preprocessing ─────────────────────────────────────────
    pldf = pl.from_pandas(df)

    pldf = pldf.with_columns(
        pl.col(id_col).cast(pl.String).str.strip_chars().str.replace_all(" ", "").alias("_id_norm"),
    ).with_columns(
        pl.col(text_col).map_elements(extract_customer_messages, return_dtype=pl.String).alias("CustomerOnly"),
    ).with_columns(
        pl.col("CustomerOnly").map_elements(
            lambda t: remove_pt_content_names(aggressive_clean_text(t)), return_dtype=pl.String
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
    # Prepend (?i) so Polars (Rust regex, case-sensitive by default) matches like Python re.IGNORECASE
    tc = pl.col("Text_For_Analysis")
    pldf = pldf.with_columns([
        (tc.str.len_bytes() == 0).alias("_blank"),
        tc.str.contains("(?i)" + very_neg_pattern.pattern).alias("_m_vneg"),
        tc.str.contains("(?i)" + very_pos_pattern.pattern).alias("_m_vpos"),
        tc.str.contains("(?i)" + positive_pattern.pattern).alias("_m_pos"),
        tc.str.contains("(?i)" + neg_pattern.pattern).alias("_m_neg"),
        tc.str.contains("(?i)" + neutral_pattern.pattern).alias("_m_neu"),
        tc.str.contains(_STRONG_NEG_RE).alias("_m_strong_neg"),
        tc.str.contains(_RESOLUTION_RE).alias("_m_resolution"),
        tc.str.count_matches(_POS_WORD_RE).alias("_pos_word_count"),
    ])

    # ── Step 4: Rule score/confidence/label (vectorized when/then) ────────────
    is_val = pl.col("_val_label").is_not_null()
    blank  = pl.col("_blank")

    pldf = pldf.with_columns([
        # rule score
        pl.when(is_val).then(
            pl.col("_val_label").map_elements(
                lambda x: SENTIMENT_TO_SCORE.get(x, 0.0), return_dtype=pl.Float64
            )
        )
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_vneg")).then(pl.lit(-0.85))
        .when(pl.col("_m_vpos")).then(pl.lit(0.85))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit(0.75))
        .when(pl.col("_m_pos")).then(pl.lit(0.35))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(-0.70))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.0))
        .when(pl.col("_m_neg")).then(pl.lit(-0.55))
        .when(pl.col("_m_neu")).then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .cast(pl.Float64).alias("_rule_score"),

        # rule confidence
        pl.when(is_val).then(pl.lit(1.0))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_m_vneg") | pl.col("_m_vpos")).then(pl.lit(0.95))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit(0.90))
        .when(pl.col("_m_pos")).then(pl.lit(0.80))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit(0.95))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit(0.75))
        .when(pl.col("_m_neg")).then(pl.lit(0.80))
        .when(pl.col("_m_neu")).then(pl.lit(0.80))
        .otherwise(pl.lit(0.5))
        .cast(pl.Float64).alias("_rule_conf"),

        # rule name
        pl.when(is_val).then(pl.lit("validation"))
        .when(blank).then(pl.lit("blank"))
        .when(pl.col("_m_vneg")).then(pl.lit("rule:very_negative"))
        .when(pl.col("_m_vpos")).then(pl.lit("rule:very_positive"))
        .when(pl.col("_m_pos") & (pl.col("_pos_word_count") >= 2)).then(pl.lit("rule:positive_strong"))
        .when(pl.col("_m_pos")).then(pl.lit("rule:positive"))
        .when(pl.col("_m_neg") & pl.col("_m_strong_neg")).then(pl.lit("rule:strong_negative"))
        .when(pl.col("_m_neg") & pl.col("_m_resolution")).then(pl.lit("rule:neutral_resolved"))
        .when(pl.col("_m_neg")).then(pl.lit("rule:negative"))
        .when(pl.col("_m_neu")).then(pl.lit("rule:neutral"))
        .otherwise(pl.lit("vader"))
        .alias("_rule_fired"),

        # source
        pl.when(is_val).then(pl.lit("Validation")).otherwise(pl.lit("Model")).alias("_source"),
    ])

    if progress_cb:
        progress_cb(len(pldf) // 2, len(pldf))

    # ── Step 5: Batch VADER only for rows that need it ────────────────────────
    texts_list = pldf["Text_For_Analysis"].fill_null("").to_list()
    needs_vader = (
        pldf["_rule_score"].is_null() |
        ((pldf["_rule_conf"] <= rule_threshold) & pldf["_val_label"].is_null() & ~pldf["_blank"])
    ).to_list()

    vader_raw = run_vader_parallel(texts_list, needs_vader, analyzer)

    pldf = pldf.with_columns(pl.Series("_vader_raw", vader_raw))

    # clamp tiny VADER scores to 0
    pldf = pldf.with_columns(
        pl.when(pl.col("_vader_raw").abs() < 0.05)
        .then(pl.lit(0.0))
        .otherwise(pl.col("_vader_raw"))
        .alias("_vader_clamped")
    )

    # ── Step 6: Final score/sentiment (vectorized) ────────────────────────────
    pldf = pldf.with_columns(
        pl.when(is_val).then(pl.col("_rule_score"))
        .when(blank).then(pl.lit(0.0))
        .when(pl.col("_rule_score").is_not_null() & (pl.col("_rule_conf") > rule_threshold))
            .then(pl.col("_rule_score"))
        .when(pl.col("_rule_score").is_not_null())
            # low-confidence blend
            .then(pl.col("_rule_score") * pl.col("_rule_conf") +
                  pl.col("_vader_clamped") * (1.0 - pl.col("_rule_conf")))
        .otherwise(pl.col("_vader_clamped"))
        .round(3)
        .alias("consumer_score")
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

    # ── Step 7: BERT correction (low-confidence, non-validated) ──────────────
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

    # ── Step 8: Keyword extraction (parallel map_elements) ────────────────────
    neg_kws = pl.Series(texts_list).map_elements(
        lambda t: extract_negative_keywords(t, _neg_kw_pattern), return_dtype=pl.String
    ).to_list()

    result_df["CustomerOnly"]      = pldf["CustomerOnly"].to_list()
    result_df["Text_For_Analysis"] = pldf["Text_For_Analysis"].to_list()
    result_df["_id_norm"]            = pldf["_id_norm"].to_list()
    result_df["validation_source"]   = pldf["_source"].to_list()
    result_df["_rule_fired"]         = result_df.get("_rule_fired", pldf["_rule_fired"].to_list())
    result_df["_raw_vader"]          = vader_raw
    result_df["Negative_Keywords"]   = neg_kws

    # drop internal Polars helper columns
    drop_cols = [c for c in result_df.columns if c.startswith("_m_") or c in
                 ("_blank", "_pos_word_count", "_rule_score", "_rule_conf",
                  "_val_label", "_vader_raw", "_vader_clamped", "_source")]
    result_df.drop(columns=drop_cols, errors="ignore", inplace=True)
    return result_df
