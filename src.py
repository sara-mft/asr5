"""
metrics_meta.py
================
Single source of truth for everything the dashboard needs to know about
each canonical metric:

  * human-readable label and unit
  * "lower is better" or "higher is better" direction
  * thematic group (for grouping selectors and radar plots)
  * a one-sentence tooltip/description
  * display format (% vs raw vs seconds vs $)

Import::

    from metrics_meta import METRICS, metric_label, metric_direction
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricMeta:
    key: str            # canonical short key matching data_loader.FIELD_MAP
    label: str          # short human label (used in charts / table headers)
    unit: str           # "", "%", "s", "x", "$", "speakers"
    direction: str      # "lower" | "higher"
    group: str          # thematic group name for organised selectors
    description: str    # one-sentence tooltip
    format: str         # Python format spec, e.g. ".2%" or ".3f"
    tags: list[str] = field(default_factory=list)   # optional feature tags


# ── Accuracy ─────────────────────────────────────────────────────────────────
_ACC = "Accuracy"
# ── Punctuation ──────────────────────────────────────────────────────────────
_PUN = "Punctuation"
# ── Hallucination / Named Entities ───────────────────────────────────────────
_HAL = "Hallucination & Terms"
# ── Latency / Speed ──────────────────────────────────────────────────────────
_SPD = "Speed & Latency"
# ── Diarization ──────────────────────────────────────────────────────────────
_DIA = "Diarization"
# ── Cost ─────────────────────────────────────────────────────────────────────
_CST = "Cost"
# ── Misc ─────────────────────────────────────────────────────────────────────
_MSC = "Misc"


METRICS: list[MetricMeta] = [
    # ── Accuracy ─────────────────────────────────────────────────────────────
    MetricMeta(
        key="wer", label="WER", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "Word Error Rate — percentage of words in the reference that the model "
            "inserted, deleted or substituted. The main accuracy metric for STT. "
            "Lower is better."
        ),
    ),
    MetricMeta(
        key="norm_wer", label="Norm WER", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "WER computed after text normalisation (lower-case, number expansion, "
            "punctuation removal). More robust to capitalisation and formatting "
            "differences. Lower is better."
        ),
    ),
    MetricMeta(
        key="cer", label="CER", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "Character Error Rate — edit distance at the character level divided by "
            "the reference character count. Useful for morphologically rich languages "
            "or noisy audio. Lower is better."
        ),
    ),
    MetricMeta(
        key="mer", label="MER", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "Match Error Rate — edit distance divided by the maximum of reference "
            "and hypothesis lengths. Penalises both insertions and deletions "
            "proportionally. Lower is better."
        ),
    ),
    MetricMeta(
        key="wip", label="WIP", unit="%", direction="higher", group=_ACC,
        format=".2%",
        description=(
            "Word Information Preserved — proportion of word-level information from "
            "the reference retained in the hypothesis. Complement of WIL. "
            "Higher is better."
        ),
    ),
    MetricMeta(
        key="wil", label="WIL", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "Word Information Lost — proportion of reference word information lost "
            "in the hypothesis. Complement of WIP. Lower is better."
        ),
    ),
    MetricMeta(
        key="onset_wer", label="Onset WER", unit="%", direction="lower", group=_ACC,
        format=".2%",
        description=(
            "WER restricted to the first word of each utterance / segment. "
            "Measures how well the model captures sentence beginnings, "
            "which are critical for downstream parsing. Lower is better."
        ),
    ),
    # ── Hallucination / Terms ────────────────────────────────────────────────
    MetricMeta(
        key="har", label="Hallucination Rate", unit="%", direction="lower", group=_HAL,
        format=".2%",
        description=(
            "Proportion of hypothesis words that have no reference counterpart "
            "(pure insertions) — proxy for hallucinated content. Lower is better."
        ),
    ),
    MetricMeta(
        key="tner", label="Term Error Rate", unit="%", direction="lower", group=_HAL,
        format=".2%",
        description=(
            "Named-Entity / Term Error Rate — WER restricted to a domain vocabulary "
            "(product names, proper nouns, etc.). Directly measures business-critical "
            "recognition quality. Lower is better."
        ),
    ),
    # ── Punctuation ──────────────────────────────────────────────────────────
    MetricMeta(
        key="punc_f1", label="Punctuation F1", unit="", direction="higher", group=_PUN,
        format=".3f",
        description=(
            "Macro-averaged F1 over punctuation classes (period, comma, question "
            "mark, …). Measures how accurately the model restores sentence boundaries. "
            "Higher is better."
        ),
    ),
    MetricMeta(
        key="pper", label="Punctuation Error Rate", unit="%", direction="lower", group=_PUN,
        format=".2%",
        description=(
            "Rate of punctuation errors relative to the reference. "
            "Complements Punctuation F1 with an error-count perspective. "
            "Lower is better."
        ),
    ),
    # ── Diarization ──────────────────────────────────────────────────────────
    MetricMeta(
        key="der", label="DER", unit="%", direction="lower", group=_DIA,
        format=".2%",
        description=(
            "Diarization Error Rate — fraction of audio time incorrectly attributed "
            "to a speaker (missed, false alarm, or confusion). "
            "Only populated for diarization-capable engines. Lower is better."
        ),
        tags=["Diarization"],
    ),
    MetricMeta(
        key="jer", label="JER", unit="%", direction="lower", group=_DIA,
        format=".2%",
        description=(
            "Jaccard Error Rate — segment-level speaker assignment error based on "
            "Jaccard distance between reference and hypothesis speaker sets. "
            "Lower is better."
        ),
        tags=["Diarization"],
    ),
    # ── Speed & Latency ──────────────────────────────────────────────────────
    MetricMeta(
        key="latency_sec", label="Latency", unit="s", direction="lower", group=_SPD,
        format=".2f",
        description=(
            "End-to-end wall-clock time from audio submission to full transcript "
            "delivery (seconds). Includes network round-trip. Lower is better."
        ),
    ),
    MetricMeta(
        key="rtf", label="RTF", unit="x", direction="lower", group=_SPD,
        format=".3f",
        description=(
            "Real-Time Factor — processing time divided by audio duration. "
            "RTF < 1 means faster than real-time; RTF > 1 means slower. "
            "Lower is better."
        ),
    ),
    MetricMeta(
        key="ttfw_sec", label="TTFW", unit="s", direction="lower", group=_SPD,
        format=".2f",
        description=(
            "Time To First Word — latency from audio start to first recognised word "
            "token (streaming engines only). Drives perceived responsiveness. "
            "Lower is better."
        ),
        tags=["Streaming"],
    ),
    # ── Cost ─────────────────────────────────────────────────────────────────
    MetricMeta(
        key="cost_usd", label="Cost (USD)", unit="$", direction="lower", group=_CST,
        format=".4f",
        description=(
            "Estimated total inference cost in US dollars for the evaluated audio "
            "corpus, based on provider pricing at time of benchmark. Lower is better."
        ),
    ),
    # ── Misc ─────────────────────────────────────────────────────────────────
    MetricMeta(
        key="confidence", label="Confidence", unit="", direction="higher", group=_MSC,
        format=".3f",
        description=(
            "Average per-utterance confidence score returned by the engine "
            "(0–1). Only populated for engines that expose confidence. "
            "Higher is better."
        ),
    ),
]

# ─── Fast lookup helpers ──────────────────────────────────────────────────────

_BY_KEY: dict[str, MetricMeta] = {m.key: m for m in METRICS}

_ALL_KEYS: list[str] = [m.key for m in METRICS]

# Keys considered "primary" — always shown by default in overview views
DEFAULT_PRIMARY_METRICS: list[str] = ["wer", "cer", "latency_sec", "rtf", "cost_usd", "punc_f1"]

# Keys useful for a radar/spider chart (pure quality scores, all on similar scales)
DEFAULT_RADAR_METRICS: list[str] = ["wer", "cer", "punc_f1", "har", "tner", "onset_wer"]

# Keys available on most models (exclude optional diarization/ttfw)
CORE_ACCURACY_METRICS: list[str] = ["wer", "norm_wer", "cer", "mer", "wip", "wil", "onset_wer"]

# Grouped index: group_name -> list[MetricMeta]
METRICS_BY_GROUP: dict[str, list[MetricMeta]] = {}
for _m in METRICS:
    METRICS_BY_GROUP.setdefault(_m.group, []).append(_m)

# Group display order (controls the order sections appear in selectors)
GROUP_ORDER: list[str] = [_ACC, _PUN, _HAL, _SPD, _DIA, _CST, _MSC]

ALL_GROUPS: list[str] = [g for g in GROUP_ORDER if g in METRICS_BY_GROUP]


def get(key: str) -> MetricMeta | None:
    return _BY_KEY.get(key)


def metric_label(key: str) -> str:
    m = _BY_KEY.get(key)
    return m.label if m else key


def metric_direction(key: str) -> str:
    m = _BY_KEY.get(key)
    return m.direction if m else "lower"


def metric_format(key: str) -> str:
    m = _BY_KEY.get(key)
    return m.format if m else ".4f"


def metric_group(key: str) -> str:
    m = _BY_KEY.get(key)
    return m.group if m else "Misc"


def format_value(key: str, value: float | None, na: str = "—") -> str:
    """Format a scalar metric value according to its display spec."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return na
    m = _BY_KEY.get(key)
    if m is None:
        return f"{value:.4f}"
    try:
        return format(value, m.format)
    except (ValueError, TypeError):
        return str(value)


def normalize_series(key: str, series):
    """
    Return a 0-1 normalised pandas Series where 1.0 = best performance.
    Handles direction automatically.
    """
    import pandas as pd
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if mn == mx or pd.isna(mn) or pd.isna(mx):
        return s.fillna(0) * 0  # all-zero; avoids division by zero
    if metric_direction(key) == "lower":
        return 1 - (s - mn) / (mx - mn)
    else:
        return (s - mn) / (mx - mn)
