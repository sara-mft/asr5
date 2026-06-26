"""
models.py — Result & Metrics Data Models  [v2]
===============================================
Immutable dataclasses that represent a single transcription attempt and its
evaluated metrics.


"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Diarization
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DiarizationSegment:
    """One time-aligned, speaker-attributed chunk of speech."""

    speaker: str
    start_sec: float
    end_sec: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def format_line(self) -> str:
        """[00:01.4 - 00:03.7]  SPEAKER_00: Hello there."""
        def _fmt(t: float) -> str:
            m, s = divmod(t, 60)
            return f"{int(m):02d}:{s:04.1f}"
        return f"[{_fmt(self.start_sec)} - {_fmt(self.end_sec)}]  {self.speaker}: {self.text}"


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TranscriptionMetrics:
    """
    Per-file metrics computed after a transcription attempt.

    All accuracy fields are None when no reference transcript is available.
    A value of None is semantically distinct from 0.0 ("not computed" vs
    "computed and equal to zero").

    Standard metrics (always computed when reference is present)
    -----------------------------------------------------------
    latency_sec             Wall-clock seconds from request start to result.
    audio_duration_sec      Duration of the source audio file in seconds.
    real_time_factor        latency_sec / audio_duration_sec (< 1 = faster than real-time).
    word_error_rate         WER ∈ [0, ∞).
    word_accuracy           max(0, 1 − WER).
    character_error_rate    CER ∈ [0, ∞).
    insertions / deletions / substitutions  Edit-distance breakdown.
    match_error_rate        MER — symmetric WER normalised by max(|ref|, |hyp|).
    word_info_preserved     WIP ∈ [0, 1].
    word_info_lost          WIL = 1 − WIP.
    estimated_cost_usd      Cost estimate based on config $/min and audio duration.
    confidence              Average word confidence [0, 1] if provided by the engine API.

    Extended metrics (controlled by MetricsConfig feature flags)
    -----------------------------------------------------------
    norm_word_error_rate    WER computed after number/amount normalisation.
    term_error_rate         TNER — domain lexicon term miss rate (requires lexicon).
    hallucination_rate      HAR — fraction of hypothesis words absent from reference.
    punctuation_f1_macro    Macro F1 over {COMMA, PERIOD, QUESTION, EXCLAM}.
    punctuation_error_rate  PPER — weighted punctuation placement error rate.


    Diarization-specific
    --------------------
    num_speakers            Distinct speaker labels found (diarization engines only).
    diarization_error_rate
    jaccard_error_rate
    """

    # ── Infrastructure ────────────────────────────────────────────────────────
    latency_sec: float = 0.0
    audio_duration_sec: float = 0.0
    ## start modif ##
    time_to_first_token_sec: float | None = None   # NEW: TTFW for streaming engines
    ## end modif ##
    real_time_factor: float = 0.0
    estimated_cost_usd: float = 0.0
    confidence: float | None = None

    # ── Standard accuracy ─────────────────────────────────────────────────────
    word_error_rate: float | None = None
    word_accuracy: float | None = None
    character_error_rate: float | None = None
    insertions: int | None = None
    deletions: int | None = None
    substitutions: int | None = None
    match_error_rate: float | None = None
    word_info_preserved: float | None = None
    word_info_lost: float | None = None

    # ── Extended accuracy ─────────────────────────────────────────────────────
    norm_word_error_rate: float | None = None  
    ## start modif ##
    onset_wer: float | None = None                 # NEW: First-N WER   
    ## end modif ##

    # ── Domain / semantic ─────────────────────────────────────────────────────
    term_error_rate: float | None = None
    hallucination_rate: float | None = None

    # ── Punctuation ───────────────────────────────────────────────────────────
    punctuation_f1_macro: float | None = None
    punctuation_error_rate: float | None = None

    # ── Diarization ───────────────────────────────────────────────────────────
    num_speakers: int | None = None
    ## diar modif ##
    diarization_error_rate: float | None = None
    jaccard_error_rate: float | None = None
    ## diar modif ##

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TranscriptionResult:
    """
    Complete record for one (engine × audio-file) benchmarking cell.

    Attributes
    ----------
    engine_id            Unique identifier matching EngineConfig.engine_id.
    engine_type          Canonical engine type string.
    audio_file           Stem of the input .wav file.
    hypothesis           Flat transcription text (used for WER/CER/all metrics).
    reference            Gold-standard transcript if provided, else None.
    metrics              Computed TranscriptionMetrics for this result.
    diarization_segments Speaker-segmented output (diarization engines only).
    error                Exception message if the attempt failed.
    raw_response         Optional raw API response payload for debugging.
    timestamp_utc        ISO-8601 timestamp of when the attempt started.
    """

    engine_id: str
    engine_type: str
    audio_file: str
    hypothesis: str
    reference: str | None
    metrics: TranscriptionMetrics
    diarization_segments: list[DiarizationSegment] = field(default_factory=list)
    error: str | None = None
    raw_response: dict | None = field(default=None, repr=False)
    timestamp_utc: str = ""

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.hypothesis)

    @property
    def has_diarization(self) -> bool:
        return len(self.diarization_segments) > 0

    @property
    def diarization_transcript(self) -> str:
        return "\n".join(seg.format_line() for seg in self.diarization_segments)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a single dictionary suitable for CSV / JSON export."""
        d: dict[str, Any] = {
            "engine_id": self.engine_id,
            "engine_type": self.engine_type,
            "audio_file": self.audio_file,
            "timestamp_utc": self.timestamp_utc,
            "success": self.success,
            "hypothesis": self.hypothesis,
            "reference": self.reference,
            "error": self.error,
            "diarization_transcript": self.diarization_transcript if self.has_diarization else "",
        }
        d.update(self.metrics.to_dict())
        return d
