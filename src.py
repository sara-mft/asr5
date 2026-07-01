"""
data_loader.py
================
Pure data-loading and parsing logic for the STT Benchmark Dashboard.

Deliberately free of any Streamlit import so it can be unit-tested in
isolation and reused outside the app (e.g. in a notebook or CI check).

Expected on-disk layout::

    results/
        <dataset_1>/
            <source>_<dataset_1>.json
            ...
        <dataset_2>/
            <source>_<dataset_2>.json
            ...

Each JSON file has the shape::

    {
      "summary": {
        "<engine_id>": {"num_files": int, "avg_wer": float, ...},
        ...
      },
      "results": [
        {"engine_id": ..., "audio_file": ..., "word_error_rate": ..., ...},
        ...
      ]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Naming-convention helpers
# ---------------------------------------------------------------------------

# Trailing tokens recognized as "feature flags" when auto-deriving tags from
# a model identifier such as ``azure_speech_fr_diarize``. Purely a fallback
# used when no explicit YAML category mapping is supplied for a model -
# extend freely, order does not matter.
KNOWN_FEATURE_TOKENS: dict[str, str] = {
    "diarize": "Diarization",
    "diarization": "Diarization",
    "fr": "French",
    "en": "English",
    "multilingual": "Multilingual",
    "streaming": "Streaming",
    "stream": "Streaming",
    "realtime": "Real-time",
    "fast": "Fast",
    "large": "Large",
    "small": "Small",
    "medium": "Medium",
    "mini": "Mini",
    "turbo": "Turbo",
    "v1": "v1",
    "v2": "v2",
    "v3": "v3",
    "beta": "Beta",
}


def parse_source_from_filename(file_path: Path, dataset_name: str) -> str:
    """
    Extract the ``source`` token from a results filename following the
    ``<source>_<dataset>.json`` convention.

    Falls back gracefully when the filename does not exactly match the
    dataset name suffix (e.g. extra qualifiers), by trying the first
    underscore-separated token instead.
    """
    stem = file_path.stem  # filename without .json
    suffix = f"_{dataset_name}"
    if stem.endswith(suffix) and len(stem) > len(suffix):
        return stem[: -len(suffix)]
    if "_" in stem:
        return stem.split("_")[0]
    return stem or "unknown"


def make_model_key(source: str, engine_id: str) -> str:
    """
    Build the canonical ``<source>_<model_name>_<features>`` identifier used
    throughout the dashboard and as the YAML category-mapping key.

    If ``engine_id`` already starts with the source token (some pipelines
    embed it, some don't - both appear in real benchmark exports) it is not
    duplicated.
    """
    src = source.strip().lower()
    eid = engine_id.strip()
    if eid.lower() == src or eid.lower().startswith(src + "_") or eid.lower().startswith(src + "-"):
        return eid
    return f"{source}_{eid}"


def split_model_key(model_key: str, source: str) -> tuple[str, list[str]]:
    """
    Best-effort split of a model key into a base ``model_name`` and a list
    of recognized trailing ``features`` (see ``KNOWN_FEATURE_TOKENS``).
    Used only for display/auto-tagging - never for identity/grouping.
    """
    tokens = model_key.split("_")
    if tokens and tokens[0].lower() == source.strip().lower():
        tokens = tokens[1:]
    if not tokens:
        return model_key, []

    features: list[str] = []
    while len(tokens) > 1:
        candidate = tokens[-1].lower()
        if candidate in KNOWN_FEATURE_TOKENS:
            features.insert(0, KNOWN_FEATURE_TOKENS[candidate])
            tokens.pop()
        else:
            break
    model_name = "_".join(tokens) if tokens else model_key
    return model_name, features


def auto_tags_from_model_key(model_key: str, source: str) -> list[str]:
    """Convenience wrapper returning only the auto-derived feature tags."""
    _, features = split_model_key(model_key, source)
    return features


# ---------------------------------------------------------------------------
# Canonical metric field mapping
# ---------------------------------------------------------------------------
# Canonical short key -> (summary JSON field, per-file result JSON field)
# A field of None means it has no counterpart in that record type.
FIELD_MAP: dict[str, tuple[str | None, str | None]] = {
    "latency_sec":   ("avg_latency_sec", "latency_sec"),
    "ttfw_sec":      ("avg_ttfw_sec", "time_to_first_token_sec"),
    "rtf":           ("avg_rtf", "real_time_factor"),
    "cost_usd":      ("total_cost_usd", "estimated_cost_usd"),
    "confidence":    ("avg_confidence", "confidence"),
    "num_speakers":  ("avg_speakers", "num_speakers"),
    "wer":           ("avg_wer", "word_error_rate"),
    "onset_wer":     ("avg_onset_wer", "onset_wer"),
    "cer":           ("avg_cer", "character_error_rate"),
    "mer":           ("avg_mer", "match_error_rate"),
    "wip":           ("avg_wip", "word_info_preserved"),
    "wil":           ("avg_wil", "word_info_lost"),
    "norm_wer":      ("avg_norm_wer", "norm_word_error_rate"),
    "tner":          ("avg_tner", "term_error_rate"),
    "har":           ("avg_har", "hallucination_rate"),
    "punc_f1":       ("avg_punc_f1", "punctuation_f1_macro"),
    "pper":          ("avg_pper", "punctuation_error_rate"),
    "der":           ("avg_der", "diarization_error_rate"),
    "jer":           ("avg_jer", "jaccard_error_rate"),
}

SUMMARY_EXTRA_FIELDS = ["num_files", "num_successful", "num_errors"]
DETAIL_EXTRA_FIELDS = [
    "audio_file", "timestamp_utc", "success", "error",
    "hypothesis", "reference", "diarization_transcript",
    "audio_duration_sec", "word_accuracy",
    "insertions", "deletions", "substitutions",
]


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

@dataclass
class LoadIssue:
    file_path: str
    message: str


@dataclass
class LoadResult:
    summary_df: pd.DataFrame
    detail_df: pd.DataFrame
    issues: list[LoadIssue] = field(default_factory=list)


def discover_result_files(results_root: Path) -> list[tuple[str, Path]]:
    """Return a list of (dataset_name, file_path) for every *.json under results_root/<dataset>/."""
    found: list[tuple[str, Path]] = []
    if not results_root.exists() or not results_root.is_dir():
        return found
    for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for file_path in sorted(dataset_dir.glob("*.json")):
            found.append((dataset_dir.name, file_path))
    return found


def _safe_get(d: dict, key: str | None):
    if key is None:
        return None
    return d.get(key)


def load_results(results_root: str | Path) -> LoadResult:
    """
    Scan ``results_root`` and parse every JSON file into two tidy
    long-format DataFrames: one row per (dataset, model) in ``summary_df``,
    one row per (dataset, model, audio_file) in ``detail_df``.
    """
    results_root = Path(results_root)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    issues: list[LoadIssue] = []

    for dataset_name, file_path in discover_result_files(results_root):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the UI
            issues.append(LoadIssue(str(file_path), f"Could not parse JSON: {exc}"))
            continue

        source = parse_source_from_filename(file_path, dataset_name)
        summary = payload.get("summary", {})
        results = payload.get("results", [])

        if not isinstance(summary, dict):
            issues.append(LoadIssue(str(file_path), "'summary' key is not an object - skipped."))
            summary = {}
        if not isinstance(results, list):
            issues.append(LoadIssue(str(file_path), "'results' key is not a list - skipped."))
            results = []

        for engine_id, entry in summary.items():
            if not isinstance(entry, dict):
                continue
            model_key = make_model_key(source, engine_id)
            model_name, auto_features = split_model_key(model_key, source)
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "source": source,
                "engine_id": engine_id,
                "model_key": model_key,
                "model_name": model_name,
                "auto_tags": auto_features,
                "file_path": str(file_path),
            }
            for extra in SUMMARY_EXTRA_FIELDS:
                row[extra] = entry.get(extra)
            for canonical, (summary_field, _detail_field) in FIELD_MAP.items():
                row[canonical] = _safe_get(entry, summary_field)
            summary_rows.append(row)

        for record in results:
            if not isinstance(record, dict):
                continue
            engine_id = record.get("engine_id", "unknown")
            model_key = make_model_key(source, engine_id)
            model_name, auto_features = split_model_key(model_key, source)
            row = {
                "dataset": dataset_name,
                "source": source,
                "engine_id": engine_id,
                "engine_type": record.get("engine_type"),
                "model_key": model_key,
                "model_name": model_name,
                "auto_tags": auto_features,
                "file_path": str(file_path),
            }
            for extra in DETAIL_EXTRA_FIELDS:
                row[extra] = record.get(extra)
            for canonical, (_summary_field, detail_field) in FIELD_MAP.items():
                row[canonical] = _safe_get(record, detail_field)
            # Keep the raw diarization segment list around for the transcript
            # viewer, but as an opaque object - never flattened into metrics.
            row["diarization_segments"] = record.get("diarization_segments")
            detail_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    return LoadResult(summary_df=summary_df, detail_df=detail_df, issues=issues)


# ---------------------------------------------------------------------------
# Model metadata (YAML) loading
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    models: dict[str, dict[str, Any]]
    providers: dict[str, dict[str, Any]]
    raw: dict[str, Any]


def load_model_metadata(yaml_path: str | Path | None) -> ModelMetadata:
    """Load the optional model-categorization YAML. Returns an empty, valid
    ModelMetadata object (never raises) when the file is missing or invalid,
    so the rest of the dashboard can run uncategorized."""
    if not yaml_path:
        return ModelMetadata(models={}, providers={}, raw={})
    path = Path(yaml_path)
    if not path.exists():
        return ModelMetadata(models={}, providers={}, raw={})
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return ModelMetadata(models={}, providers={}, raw={})
    models = raw.get("models", {}) if isinstance(raw, dict) else {}
    providers = raw.get("providers", {}) if isinstance(raw, dict) else {}
    if not isinstance(models, dict):
        models = {}
    if not isinstance(providers, dict):
        providers = {}
    return ModelMetadata(models=models, providers=providers, raw=raw)


def parse_yaml_text(text: str) -> ModelMetadata:
    """Same as load_model_metadata but from an in-memory string (used for
    the sidebar file-uploader path). Never raises."""
    try:
        raw = yaml.safe_load(text) or {}
    except Exception:
        return ModelMetadata(models={}, providers={}, raw={})
    models = raw.get("models", {}) if isinstance(raw, dict) else {}
    providers = raw.get("providers", {}) if isinstance(raw, dict) else {}
    if not isinstance(models, dict):
        models = {}
    if not isinstance(providers, dict):
        providers = {}
    return ModelMetadata(models=models, providers=providers, raw=raw)


def lookup_model_metadata(meta: ModelMetadata, model_key: str, engine_id: str) -> dict[str, Any]:
    """
    Resolve metadata for a model with graceful fallbacks:
    1. exact ``model_key`` match
    2. case-insensitive ``model_key`` match
    3. exact ``engine_id`` match (in case the YAML wasn't source-prefixed)
    4. case-insensitive ``engine_id`` match
    5. {} (uncategorized)
    """
    if model_key in meta.models:
        return meta.models[model_key] or {}
    lower_map = {k.lower(): v for k, v in meta.models.items()}
    if model_key.lower() in lower_map:
        return lower_map[model_key.lower()] or {}
    if engine_id in meta.models:
        return meta.models[engine_id] or {}
    if engine_id.lower() in lower_map:
        return lower_map[engine_id.lower()] or {}
    return {}


def get_model_categories(meta: ModelMetadata, model_key: str, engine_id: str, auto_tags: list[str]) -> list[str]:
    """Categories from YAML if present, otherwise fall back to auto-derived tags."""
    entry = lookup_model_metadata(meta, model_key, engine_id)
    cats = entry.get("categories")
    if isinstance(cats, list) and cats:
        return [str(c) for c in cats]
    return list(auto_tags)


def get_model_display_name(meta: ModelMetadata, model_key: str, engine_id: str, fallback: str) -> str:
    entry = lookup_model_metadata(meta, model_key, engine_id)
    name = entry.get("display_name")
    return str(name) if name else fallback


# ---------------------------------------------------------------------------
# Enrichment helpers (attach YAML-derived columns to a loaded DataFrame)
# ---------------------------------------------------------------------------

def enrich_with_metadata(df: pd.DataFrame, meta: ModelMetadata) -> pd.DataFrame:
    """Add category/display_name/provider columns derived from the YAML
    metadata (with auto-tag fallback) to a summary or detail DataFrame."""
    if df.empty:
        df = df.copy()
        df["categories"] = pd.Series(dtype=object)
        df["display_name"] = pd.Series(dtype=object)
        df["provider"] = pd.Series(dtype=object)
        df["license"] = pd.Series(dtype=object)
        return df

    df = df.copy()
    categories_col = []
    display_col = []
    provider_col = []
    license_col = []
    for _, row in df.iterrows():
        entry = lookup_model_metadata(meta, row["model_key"], row["engine_id"])
        cats = entry.get("categories")
        if not (isinstance(cats, list) and cats):
            cats = row.get("auto_tags") or []
        categories_col.append([str(c) for c in cats])
        display_col.append(entry.get("display_name") or row["model_key"])
        provider_col.append(entry.get("provider") or row["source"])
        license_col.append(entry.get("license"))
    df["categories"] = categories_col
    df["display_name"] = display_col
    df["provider"] = provider_col
    df["license"] = license_col
    return df


def all_categories(df: pd.DataFrame) -> list[str]:
    """Flatten and de-duplicate the 'categories' list-column across a DataFrame."""
    if df.empty or "categories" not in df.columns:
        return []
    seen: set[str] = set()
    for cats in df["categories"]:
        for c in cats or []:
            seen.add(c)
    return sorted(seen)
