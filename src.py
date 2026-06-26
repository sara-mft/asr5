"""
benchmark_runner.py — Orchestration Layer  [v2]
================================================
BenchmarkRunner drives the full benchmarking loop:

    for each enabled engine:
        for each audio file:
            attempt transcription (with retry)
            compute metrics (controlled by MetricsConfig feature flags)
            collect TranscriptionResult

"""

from __future__ import annotations

import concurrent.futures
import datetime
import logging
import time
from pathlib import Path
from typing import Sequence

from config import BenchmarkConfig, MetricsConfig
from engines import BaseSTTEngine, EngineFactory, extract_confidence
from metrics import (
    compute_wer,
    compute_cer,
    compute_rtf,
    estimate_cost,
    compute_mer,
    compute_wip_wil,
    compute_tner,
    compute_har,
    compute_punctuation_metrics,
    compute_norm_wer,
    compute_onset_wer,
)
from models import TranscriptionMetrics, TranscriptionResult, DiarizationSegment

log = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates the benchmarking of multiple STT engines against a set of
    audio files.

    Parameters
    ----------
    config:           Global BenchmarkConfig object.
    engine_whitelist: Optional list of engine_id strings to run.
                      When None, all enabled engines in config are used.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        engine_whitelist: list[str] | None = None,
    ) -> None:
        self.config = config
        self._engines: list[BaseSTTEngine] = self._build_engines(engine_whitelist)


    # ── Public API ────────────────────────────────────────────────────────────
    
    def run(
        self,
        audio_files: Sequence[Path],
        ref_transcripts: dict[str, str] | None = None,
        diar_ref_transcripts: dict[str, str] = None ## diar modif ##
    ) -> list[TranscriptionResult]:
        """
        Run the full benchmark matrix (engines × files).

        Uses sequential execution by default. Set config.max_workers > 1 to
        enable parallel execution via ThreadPoolExecutor.

        Parameters
        ----------
        audio_files:     Ordered list of .wav file paths to evaluate.
        ref_transcripts: Optional {stem → reference_text} mapping for accuracy metrics.

        Returns
        -------
        Flat list of TranscriptionResult objects (one per engine × file cell),
        ordered by engine then file.
        """
        ref_transcripts = ref_transcripts or {}
        total_cells = len(self._engines) * len(audio_files)
        log.info(
            "Benchmark matrix: %d engine(s) × %d file(s) = %d cell(s)  "
            "[workers=%d]",
            len(self._engines),
            len(audio_files),
            total_cells,
            self.config.max_workers,
        )

        if self.config.max_workers > 1:
            return self._run_parallel(audio_files, ref_transcripts, diar_ref_transcripts) ## diar modif ##
        return self._run_sequential(audio_files, ref_transcripts, diar_ref_transcripts) ## diar modif ##

    # ── Execution strategies ──────────────────────────────────────────────────

    def _run_sequential(
        self,
        audio_files: Sequence[Path],
        ref_transcripts: dict[str, str],
        diar_ref_transcripts: dict[str, str] = None ## diar modif ##
    ) -> list[TranscriptionResult]:
        all_results: list[TranscriptionResult] = []
        for engine_idx, engine in enumerate(self._engines, start=1):
            log.info(
                "[Engine %d/%d] %s (%s)",
                engine_idx, len(self._engines),
                engine.engine_id, engine.config.engine_type,
            )
            for file_idx, audio_path in enumerate(audio_files, start=1):
                log.info("  [File %d/%d] %s", file_idx, len(audio_files), audio_path.name)
                result = self._run_single(engine, audio_path, ref_transcripts, diar_ref_transcripts) ## diar modif ##
                all_results.append(result)
                self._log_result_summary(result)
        return all_results


            ## start modif ##

    # def _run_parallel(
    #     self,
    #     audio_files: Sequence[Path],
    #     ref_transcripts: dict[str, str],
    # ) -> list[TranscriptionResult]:
    #     """
    #     Parallel execution: each (engine, file) cell runs in a thread.
    #     Results are sorted to preserve the same order as sequential mode.
    #     """
    #     cells = [
    #         (ei, fi, engine, audio_path)
    #         for ei, engine in enumerate(self._engines)
    #         for fi, audio_path in enumerate(audio_files)
    #     ]

    #     results_map: dict[tuple[int, int], TranscriptionResult] = {}

    #     def _task(cell):
    #         ei, fi, engine, audio_path = cell
    #         log.info("  [parallel] %s × %s", engine.engine_id, audio_path.name)
    #         result = self._run_single(engine, audio_path, ref_transcripts)
    #         self._log_result_summary(result)
    #         return (ei, fi), result

    #     with concurrent.futures.ThreadPoolExecutor(
    #         max_workers=self.config.max_workers
    #     ) as executor:
    #         futures = {executor.submit(_task, cell): cell for cell in cells}
    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 key, result = future.result()
    #                 results_map[key] = result
    #             except Exception as exc:
    #                 cell = futures[future]
    #                 log.error("Cell (%s, %s) raised: %s", cell[2].engine_id, cell[3].name, exc)
    #                 results_map[(cell[0], cell[1])] = TranscriptionResult(
    #                     engine_id=cell[2].engine_id,
    #                     engine_type=cell[2].config.engine_type,
    #                     audio_file=cell[3].stem,
    #                     hypothesis="",
    #                     reference=None,
    #                     metrics=TranscriptionMetrics(),
    #                     error=f"Unhandled executor exception: {exc}",
    #                     timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    #                 )

    #     # Restore deterministic order: engine-major, file-minor
    #     return [
    #         results_map[(ei, fi)]
    #         for ei in range(len(self._engines))
    #         for fi in range(len(audio_files))
    #         if (ei, fi) in results_map
    #     ]


    def _run_parallel(
        self,
        audio_files: Sequence[Path],
        ref_transcripts: dict[str, str],
        diar_ref_transcripts: dict[str, str] = None ## diar modif ##
    ) -> list[TranscriptionResult]:
        """
        Hybrid execution: REST API engines run in parallel via ThreadPoolExecutor.
        Native Azure Speech SDK engines run sequentially to prevent C++ GIL crashes.
        """
        results_map: dict[tuple[int, int], TranscriptionResult] = {}
        parallel_cells = []

        # Define which engine types MUST run sequentially
        sequential_types = {"azure_speech", "azure_custom_speech"}

        # 1. Distribute the workload
        for ei, engine in enumerate(self._engines):
            if engine.config.engine_type in sequential_types:
                log.info(
                    "  [Hybrid] Running '%s' sequentially (Native SDK isolated)", 
                    engine.engine_id
                )
                for fi, audio_path in enumerate(audio_files):
                    result = self._run_single(engine, audio_path, ref_transcripts, diar_ref_transcripts) ## diar modif ##
                    self._log_result_summary(result)
                    results_map[(ei, fi)] = result
            else:
                for fi, audio_path in enumerate(audio_files):
                    parallel_cells.append((ei, fi, engine, audio_path))

        # 2. Process REST API engines in parallel
        def _task(cell):
            ei, fi, engine, audio_path = cell
            log.info("  [parallel] %s × %s", engine.engine_id, audio_path.name)
            result = self._run_single(engine, audio_path, ref_transcripts, diar_ref_transcripts) ## diar modif ##
            self._log_result_summary(result)
            return (ei, fi), result

        if parallel_cells:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                futures = {executor.submit(_task, cell): cell for cell in parallel_cells}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        key, result = future.result()
                        results_map[key] = result
                    except Exception as exc:
                        cell = futures[future]
                        log.error("Cell (%s, %s) raised: %s", cell[2].engine_id, cell[3].name, exc)
                        results_map[(cell[0], cell[1])] = TranscriptionResult(
                            engine_id=cell[2].engine_id,
                            engine_type=cell[2].config.engine_type,
                            audio_file=cell[3].stem,
                            hypothesis="",
                            reference=None,
                            metrics=TranscriptionMetrics(),
                            error=f"Unhandled executor exception: {exc}",
                            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        )

        # 3. Restore deterministic order: engine-major, file-minor
        return [
            results_map[(ei, fi)]
            for ei in range(len(self._engines))
            for fi in range(len(audio_files))
            if (ei, fi) in results_map
        ]

            ## end modif ##

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_engines(self, whitelist: list[str] | None) -> list[BaseSTTEngine]:
        engines: list[BaseSTTEngine] = []
        for eng_cfg in self.config.engines:
            if not eng_cfg.enabled:
                log.debug("Skipping disabled engine: %s", eng_cfg.engine_id)
                continue
            if whitelist and eng_cfg.engine_id not in whitelist:
                log.debug("Skipping engine (not in whitelist): %s", eng_cfg.engine_id)
                continue
            try:
                engine = EngineFactory.create(eng_cfg)
                engines.append(engine)
                log.info(
                    "Registered engine: %s (%s)%s",
                    eng_cfg.engine_id,
                    eng_cfg.engine_type,
                    "  ⚠ experimental" if eng_cfg.experimental else "",
                )
            except Exception as exc:
                log.error("Failed to create engine '%s': %s", eng_cfg.engine_id, exc)
        if not engines:
            log.warning("No engines were initialised — check config and whitelist.")
        return engines

    def _run_single(
        self,
        engine: BaseSTTEngine,
        audio_path: Path,
        ref_transcripts: dict[str, str],
        diar_ref_transcripts: dict[str, str], ## diar modif ##
    ) -> TranscriptionResult:
        audio_duration = BaseSTTEngine.get_audio_duration(audio_path)
        reference = ref_transcripts.get(audio_path.stem)
        diar_reference = diar_ref_transcripts.get(audio_path.stem) if diar_ref_transcripts else None ## diar modif ##
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()


        hypothesis = ""
        raw_response = None
        error_msg: str | None = None
        latency_sec = 0.0

        max_attempts = self.config.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                hypothesis, raw_response = engine.transcribe(audio_path)
                latency_sec = time.perf_counter() - t0
                error_msg = None
                break
            except Exception as exc:
                latency_sec = time.perf_counter() - t0 if "t0" in locals() else 0.0

                error_msg = f"[Attempt {attempt}/{max_attempts}] {type(exc).__name__}: {exc}"
                log.warning("    %s", error_msg)
                if attempt < max_attempts:
                    log.info("    Retrying in %.1f s …", self.config.retry_delay_sec)
                    time.sleep(self.config.retry_delay_sec)

        metrics = self._compute_metrics(
            hypothesis=hypothesis,
            reference=reference,
            latency_sec=latency_sec,
            audio_duration_sec=audio_duration,
            cost_per_min=engine.config.cost_per_audio_minute_usd,
            raw_response=raw_response,
            domain_lexicon=self.config.domain_lexicon,
            metrics_cfg=self.config.metrics,
        )

        # Extract diarization segments from raw response
        diarization_segments: list[DiarizationSegment] = []
        if raw_response and "diarization_segments" in raw_response:
            for seg_dict in raw_response["diarization_segments"]:
                diarization_segments.append(DiarizationSegment(**seg_dict))
            if diarization_segments:
                metrics.num_speakers = raw_response.get(
                    "num_speakers",
                    len({s.speaker for s in diarization_segments}),
                )

            ## diar modif ##

            if diar_reference and diarization_segments:
                from metrics import compute_diarization_metrics
                try:
                    d_metrics = compute_diarization_metrics(diar_reference, diarization_segments)
                    metrics.diarization_error_rate = d_metrics["der"]
                    metrics.jaccard_error_rate = d_metrics["jer"]
                except Exception as exc:
                    log.error("Failed to compute diarization metrics for %s: %s", audio_path.stem, exc)

            ## diar modif ##

        return TranscriptionResult(
            engine_id=engine.engine_id,
            engine_type=engine.config.engine_type,
            audio_file=audio_path.stem,
            hypothesis=hypothesis,
            reference=reference,
            metrics=metrics,
            diarization_segments=diarization_segments,
            error=error_msg,
            raw_response=raw_response,
            timestamp_utc=timestamp_utc,
        )

    @staticmethod
    def _compute_metrics(
        hypothesis: str,
        reference: str | None,
        latency_sec: float,
        audio_duration_sec: float,
        cost_per_min: float,
        raw_response: dict | None,
        domain_lexicon: list[str],
        metrics_cfg: MetricsConfig,
    ) -> TranscriptionMetrics:
        """
        Assemble a TranscriptionMetrics object from available data.
        Optional metrics are controlled by MetricsConfig feature flags.
        """
        rtf = compute_rtf(latency_sec, audio_duration_sec)
        cost = estimate_cost(audio_duration_sec, cost_per_min)
        confidence = extract_confidence(raw_response)

        ## start modif ##
        # Extract TTFW safely (batch engines will simply yield None)
        ttfw = raw_response.get("time_to_first_token_sec") if raw_response else None
        ## end modif ##

        m = TranscriptionMetrics(
            latency_sec=round(latency_sec, 4),
            ## start modif ##
            time_to_first_token_sec=ttfw,
            ## end modif ##
            audio_duration_sec=round(audio_duration_sec, 4),
            real_time_factor=rtf,
            estimated_cost_usd=cost,
            confidence=confidence,
        )

        if not (reference and hypothesis):
            return m

        # ── Always-on accuracy metrics ────────────────────────────────────
        wer_result = compute_wer(reference, hypothesis)
        m.word_error_rate  = wer_result["wer"]
        m.word_accuracy    = wer_result["word_accuracy"]
        m.substitutions    = wer_result["substitutions"]
        m.deletions        = wer_result["deletions"]
        m.insertions       = wer_result["insertions"]
        m.character_error_rate = compute_cer(reference, hypothesis)
        m.match_error_rate = compute_mer(reference, hypothesis)

        wip_wil = compute_wip_wil(reference, hypothesis)
        m.word_info_preserved = wip_wil["wip"]
        m.word_info_lost      = wip_wil["wil"]

        if metrics_cfg.enable_norm_wer:
            nwer = compute_norm_wer(reference, hypothesis)
            m.norm_word_error_rate = nwer["wer"]

        # ── Optional metrics ──────────────────────────────────────────────

        if metrics_cfg.enable_onset_wer:
            n_toks = getattr(metrics_cfg, "onset_wer_tokens", 3)
            m.onset_wer = compute_onset_wer(reference, hypothesis, n_tokens=n_toks)

        if metrics_cfg.enable_tner:
            m.term_error_rate = compute_tner(reference, hypothesis, domain_lexicon)

        if metrics_cfg.enable_har:
            m.hallucination_rate = compute_har(reference, hypothesis)

        if metrics_cfg.enable_punctuation:
            punc = compute_punctuation_metrics(reference, hypothesis)
            m.punctuation_f1_macro    = punc["macro_f1"]
            m.punctuation_error_rate  = punc["pper"]

        return m

    @staticmethod
    def _log_result_summary(result: TranscriptionResult) -> None:
        status = "✓" if result.success else "✗"
        wer_str = (
            f"WER={result.metrics.word_error_rate:.2%}"
            if result.metrics.word_error_rate is not None
            else "WER=N/A"
        )
        tner_str = (
            f"TNER={result.metrics.term_error_rate:.2%}"
            if result.metrics.term_error_rate is not None
            else ""
        )
        extra = tner_str
        log.info(
            "    %s latency=%.2fs rtf=%.2f %s%s cost=$%.5f",
            status,
            result.metrics.latency_sec,
            result.metrics.real_time_factor,
            wer_str,
            f"  {extra}" if extra else "",
            result.metrics.estimated_cost_usd,
        )
        if result.error:
            log.warning("    Error: %s", result.error)
