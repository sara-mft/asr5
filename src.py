"""
Azure Speech-to-Text Benchmarking Suite  [v2]
==============================================
Entry point for running benchmarks across multiple Azure STT solutions.

Usage
-----
    python main.py --config config.yaml --audio-dir ./audio --output-dir ./results
    --parallel N    Run N engines/files in parallel (default: 1 = sequential)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmark_runner import BenchmarkRunner
from config import BenchmarkConfig
from reporters import (
    JSONReporter,
    CSVReporter,
    ConsoleReporter,
    ExcelReporter
)


def setup_logging(log_level: str, log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple Speech-to-Text solutions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python main.py --config config.yaml --audio-dir ./audio --output-dir ./results

  # With references (enables WER/CER/all accuracy metrics)
  python main.py --config config.yaml --audio-dir ./audio --ref-dir ./transcripts

  # Parallel run across 4 workers
  python main.py --config config.yaml --audio-dir ./audio \\
      --parallel 4 --output-formats json excel console

  # Specific engines only
  python main.py --config config.yaml --audio-dir ./audio \\
      --engines azure_speech_fr azure_mai_transcribe
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing .wav audio files to benchmark",
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        help="Optional directory with reference .txt transcripts (same stem as audio files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where benchmark results will be written (default: ./results)",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=None,
        help="Whitelist of engine IDs to run (runs all configured engines if omitted)",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        choices=["json", "csv", "excel", "console"],
        default=["json", "excel", "console"],
        help="Output formats (default: json excel console)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel workers (default: use config.yaml max_workers). "
            "Set to 1 for sequential execution (safer for latency measurements)."
        ),
    )
    ## diar modif ##
    parser.add_argument(
        "--diar-ref-dir",
        type=Path,
        default=None,
        help="Optional directory with diarization references (format: [MM:SS.s - MM:SS.s] Speaker: Text)",
    )
    ## diar modif ##


    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write log output to a file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    log = logging.getLogger("main")

    # ── Load configuration ────────────────────────────────────────────────────
    log.info("Loading configuration from: %s", args.config)
    try:
        config = BenchmarkConfig.from_yaml(args.config)
    except FileNotFoundError:
        log.error("Config file not found: %s", args.config)
        return 1
    except Exception as exc:
        log.error("Failed to parse config: %s", exc)
        return 1

    # CLI --parallel overrides config.yaml max_workers
    if args.parallel is not None:
        config.max_workers = args.parallel
        log.info("Parallel workers overridden by CLI: %d", config.max_workers)

    # ── Collect audio files ───────────────────────────────────────────────────
    audio_files = sorted(args.audio_dir.glob("*.wav"))
    if not audio_files:
        log.error("No .wav files found in: %s", args.audio_dir)
        return 1
    log.info("Found %d audio file(s) in %s", len(audio_files), args.audio_dir)

    # ── Load reference transcripts ────────────────────────────────────────────
    ref_transcripts: dict[str, str] = {}
    if args.ref_dir and args.ref_dir.exists():
        for txt_path in sorted(args.ref_dir.glob("*.txt")):
            ref_transcripts[txt_path.stem] = txt_path.read_text(encoding="utf-8").strip()
        log.info("Loaded %d reference transcript(s)", len(ref_transcripts))
    elif args.ref_dir:
        log.warning("Reference directory not found: %s — accuracy metrics disabled", args.ref_dir)

    ## diar modif ##
    # ── Load reference diarization transcripts ────────────────────────────────────────────
    diar_ref_transcripts: dict[str, str] = {}
    if args.diar_ref_dir and args.diar_ref_dir.exists():
        for txt_path in sorted(args.diar_ref_dir.glob("*.txt")):
            diar_ref_transcripts[txt_path.stem] = txt_path.read_text(encoding="utf-8").strip()
        log.info("Loaded %d diarization reference transcript(s)", len(diar_ref_transcripts))


    ## diar modif ##

    # ── Run benchmarks ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(config, engine_whitelist=args.engines)

    log.info("Starting benchmark run …")
    
    ## diar modif ##
    #results = runner.run(audio_files, ref_transcripts)
    results = runner.run(audio_files, ref_transcripts, diar_ref_transcripts)

    ## diar modif ##
    log.info("Benchmark complete — %d result(s) collected", len(results))

    # ── Write reports ─────────────────────────────────────────────────────────
    reporters = []
    fmt = args.output_formats

    if "json" in fmt:
        reporters.append(
            JSONReporter(
                args.output_dir / "detailed_results.json",
            )
        )
    if "csv" in fmt:
        reporters.append(CSVReporter(args.output_dir / "detailed_results.csv"))
    if "excel" in fmt:
        reporters.append(ExcelReporter(args.output_dir / "detailed_results.xlsx"))

    if "console" in fmt:
        reporters.append(ConsoleReporter())

    for reporter in reporters:
        try:
            reporter.write(results)
        except Exception as exc:
            log.error("Reporter %s failed: %s", type(reporter).__name__, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
