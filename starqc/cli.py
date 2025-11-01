"""Command-line interface for StarQC."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .clean import clean_lfp
from .config import PipelineConfig, get_default_config
from .provenance import dump_report


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "clean":
        return _cmd_clean(args)
    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="starqc", description="StarQC data cleaner")
    subparsers = parser.add_subparsers(dest="command")

    clean_parser = subparsers.add_parser("clean", help="Clean a .npy voltage file")
    clean_parser.add_argument("input", type=Path, help="Path to input .npy file")
    clean_parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    clean_parser.add_argument("--stim", type=Path, help="CSV of stimulation times (seconds)")
    clean_parser.add_argument(
        "--out",
        type=Path,
        required=False,
        default=None,
        help="Output .npy for cleaned data (default: <input>_clean.npy)",
    )
    clean_parser.add_argument(
        "--report",
        type=Path,
        required=False,
        default=None,
        help="Output JSON report path (default: <input>_report.json)",
    )
    clean_parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Disable common-median referencing",
    )
    clean_parser.add_argument(
        "--voltage-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Expected voltage range used for clipping detection",
    )
    return parser


def _cmd_clean(args: argparse.Namespace) -> int:
    data = np.load(args.input)
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    config = get_default_config()
    if args.no_reference:
        config = replace(config, standardize=replace(config.standardize, re_reference=False))

    stim_times = _load_stim_csv(args.stim) if args.stim else None

    # Derive default outputs if not provided
    out_path = args.out or args.input.with_name(args.input.stem + "_clean.npy")
    report_path = args.report or args.input.with_name(args.input.stem + "_report.json")

    clean, report = clean_lfp(
        data,
        fs=float(args.fs),
        stim_times_s=stim_times,
        voltage_range=tuple(args.voltage_range) if args.voltage_range else None,
        config=config,
    )

    np.save(out_path, clean)
    dump_report(report_path, _serialise_report(report))
    return 0


def _load_stim_csv(path: Path) -> List[float]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    values = []
    for token in content.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _serialise_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    serialised: Dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            serialised[key] = value.tolist()
        elif isinstance(value, Mapping):
            serialised[key] = _serialise_report(value)
        elif isinstance(value, list):
            serialised[key] = [
                _serialise_report(item) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            serialised[key] = value
    return serialised


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
