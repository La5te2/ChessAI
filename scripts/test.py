#!/usr/bin/env python3
"""Compare a Gadus FCPI current model with the exact model that started its run."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def latest_fcpi_summary() -> Path:
    """Find the most recently modified Gadus FCPI summary."""
    summaries = list((ROOT / "data" / "runs").glob("fcpi_*/summary.json"))
    if not summaries:
        raise FileNotFoundError(
            "no FCPI summary found under data/runs; provide --summary"
        )
    return max(summaries, key=lambda path: path.stat().st_mtime)


def repository_path(value: str | Path) -> Path:
    """Resolve a command-line or summary path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    """Use a stable repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def arena_executable(argument: str | None) -> Path:
    """Select the platform-specific Gadus Arena executable."""
    if argument:
        return repository_path(argument)
    name = "arena.exe" if os.name == "nt" else "arena"
    return ROOT / "build" / "gadus" / name


def parser() -> argparse.ArgumentParser:
    """Describe a reproducible cumulative-strength Arena test."""
    result = argparse.ArgumentParser(
        description=(
            "Run a paired Gadus Arena match between an FCPI current model and "
            "the initial model recorded by the same run."
        )
    )
    result.add_argument(
        "--summary",
        help="FCPI summary.json; defaults to the newest data/runs/fcpi_*/summary.json",
    )
    result.add_argument("--candidate", help="Override summary current_model")
    result.add_argument("--baseline", help="Override summary initial_model")
    result.add_argument("--arena", help="Override the Gadus Arena executable")
    result.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    result.add_argument("--precision", choices=("fp32", "bf16"))
    result.add_argument("--games", type=int, default=2000)
    result.add_argument("--games-in-flight", type=int, default=256)
    result.add_argument("--max-plies", type=int, default=240)
    result.add_argument("--opening-book", default="data/openings.gen.bin")
    result.add_argument("--book-plies", type=int, default=8)
    result.add_argument("--max-book-positions", type=int, default=50000)
    result.add_argument(
        "--search-type", default="closed", choices=("closed", "only-mcts")
    )
    result.add_argument("--sims", type=int, default=0)
    result.add_argument("--mcts-batch-size", type=int, default=512)
    result.add_argument("--movetime-ms", type=float, default=0.0)
    result.add_argument("--mcts-games", type=int, default=2)
    result.add_argument("--mcts-sims", type=int, default=10000)
    result.add_argument("--mcts-movetime-ms", type=float, default=0.0)
    result.add_argument("--repetition-policy-penalty", type=float, default=1.0)
    result.add_argument("--instant-mate-first", type=int, default=1, choices=(0, 1))
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument(
        "--output",
        help="Output directory; defaults to data/tests/gadus_vs_initial_<timestamp>",
    )
    return result


def require_file(path: Path, label: str) -> None:
    """Fail before launching Arena when a required input is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def run_arena(command: list[str], log, phase: str) -> dict:
    """Stream one Arena phase and decode its final JSON summary."""
    heading = f"\n=== {phase} ===\n"
    print(heading, end="", flush=True)
    log.write(heading)
    log.flush()
    transcript: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
        transcript.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Gadus Arena {phase} exited with code {return_code}; see {log.name}"
        )

    output_text = "".join(transcript)
    finished = output_text.rfind("arena: finished")
    json_start = output_text.find("{", finished if finished >= 0 else 0)
    if json_start < 0:
        raise RuntimeError(f"Arena {phase} summary was not found; see {log.name}")
    try:
        summary, _ = json.JSONDecoder().raw_decode(output_text[json_start:])
        return summary
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Arena {phase} summary could not be parsed: {error}; see {log.name}"
        ) from error


def main() -> int:
    """Run the closed benchmark followed by a two-game startpos MCTS match."""
    args = parser().parse_args()
    summary_path = (
        repository_path(args.summary) if args.summary else latest_fcpi_summary()
    )
    require_file(summary_path, "FCPI summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    candidate_value = args.candidate or summary.get("current_model")
    baseline_value = args.baseline or summary.get("initial_model")
    if not candidate_value or not baseline_value:
        raise ValueError(
            "summary must contain current_model and initial_model, or both models "
            "must be supplied explicitly"
        )

    candidate = repository_path(candidate_value)
    baseline = repository_path(baseline_value)
    arena = arena_executable(args.arena)
    require_file(candidate, "candidate model")
    require_file(baseline, "baseline model")
    require_file(arena, "Gadus Arena executable")

    opening_book = repository_path(args.opening_book) if args.opening_book else None
    if opening_book is not None:
        require_file(opening_book, "opening book")

    precision = args.precision or summary.get("precision", "bf16")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        repository_path(args.output)
        if args.output
        else ROOT / "data" / "tests" / f"gadus_vs_initial_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "info.log"
    closed_pgn_path = output / "closed.pgn"
    mcts_pgn_path = output / "startpos-mcts.pgn"
    result_path = output / "summary.json"

    common_command = [
        str(arena),
        "--candidate",
        str(candidate),
        "--baseline",
        str(baseline),
        "--device",
        args.device,
        "--precision",
        precision,
        "--max-plies",
        str(args.max_plies),
        "--mcts-batch-size",
        str(args.mcts_batch_size),
        "--repetition-policy-penalty",
        str(args.repetition_policy_penalty),
        "--instant-mate-first",
        str(args.instant_mate_first),
        "--min-net-wins",
        "0",
        "--seed",
        str(args.seed),
    ]
    closed_command = common_command + [
        "--games",
        str(args.games),
        "--games-in-flight",
        str(args.games_in_flight),
        "--opening-book",
        str(opening_book) if opening_book is not None else "",
        "--book-plies",
        str(args.book_plies),
        "--max-book-positions",
        str(args.max_book_positions),
        "--search-type",
        args.search_type,
        "--sims",
        str(args.sims),
        "--movetime-ms",
        str(args.movetime_ms),
        "--pgn-output",
        str(closed_pgn_path),
        "--log-every",
        str(args.log_every),
    ]
    mcts_command = common_command + [
        "--games",
        str(args.mcts_games),
        "--games-in-flight",
        str(args.mcts_games),
        "--opening-book",
        "",
        "--book-plies",
        str(args.book_plies),
        "--max-book-positions",
        str(args.max_book_positions),
        "--search-type",
        "only-mcts",
        "--sims",
        str(args.mcts_sims),
        "--movetime-ms",
        str(args.mcts_movetime_ms),
        "--pgn-output",
        str(mcts_pgn_path),
        "--log-every",
        "1",
    ]

    print("Gadus cumulative Arena test")
    print(f"summary={display_path(summary_path)}")
    print(f"candidate={display_path(candidate)}")
    print(f"baseline={display_path(baseline)}")
    print(
        f"games={args.games} search_type={args.search_type} "
        f"device={args.device} precision={precision}"
    )
    print(
        f"startpos_mcts_games={args.mcts_games} "
        f"startpos_mcts_sims={args.mcts_sims}"
    )
    print(f"output={display_path(output)}")

    with log_path.open("w", encoding="utf-8", newline="") as log:
        closed_summary = run_arena(closed_command, log, "closed")
        mcts_summary = run_arena(mcts_command, log, "startpos MCTS")

    result = {
        "source_summary": display_path(summary_path),
        "run_id": summary.get("run_id"),
        "candidate": display_path(candidate),
        "baseline": display_path(baseline),
        "closed": closed_summary,
        "startpos_mcts": mcts_summary,
    }
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"result={display_path(result_path)}")
    print(f"closed_pgn={display_path(closed_pgn_path)}")
    print(f"startpos_mcts_pgn={display_path(mcts_pgn_path)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"test error: {error}", file=sys.stderr)
        raise SystemExit(1)
