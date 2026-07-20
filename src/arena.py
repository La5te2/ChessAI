"""Paired model comparison based on game results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
import re
import textwrap
from typing import Dict, List

import chess
import chess.pgn
import numpy as np

from config import CONFIDENCE_Z, DEVICE
from game_rules import game_is_over, game_result, game_termination_text
from model import load_model
from opening_book import make_arena_specs
from search import SearchOptions, UnifiedSearch, VALID_SEARCH_TYPES


def progress_print(enabled: bool, *parts):
    if enabled:
        print(*parts, flush=True)


def file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def trace_search_info(info: Dict, root_topn: int) -> Dict:
    root_topn = max(0, int(root_topn))
    keys = (
        "search_type",
        "search_backend",
        "best_move",
        "best_san",
        "value",
        "sims_completed",
        "mcts_completed",
        "mcts_dynamic_target",
        "mcts_soft_cap",
        "uncertainty",
        "c_puct_initial",
        "c_puct_root",
        "c_puct_base",
        "c_puct_factor",
        "fpu_reduction",
        "fpu_root",
        "q_prior_penalty_reduction",
        "q_prior_penalty_root",
        "virtual_loss",
        "nodes",
        "expanded_nodes",
        "nn_batches",
        "mcts_move",
        "elapsed_ms",
    )
    payload = {key: info.get(key) for key in keys if key in info}
    if root_topn > 0:
        payload["root"] = list(info.get("root") or [])[:root_topn]
    return json_safe(payload)


def pgn_search_comment(owner: str, info: Dict) -> str:
    parts = [
        f"owner={owner}",
        f"backend={info.get('search_backend', '?')}",
        f"best={info.get('best_san', info.get('best_move', '?'))}",
    ]
    if "mcts_completed" in info and "mcts_soft_cap" in info:
        parts.append(f"sims={info.get('mcts_completed')}/{info.get('mcts_soft_cap')}")
    elif "sims_completed" in info and "mcts_soft_cap" in info:
        parts.append(f"sims={info.get('sims_completed')}/{info.get('mcts_soft_cap')}")
    if "value" in info:
        parts.append(f"value={float(info.get('value')):+.3f}")
    root = []
    for row in list(info.get("root") or [])[:3]:
        root.append(
            f"{row.get('san', row.get('uci', '?'))}:"
            f"v{row.get('visits', 0)}:"
            f"q{float(row.get('q', 0.0)):+.3f}"
        )
    if root:
        parts.append("root=" + ", ".join(root))
    return " ".join(parts).replace("{", "(").replace("}", ")")


def pgn_result_and_termination(
    board: chess.Board,
    ply: int,
    max_plies: int,
):
    termination = game_termination_text(board)
    if termination is not None:
        return game_result(board), termination
    if int(max_plies) > 0 and int(ply) >= int(max_plies):
        return "1/2-1/2", "max plies"
    return "*", "unfinished"


def render_pgn(game: chess.pgn.Game, columns: int = 88) -> str:
    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=True,
        columns=None if int(columns) <= 0 else int(columns),
    )
    text = game.accept(exporter)
    if int(columns) <= 0:
        return text
    return wrap_pgn_comments(text, columns=int(columns))


def wrap_pgn_comments(text: str, columns: int) -> str:
    wrapped_lines = []
    width = max(40, int(columns) - 4)
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            wrapped_lines.append(line)
            continue

        content = stripped[1:-1].strip()
        content = re.sub(r",(?=\S)", ", ", content)
        parts = textwrap.wrap(
            content,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if len(parts) <= 1:
            wrapped_lines.append("{ " + content + " }")
            continue

        wrapped_lines.append("{ " + parts[0])
        for part in parts[1:-1]:
            wrapped_lines.append("  " + part)
        wrapped_lines.append("  " + parts[-1] + " }")
    return "\n".join(wrapped_lines)


def score_from_result(result: str, candidate_color: chess.Color) -> float:
    if result in {"1/2-1/2", "*"}:
        return 0.5
    candidate_won = (
        result == "1-0" and candidate_color == chess.WHITE
    ) or (
        result == "0-1" and candidate_color == chess.BLACK
    )
    return 1.0 if candidate_won else 0.0


def outcome_from_score(score: float) -> str:
    if score == 1.0:
        return "win"
    if score == 0.0:
        return "loss"
    return "draw"


def normal_ci(scores, z=CONFIDENCE_Z):
    count = len(scores)
    if count == 0:
        return 0.0, 0.0, 0.0
    array = np.asarray(scores, dtype=np.float64)
    mean = float(array.mean())
    if count == 1:
        return mean, mean, mean
    standard_error = float(array.std(ddof=1) / math.sqrt(count))
    return (
        mean,
        max(0.0, mean - z * standard_error),
        min(1.0, mean + z * standard_error),
    )


def elo_from_score(score):
    score = min(0.999, max(0.001, float(score)))
    return 400.0 * math.log10(score / (1.0 - score))


@dataclass
class ArenaGameState:
    game_id: int
    candidate_color: chess.Color
    board: chess.Board
    game: object
    node: object
    trace: List[Dict]
    ply: int = 0


def make_game_state(game_id, start_fen, candidate_color, collect_pgn):
    board = chess.Board(start_fen)
    game = chess.pgn.Game() if collect_pgn else None
    if game is not None:
        game.headers["Event"] = "ChessAI Arena"
        game.headers["Round"] = str(game_id)
        game.headers["White"] = (
            "candidate" if candidate_color == chess.WHITE else "baseline"
        )
        game.headers["Black"] = (
            "baseline" if candidate_color == chess.WHITE else "candidate"
        )
        if start_fen != chess.Board().fen():
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = start_fen
    return ArenaGameState(
        game_id=int(game_id),
        candidate_color=candidate_color,
        board=board,
        game=game,
        node=game,
        trace=[],
    )


def apply_game_move(
    state: ArenaGameState,
    search_result,
    owner: str,
    pgn_comments: bool,
    trace_root_topn: int,
    collect_trace: bool,
):
    board = state.board
    info = dict(search_result.info or {})
    move = search_result.move
    if move not in board.legal_moves:
        raise RuntimeError(
            "arena search returned illegal move: "
            f"game={state.game_id} ply={state.ply + 1} owner={owner} "
            f"move={move.uci() if move is not None else None} "
            f"fen={board.fen()} best={info.get('best_move')} "
            f"root={json.dumps(trace_search_info(info, trace_root_topn), ensure_ascii=False)}"
        )

    san = safe_san(board, move)
    if collect_trace:
        state.trace.append({
            "game_id": state.game_id,
            "ply": state.ply + 1,
            "fen": board.fen(),
            "move": move.uci(),
            "san": san,
            "owner": owner,
            "candidate_color": (
                "white" if state.candidate_color == chess.WHITE else "black"
            ),
            "search_backend": info.get("search_backend"),
            "search": trace_search_info(info, trace_root_topn),
        })
    if state.node is not None:
        state.node = state.node.add_variation(move)
        if pgn_comments:
            state.node.comment = pgn_search_comment(owner, info)
    board.push(move)
    state.ply += 1


def finish_game_state(
    state: ArenaGameState,
    max_plies: int,
    pgn_columns: int,
):
    result, termination = pgn_result_and_termination(
        state.board,
        ply=state.ply,
        max_plies=max_plies,
    )
    if state.game is not None:
        state.game.headers["Result"] = result
        state.game.headers["Termination"] = termination
    return {
        "game_id": state.game_id,
        "candidate_color": state.candidate_color,
        "result": result,
        "score": score_from_result(result, state.candidate_color),
        "plies": state.ply,
        "trace": state.trace,
        "pgn": (
            render_pgn(state.game, columns=pgn_columns)
            if state.game is not None
            else None
        ),
    }


def play_batched_games(
    candidate_searcher,
    baseline_searcher,
    specs,
    games_in_flight,
    max_plies,
    pgn_comments=False,
    pgn_columns=88,
    trace_root_topn=12,
    collect_trace=True,
    collect_pgn=True,
    progress=True,
):
    completed = []
    total_games = len(specs)
    next_spec = 0
    active: List[ArenaGameState] = []

    while next_spec < total_games or active:
        while next_spec < total_games and len(active) < games_in_flight:
            start_fen, candidate_color = specs[next_spec]
            active.append(
                make_game_state(
                    next_spec + 1,
                    start_fen,
                    candidate_color,
                    collect_pgn=collect_pgn,
                )
            )
            next_spec += 1

        finished_before_search = [
            state
            for state in active
            if game_is_over(state.board)
            or state.ply >= int(max_plies)
        ]
        if finished_before_search:
            for state in finished_before_search:
                record = finish_game_state(
                    state, max_plies, pgn_columns
                )
                completed.append(record)
                progress_print(
                    progress,
                    f"arena game {state.game_id}/{total_games}: "
                    f"candidate_color={'white' if state.candidate_color == chess.WHITE else 'black'} "
                    f"result={record['result']} candidate_score={record['score']:.1f} "
                    f"plies={record['plies']}",
                )
            finished_ids = {state.game_id for state in finished_before_search}
            active = [state for state in active if state.game_id not in finished_ids]
            continue

        candidate_states = [
            state for state in active if state.board.turn == state.candidate_color
        ]
        baseline_states = [
            state for state in active if state.board.turn != state.candidate_color
        ]
        if candidate_states:
            results = candidate_searcher.search_many(
                [state.board for state in candidate_states]
            )
            for state, result in zip(candidate_states, results):
                apply_game_move(
                    state,
                    result,
                    owner="candidate",
                    pgn_comments=pgn_comments,
                    trace_root_topn=trace_root_topn,
                    collect_trace=collect_trace,
                )
        if baseline_states:
            results = baseline_searcher.search_many(
                [state.board for state in baseline_states]
            )
            for state, result in zip(baseline_states, results):
                apply_game_move(
                    state,
                    result,
                    owner="baseline",
                    pgn_comments=pgn_comments,
                    trace_root_topn=trace_root_topn,
                    collect_trace=collect_trace,
                )

        just_finished = [
            state
            for state in active
            if game_is_over(state.board)
            or state.ply >= int(max_plies)
        ]
        for state in just_finished:
            record = finish_game_state(state, max_plies, pgn_columns)
            completed.append(record)
            progress_print(
                progress,
                f"arena game {state.game_id}/{total_games}: "
                f"candidate_color={'white' if state.candidate_color == chess.WHITE else 'black'} "
                f"result={record['result']} candidate_score={record['score']:.1f} "
                f"plies={record['plies']}",
            )
        if just_finished:
            finished_ids = {state.game_id for state in just_finished}
            active = [state for state in active if state.game_id not in finished_ids]

    completed.sort(key=lambda row: row["game_id"])
    return completed


def evaluate_models(
    candidate_path,
    baseline_path,
    games=100,
    sims=80,
    games_in_flight=32,
    device=DEVICE,
    max_plies=240,
    seed=2026,
    opening_book="data/openings.bin",
    book_plies=8,
    max_book_positions=50000,
    mcts_batch_size=32,
    movetime_ms=5000,
    search_type="closed",
    c_puct=1.5,
    c_puct_base=19652.0,
    c_puct_factor=1.0,
    fpu_reduction=0.15,
    repetition_policy_penalty=0.0,
    instant_mate_first=False,
    pgn_output=None,
    trace_output=None,
    pgn_comments=False,
    pgn_columns=88,
    trace_root_topn=12,
    log_every=1000,
    progress=True,
):
    candidate_hash = file_sha256(candidate_path)
    baseline_hash = file_sha256(baseline_path)
    progress_print(
        progress,
        "arena: start",
        f"candidate={candidate_path}",
        f"candidate_sha256={candidate_hash}",
        f"baseline={baseline_path}",
        f"baseline_sha256={baseline_hash}",
        f"games={games}",
        f"sims={sims}",
        f"device={device}",
        f"games_in_flight={games_in_flight}",
    )
    specs = make_arena_specs(
        games=games,
        seed=seed,
        opening_book=opening_book,
        book_plies=book_plies,
        max_positions=max_book_positions,
        paired=True,
    )
    if not specs:
        raise ValueError("no arena game specs generated")
    progress_print(
        progress,
        f"arena: generated {len(specs)} paired game specs "
        f"from {len({fen for fen, _ in specs})} unique start positions",
    )

    games_in_flight = max(1, min(len(specs), int(games_in_flight)))
    progress_print(progress, f"arena: loading candidate model on {device}")
    candidate = load_model(candidate_path, device=device)
    progress_print(progress, f"arena: loading baseline model on {device}")
    baseline = load_model(baseline_path, device=device)
    candidate_arch = candidate.arch() if hasattr(candidate, "arch") else {}
    baseline_arch = baseline.arch() if hasattr(baseline, "arch") else {}
    options = SearchOptions(
        search_type=search_type,
        mcts_sims=sims,
        mcts_batch_size=mcts_batch_size,
        time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        c_puct=c_puct,
        c_puct_base=c_puct_base,
        c_puct_factor=c_puct_factor,
        fpu_reduction=fpu_reduction,
        repetition_policy_penalty=repetition_policy_penalty,
        instant_mate_first=instant_mate_first,
    )
    candidate_searcher = UnifiedSearch(candidate, replace(options), device=device)
    baseline_searcher = UnifiedSearch(baseline, replace(options), device=device)
    candidate_search_backend = candidate_searcher.backend.name
    baseline_search_backend = baseline_searcher.backend.name
    progress_print(
        progress,
        "arena: models ready",
        f"candidate_arch={candidate_arch.get('type')}",
        f"candidate_backend={candidate_search_backend}",
        f"baseline_arch={baseline_arch.get('type')}",
        f"baseline_backend={baseline_search_backend}",
    )

    records = play_batched_games(
        candidate_searcher=candidate_searcher,
        baseline_searcher=baseline_searcher,
        specs=specs,
        games_in_flight=games_in_flight,
        max_plies=max_plies,
        pgn_comments=pgn_comments,
        pgn_columns=pgn_columns,
        trace_root_topn=trace_root_topn,
        collect_trace=bool(trace_output),
        collect_pgn=bool(pgn_output),
        progress=progress,
    )

    scores = [float(record["score"]) for record in records]
    counts = {"win": 0, "draw": 0, "loss": 0}
    raw_results = {}
    plies_total = sum(int(record["plies"]) for record in records)
    traces = []
    pgns = []
    for record in records:
        counts[outcome_from_score(record["score"])] += 1
        result = str(record["result"])
        raw_results[result] = raw_results.get(result, 0) + 1
        traces.extend(record["trace"])
        if record["pgn"] is not None:
            pgns.append(record["pgn"])

    score, ci_low, ci_high = normal_ci(scores)

    games_done = len(scores)
    wins = counts["win"]
    draws = counts["draw"]
    losses = counts["loss"]
    game_summary = {
        "games": games_done,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "net_wins": wins - losses,
        "score": score,
        "score_ci_low": ci_low,
        "score_ci_high": ci_high,
        "elo_diff": elo_from_score(score),
        "elo_ci_low": elo_from_score(ci_low),
        "elo_ci_high": elo_from_score(ci_high),
        "raw_results": raw_results,
        "avg_plies": plies_total / max(1, games_done),
    }
    progress_print(progress, "arena game summary:")
    progress_print(
        progress,
        json.dumps(game_summary, ensure_ascii=False, indent=2),
    )

    progress_print(
        progress,
        "arena: finished",
        f"wins={wins}",
        f"draws={draws}",
        f"losses={losses}",
        f"net_wins={wins - losses}",
        f"score={score:.3f}",
        f"elo_diff={elo_from_score(score):+.1f}",
    )
    if pgn_output:
        os.makedirs(os.path.dirname(pgn_output) or ".", exist_ok=True)
        with open(pgn_output, "w", encoding="utf-8") as handle:
            handle.write("\n\n".join(pgns))
            handle.write("\n")
        progress_print(progress, f"arena PGN saved: {pgn_output}")
    if trace_output:
        os.makedirs(os.path.dirname(trace_output) or ".", exist_ok=True)
        with open(trace_output, "w", encoding="utf-8") as handle:
            for row in traces:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        progress_print(progress, f"arena trace saved: {trace_output}")
    effective_sims = 0 if str(search_type) == "closed" else int(sims)
    return {
        "candidate": candidate_path,
        "candidate_sha256": candidate_hash,
        "candidate_arch": candidate_arch,
        "candidate_search_backend": candidate_search_backend,
        "baseline": baseline_path,
        "baseline_sha256": baseline_hash,
        "baseline_arch": baseline_arch,
        "baseline_search_backend": baseline_search_backend,
        **game_summary,
        "search_type": str(search_type),
        "sims_soft_cap": int(effective_sims),
        "mcts_batch_size": int(mcts_batch_size),
        "games_in_flight": int(games_in_flight),
        "movetime_ms": int(movetime_ms),
        "c_puct_initial": float(c_puct),
        "c_puct_base": float(c_puct_base),
        "c_puct_factor": float(c_puct_factor),
        "fpu_reduction": float(fpu_reduction),
        "repetition_policy_penalty": float(repetition_policy_penalty),
        "instant_mate_first": bool(instant_mate_first),
        "opening_book": opening_book,
        "book_plies": int(book_plies),
        "paired_openings": True,
        "unique_start_positions": len({fen for fen, _ in specs}),
        "trace_output": trace_output,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models by paired games"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--games-in-flight", type=int, default=32)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--opening-book", default="data/openings.bin")
    parser.add_argument("--book-plies", type=int, default=8)
    parser.add_argument("--max-book-positions", type=int, default=50000)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=5000)
    parser.add_argument(
        "--search-type",
        choices=sorted(VALID_SEARCH_TYPES),
        default="closed",
    )
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--c-puct-base", type=float, default=19652.0)
    parser.add_argument("--c-puct-factor", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.15)
    parser.add_argument("--repetition-policy-penalty", type=float, default=0.0)
    parser.add_argument(
        "--instant-mate-first",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--pgn-output", default=None)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--pgn-comments", action="store_true", default=False)
    parser.add_argument("--pgn-columns", type=int, default=88)
    parser.add_argument("--trace-root-topn", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = evaluate_models(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        games=args.games,
        sims=args.sims,
        games_in_flight=args.games_in_flight,
        device=args.device,
        max_plies=args.max_plies,
        seed=args.seed,
        opening_book=args.opening_book,
        book_plies=args.book_plies,
        max_book_positions=args.max_book_positions,
        mcts_batch_size=args.mcts_batch_size,
        movetime_ms=args.movetime_ms,
        search_type=args.search_type,
        c_puct=args.c_puct,
        c_puct_base=args.c_puct_base,
        c_puct_factor=args.c_puct_factor,
        fpu_reduction=args.fpu_reduction,
        repetition_policy_penalty=args.repetition_policy_penalty,
        instant_mate_first=args.instant_mate_first,
        pgn_output=args.pgn_output,
        trace_output=args.trace_output,
        pgn_comments=args.pgn_comments,
        pgn_columns=args.pgn_columns,
        trace_root_topn=args.trace_root_topn,
        log_every=args.log_every,
        progress=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
