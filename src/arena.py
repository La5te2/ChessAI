"""Paired model comparison with Stockfish move-quality validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import textwrap
from typing import Dict, List

import chess
import chess.pgn
import numpy as np

from config import CONFIDENCE_Z, DEVICE, STOCKFISH_PATH
from model import load_model
from opening_book import make_arena_specs
from search import SearchOptions, UnifiedSearch, VALID_SEARCH_TYPES
from teacher import (
    StockfishTeacher,
    TeacherConfig,
    move_accuracy_from_regret,
)


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
        "virtual_loss",
        "mcts_time_fraction",
        "nodes",
        "expanded_nodes",
        "nn_batches",
        "mcts_move",
        "mate_plies",
        "mate_topk",
        "mate_nodes",
        "mate_completed",
        "mate_forced_move",
        "mate_reasons",
        "elapsed_ms",
    )
    payload = {key: info.get(key) for key in keys if key in info}
    if root_topn > 0:
        payload["root"] = list(info.get("root") or [])[:root_topn]
    return json_safe(payload)


def pgn_search_comment(owner: str, info: Dict) -> str:
    parts = [
        f"owner={owner}",
        f"best={info.get('best_san', info.get('best_move', '?'))}",
    ]
    if "mcts_completed" in info and "mcts_soft_cap" in info:
        parts.append(f"sims={info.get('mcts_completed')}/{info.get('mcts_soft_cap')}")
    elif "sims_completed" in info and "mcts_soft_cap" in info:
        parts.append(f"sims={info.get('sims_completed')}/{info.get('mcts_soft_cap')}")
    if "value" in info:
        parts.append(f"value={float(info.get('value')):+.3f}")
    if info.get("mate_forced_move"):
        parts.append(f"mate_forced={info.get('mate_forced_move')}")
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
    claim_draws: bool,
):
    outcome = board.outcome(claim_draw=claim_draws)
    result = board.result(claim_draw=claim_draws)
    if result == "*":
        result = "1/2-1/2"

    if outcome is None:
        if int(max_plies) > 0 and int(ply) >= int(max_plies):
            return result, "max plies"
        return result, "unfinished"

    termination = outcome.termination.name.lower().replace("_", " ")
    if outcome.termination == chess.Termination.THREEFOLD_REPETITION and claim_draws:
        termination = "claimed threefold repetition"
    elif outcome.termination == chess.Termination.FIFTY_MOVES and claim_draws:
        termination = "claimed fifty-move rule"
    return result, termination


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


def worker_cache_path(cache_path, worker_index, worker_count):
    if not cache_path or int(worker_count) <= 1:
        return cache_path
    root, ext = os.path.splitext(cache_path)
    return f"{root}.worker{int(worker_index)}{ext or '.sqlite'}"


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


def play_arena_game(
    candidate_searcher,
    baseline_searcher,
    candidate_color,
    max_plies,
    start_fen,
    game_id=None,
    pgn_comments=False,
    pgn_columns=88,
    claim_draws=False,
    trace_root_topn=12,
):
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessAI Arena"
    if game_id is not None:
        game.headers["Round"] = str(game_id)
    game.headers["White"] = "candidate" if candidate_color == chess.WHITE else "baseline"
    game.headers["Black"] = "baseline" if candidate_color == chess.WHITE else "candidate"
    if start_fen != chess.Board().fen():
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = start_fen
    node = game
    trace = []
    ply = 0

    while not board.is_game_over(claim_draw=bool(claim_draws)) and ply < int(max_plies):
        candidate_turn = board.turn == candidate_color
        owner = "candidate" if candidate_turn else "baseline"
        searcher = candidate_searcher if candidate_turn else baseline_searcher
        result = searcher.search(board.copy(stack=False))
        info = dict(result.info or {})
        move = result.move
        if move not in board.legal_moves:
            raise RuntimeError(
                "arena search returned illegal move: "
                f"game={game_id} ply={ply + 1} owner={owner} "
                f"move={move.uci() if move is not None else None} "
                f"fen={board.fen()} best={info.get('best_move')} "
                f"root={json.dumps(trace_search_info(info, trace_root_topn), ensure_ascii=False)}"
            )

        san = safe_san(board, move)
        trace.append({
            "game_id": game_id,
            "ply": ply + 1,
            "fen": board.fen(),
            "move": move.uci(),
            "san": san,
            "owner": owner,
            "candidate_color": "white" if candidate_color == chess.WHITE else "black",
            "search": trace_search_info(info, trace_root_topn),
        })
        node = node.add_variation(move)
        if pgn_comments:
            node.comment = pgn_search_comment(owner, info)
        board.push(move)
        ply += 1

    result_string, termination = pgn_result_and_termination(
        board,
        ply=ply,
        max_plies=max_plies,
        claim_draws=bool(claim_draws),
    )
    game.headers["Result"] = result_string
    game.headers["Termination"] = termination
    return (
        result_string,
        score_from_result(result_string, candidate_color),
        ply,
        trace,
        render_pgn(game, columns=pgn_columns),
    )


def _worker(job):
    (
        worker_index,
        candidate_path,
        baseline_path,
        specs,
        game_id_offset,
        sims,
        device,
        max_plies,
        mcts_batch_size,
        movetime_ms,
        search_type,
        c_puct,
        c_puct_base,
        c_puct_factor,
        fpu_reduction,
        mcts_time_fraction,
        mate_plies,
        mate_topk,
        mate_nodes,
        pgn_comments,
        pgn_columns,
        claim_draws,
        trace_root_topn,
        progress,
    ) = job

    progress_print(
        progress,
        f"arena worker {worker_index}: loading models on {device}",
    )
    candidate = load_model(candidate_path, device=device)
    baseline = load_model(baseline_path, device=device)
    options = SearchOptions(
        search_type=search_type,
        mcts_sims=sims,
        mcts_batch_size=mcts_batch_size,
        time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        c_puct=c_puct,
        c_puct_base=c_puct_base,
        c_puct_factor=c_puct_factor,
        fpu_reduction=fpu_reduction,
        mcts_time_fraction=mcts_time_fraction,
        mate_plies=mate_plies,
        mate_topk=mate_topk,
        mate_nodes=mate_nodes,
    )
    candidate_searcher = UnifiedSearch(candidate, options, device=device)
    baseline_searcher = UnifiedSearch(baseline, options, device=device)

    scores = []
    counts = {"win": 0, "draw": 0, "loss": 0}
    raw_results = {}
    plies_total = 0
    traces = []
    pgns = []

    total_specs = len(specs)
    for local_index, (fen, candidate_color) in enumerate(specs, 1):
        game_id = int(game_id_offset) + local_index
        result, score, plies, trace, pgn = play_arena_game(
            candidate_searcher,
            baseline_searcher,
            candidate_color,
            max_plies,
            fen,
            game_id,
            pgn_comments=pgn_comments,
            pgn_columns=pgn_columns,
            claim_draws=claim_draws,
            trace_root_topn=trace_root_topn,
        )
        scores.append(score)
        counts[outcome_from_score(score)] += 1
        raw_results[result] = raw_results.get(result, 0) + 1
        plies_total += plies
        traces.extend(trace)
        pgns.append(pgn)
        progress_print(
            progress,
            f"arena worker {worker_index}: game {local_index}/{total_specs} "
            f"global_game={game_id} "
            f"candidate_color={'white' if candidate_color == chess.WHITE else 'black'} "
            f"result={result} candidate_score={score:.1f} plies={plies}",
        )

    return {
        "scores": scores,
        "counts": counts,
        "raw_results": raw_results,
        "plies_total": plies_total,
        "traces": traces,
        "pgns": pgns,
    }


def evaluate_move_quality(
    traces: List[Dict],
    teacher_config: TeacherConfig,
    loss_cap_cp: int = 1000,
    log_every: int = 1000,
    workers: int = 1,
    progress: bool = True,
):
    total = len(traces)
    if total == 0:
        return {
            "candidate": {
                "moves": 0,
                "acpl": None,
                "accuracy": None,
                "blunders": 0,
                "mistakes": 0,
                "inaccuracies": 0,
            },
            "baseline": {
                "moves": 0,
                "acpl": None,
                "accuracy": None,
                "blunders": 0,
                "mistakes": 0,
                "inaccuracies": 0,
            },
        }
    workers = max(1, min(int(workers), max(1, total)))
    log_every = max(0, int(log_every))
    progress_print(
        progress,
        f"arena quality: analysing {total} moves with Stockfish "
        f"workers={workers}",
    )

    splits = [total // workers] * workers
    for index in range(total % workers):
        splits[index] += 1

    jobs = []
    offset = 0
    for worker_index, count in enumerate(splits, 1):
        if count <= 0:
            continue
        subset = traces[offset:offset + count]
        offset += count
        jobs.append((
            worker_index,
            len(splits),
            subset,
            teacher_config,
            loss_cap_cp,
            log_every,
            progress,
        ))

    if len(jobs) == 1:
        outputs = [_quality_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            outputs = pool.map(_quality_worker, jobs)

    values = {
        "candidate": {"regret": [], "accuracy": []},
        "baseline": {"regret": [], "accuracy": []},
    }
    for output in outputs:
        for owner in ("candidate", "baseline"):
            values[owner]["regret"].extend(output[owner]["regret"])
            values[owner]["accuracy"].extend(output[owner]["accuracy"])

    metrics = {}
    for owner in ("candidate", "baseline"):
        regrets = values[owner]["regret"]
        accuracies = values[owner]["accuracy"]
        metrics[owner] = {
            "moves": len(regrets),
            "acpl": float(np.mean(regrets)) if regrets else None,
            "accuracy": float(np.mean(accuracies)) if accuracies else None,
            "blunders": int(sum(regret >= 300 for regret in regrets)),
            "mistakes": int(sum(100 <= regret < 300 for regret in regrets)),
            "inaccuracies": int(sum(50 <= regret < 100 for regret in regrets)),
        }
    return metrics


def _quality_worker(job):
    (
        worker_index,
        worker_count,
        traces,
        teacher_config,
        loss_cap_cp,
        log_every,
        progress,
    ) = job

    config = replace(
        teacher_config,
        cache_path=worker_cache_path(
            teacher_config.cache_path,
            worker_index,
            worker_count,
        ),
    )
    values = {
        "candidate": {"regret": [], "accuracy": []},
        "baseline": {"regret": [], "accuracy": []},
    }
    total = len(traces)
    with StockfishTeacher(config) as teacher:
        for index, row in enumerate(traces, 1):
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["move"])
            if move not in board.legal_moves:
                continue
            result = teacher.analyse(board, played_move=move)
            regret = float(
                min(
                    max(1, int(loss_cap_cp)),
                    max(0, int(result.get("regret_cp", 0))),
                )
            )
            owner = row["owner"]
            values[owner]["regret"].append(regret)
            values[owner]["accuracy"].append(move_accuracy_from_regret(regret))
            if log_every > 0 and (index == total or index % log_every == 0):
                progress_print(
                    progress,
                    f"arena quality worker {worker_index}: "
                    f"{index}/{total} moves analysed",
                )
    return values


def evaluate_models(
    candidate_path,
    baseline_path,
    games=100,
    sims=80,
    workers=1,
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
    mcts_time_fraction=0.90,
    mate_plies=0,
    mate_topk=4,
    mate_nodes=20000,
    uci=STOCKFISH_PATH,
    uci_depth=8,
    uci_movetime_ms=0,
    uci_threads=4,
    uci_hash_mb=512,
    uci_multipv=4,
    teacher_cache="data/selflearn/teacher_cache.sqlite",
    quality_loss_cap_cp=1000,
    pgn_output=None,
    trace_output=None,
    pgn_comments=False,
    pgn_columns=88,
    claim_draws=False,
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
        f"workers={workers}",
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

    workers = max(1, int(workers))
    splits = [len(specs) // workers] * workers
    for index in range(len(specs) % workers):
        splits[index] += 1

    jobs = []
    offset = 0
    for count in splits:
        if count <= 0:
            continue
        game_id_offset = offset
        subset = specs[offset:offset + count]
        offset += count
        jobs.append((
            len(jobs) + 1,
            candidate_path,
            baseline_path,
            subset,
            game_id_offset,
            sims,
            device,
            max_plies,
            mcts_batch_size,
            movetime_ms,
            search_type,
            c_puct,
            c_puct_base,
            c_puct_factor,
            fpu_reduction,
            mcts_time_fraction,
            mate_plies,
            mate_topk,
            mate_nodes,
            pgn_comments,
            pgn_columns,
            claim_draws,
            trace_root_topn,
            progress,
        ))

    if len(jobs) == 1:
        outputs = [_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            outputs = pool.map(_worker, jobs)

    scores = []
    counts = {"win": 0, "draw": 0, "loss": 0}
    raw_results = {}
    plies_total = 0
    traces = []
    pgns = []
    for output in outputs:
        scores.extend(output["scores"])
        plies_total += int(output["plies_total"])
        traces.extend(output["traces"])
        pgns.extend(output.get("pgns", []))
        for key, value in output["counts"].items():
            counts[key] += int(value)
        for key, value in output["raw_results"].items():
            raw_results[key] = raw_results.get(key, 0) + int(value)

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

    teacher_config = TeacherConfig(
        uci=uci,
        depth=uci_depth,
        movetime_ms=uci_movetime_ms,
        multipv=uci_multipv,
        threads=uci_threads,
        hash_mb=uci_hash_mb,
        cache_path=teacher_cache,
    )
    quality = evaluate_move_quality(
        traces,
        teacher_config,
        loss_cap_cp=quality_loss_cap_cp,
        log_every=log_every,
        workers=workers,
        progress=progress,
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
    effective_mate_plies = int(mate_plies) if str(search_type) == "mcts-mate" else 0
    effective_mate_topk = int(mate_topk) if str(search_type) == "mcts-mate" else 0
    effective_mate_nodes = int(mate_nodes) if str(search_type) == "mcts-mate" else 0
    return {
        "candidate": candidate_path,
        "candidate_sha256": candidate_hash,
        "baseline": baseline_path,
        "baseline_sha256": baseline_hash,
        **game_summary,
        "quality": quality,
        "quality_loss_cap_cp": int(quality_loss_cap_cp),
        "search_type": str(search_type),
        "sims_soft_cap": int(effective_sims),
        "mcts_batch_size": int(mcts_batch_size),
        "movetime_ms": int(movetime_ms),
        "mate_plies": int(effective_mate_plies),
        "mate_topk": int(effective_mate_topk),
        "mate_nodes": int(effective_mate_nodes),
        "c_puct_initial": float(c_puct),
        "c_puct_base": float(c_puct_base),
        "c_puct_factor": float(c_puct_factor),
        "fpu_reduction": float(fpu_reduction),
        "mcts_time_fraction": float(mcts_time_fraction),
        "claim_draws": bool(claim_draws),
        "opening_book": opening_book,
        "book_plies": int(book_plies),
        "paired_openings": True,
        "unique_start_positions": len({fen for fen, _ in specs}),
        "trace_output": trace_output,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models by paired games and Stockfish move quality"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
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
    parser.add_argument("--mcts-time-fraction", type=float, default=0.90)
    parser.add_argument("--mate-plies", type=int, default=0)
    parser.add_argument("--mate-topk", type=int, default=4)
    parser.add_argument("--mate-nodes", type=int, default=20000)

    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--uci-depth", type=int, default=8)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--uci-multipv", type=int, default=4)
    parser.add_argument("--teacher-cache", default="data/selflearn/teacher_cache.sqlite")
    parser.add_argument("--quality-loss-cap-cp", type=int, default=1000)
    parser.add_argument("--pgn-output", default=None)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--pgn-comments", action="store_true", default=False)
    parser.add_argument("--pgn-columns", type=int, default=88)
    parser.add_argument("--claim-draws", action="store_true", default=False)
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
        workers=args.workers,
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
        mcts_time_fraction=args.mcts_time_fraction,
        mate_plies=args.mate_plies,
        mate_topk=args.mate_topk,
        mate_nodes=args.mate_nodes,
        uci=args.uci,
        uci_depth=args.uci_depth,
        uci_movetime_ms=args.uci_movetime_ms,
        uci_threads=args.uci_threads,
        uci_hash_mb=args.uci_hash_mb,
        uci_multipv=args.uci_multipv,
        teacher_cache=args.teacher_cache,
        quality_loss_cap_cp=args.quality_loss_cap_cp,
        pgn_output=args.pgn_output,
        trace_output=args.trace_output,
        pgn_comments=args.pgn_comments,
        pgn_columns=args.pgn_columns,
        claim_draws=args.claim_draws,
        trace_root_topn=args.trace_root_topn,
        log_every=args.log_every,
        progress=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
