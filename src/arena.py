"""Paired model comparison with Stockfish move-quality validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import multiprocessing as mp
import os
from typing import Dict, List

import chess
import numpy as np

from config import CONFIDENCE_Z, DEVICE, STOCKFISH_PATH
from model import load_model
from opening_book import make_arena_specs
from search import SearchOptions, UnifiedSearch
from teacher import (
    StockfishTeacher,
    TeacherConfig,
    move_accuracy_from_regret,
)


def progress_print(enabled: bool, *parts):
    if enabled:
        print(*parts, flush=True)


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
):
    board = chess.Board(start_fen)
    trace = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < int(max_plies):
        candidate_turn = board.turn == candidate_color
        searcher = candidate_searcher if candidate_turn else baseline_searcher
        result = searcher.search(board.copy(stack=False))
        move = result.move
        if move not in board.legal_moves:
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[0]

        trace.append({
            "fen": board.fen(),
            "move": move.uci(),
            "owner": "candidate" if candidate_turn else "baseline",
        })
        board.push(move)
        ply += 1

    result_string = board.result(claim_draw=True)
    if result_string == "*":
        result_string = "1/2-1/2"
    return (
        result_string,
        score_from_result(result_string, candidate_color),
        ply,
        trace,
    )


def _worker(job):
    (
        worker_index,
        candidate_path,
        baseline_path,
        specs,
        sims,
        device,
        max_plies,
        mcts_batch_size,
        movetime_ms,
        c_puct,
        alpha_beta_depth,
        alpha_beta_topk,
        alpha_beta_nodes,
        alpha_beta_quiescence,
        alpha_beta_margin,
        alpha_beta_time_fraction,
        mate_guard_plies,
        q_tiebreak,
        q_tiebreak_min_visits,
        q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio,
        q_tiebreak_margin,
        progress,
    ) = job

    progress_print(
        progress,
        f"arena worker {worker_index}: loading models on {device}",
    )
    candidate = load_model(candidate_path, device=device)
    baseline = load_model(baseline_path, device=device)
    options = SearchOptions(
        mcts_sims=sims,
        mcts_batch_size=mcts_batch_size,
        time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        c_puct=c_puct,
        alpha_beta_depth=alpha_beta_depth,
        alpha_beta_topk=alpha_beta_topk,
        alpha_beta_nodes=alpha_beta_nodes,
        alpha_beta_quiescence=alpha_beta_quiescence,
        alpha_beta_margin=alpha_beta_margin,
        alpha_beta_time_fraction=alpha_beta_time_fraction,
        mate_guard_plies=mate_guard_plies,
        q_tiebreak=q_tiebreak,
        q_tiebreak_min_visits=q_tiebreak_min_visits,
        q_tiebreak_p_ratio=q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio=q_tiebreak_visit_ratio,
        q_tiebreak_margin=q_tiebreak_margin,
    )
    candidate_searcher = UnifiedSearch(candidate, options, device=device)
    baseline_searcher = UnifiedSearch(baseline, options, device=device)

    scores = []
    counts = {"win": 0, "draw": 0, "loss": 0}
    raw_results = {}
    plies_total = 0
    traces = []

    total_specs = len(specs)
    for game_index, (fen, candidate_color) in enumerate(specs, 1):
        result, score, plies, trace = play_arena_game(
            candidate_searcher,
            baseline_searcher,
            candidate_color,
            max_plies,
            fen,
        )
        scores.append(score)
        counts[outcome_from_score(score)] += 1
        raw_results[result] = raw_results.get(result, 0) + 1
        plies_total += plies
        traces.extend(trace)
        progress_print(
            progress,
            f"arena worker {worker_index}: game {game_index}/{total_specs} "
            f"candidate_color={'white' if candidate_color == chess.WHITE else 'black'} "
            f"result={result} candidate_score={score:.1f} plies={plies}",
        )

    return {
        "scores": scores,
        "counts": counts,
        "raw_results": raw_results,
        "plies_total": plies_total,
        "traces": traces,
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
    c_puct=1.5,
    alpha_beta_depth=4,
    alpha_beta_topk=4,
    alpha_beta_nodes=20000,
    alpha_beta_quiescence=3,
    alpha_beta_margin=0.10,
    alpha_beta_time_fraction=0.25,
    mate_guard_plies=3,
    q_tiebreak=True,
    q_tiebreak_min_visits=32,
    q_tiebreak_p_ratio=0.90,
    q_tiebreak_visit_ratio=0.80,
    q_tiebreak_margin=0.25,
    uci=STOCKFISH_PATH,
    uci_depth=8,
    uci_movetime_ms=0,
    uci_threads=4,
    uci_hash_mb=512,
    uci_multipv=4,
    teacher_cache="data/selflearn/teacher_cache.sqlite",
    quality_loss_cap_cp=1000,
    log_every=1000,
    progress=True,
):
    progress_print(
        progress,
        "arena: start",
        f"candidate={candidate_path}",
        f"baseline={baseline_path}",
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
        subset = specs[offset:offset + count]
        offset += count
        jobs.append((
            len(jobs) + 1,
            candidate_path,
            baseline_path,
            subset,
            sims,
            device,
            max_plies,
            mcts_batch_size,
            movetime_ms,
            c_puct,
            alpha_beta_depth,
            alpha_beta_topk,
            alpha_beta_nodes,
            alpha_beta_quiescence,
            alpha_beta_margin,
            alpha_beta_time_fraction,
            mate_guard_plies,
            q_tiebreak,
            q_tiebreak_min_visits,
            q_tiebreak_p_ratio,
            q_tiebreak_visit_ratio,
            q_tiebreak_margin,
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
    for output in outputs:
        scores.extend(output["scores"])
        plies_total += int(output["plies_total"])
        traces.extend(output["traces"])
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
    return {
        "candidate": candidate_path,
        "baseline": baseline_path,
        **game_summary,
        "quality": quality,
        "quality_loss_cap_cp": int(quality_loss_cap_cp),
        "search_type": "uncertainty_mcts_alpha_beta",
        "sims_soft_cap": int(sims),
        "mcts_batch_size": int(mcts_batch_size),
        "movetime_ms": int(movetime_ms),
        "mate_guard_plies": int(mate_guard_plies),
        "q_tiebreak": bool(q_tiebreak),
        "q_tiebreak_min_visits": int(q_tiebreak_min_visits),
        "q_tiebreak_p_ratio": float(q_tiebreak_p_ratio),
        "q_tiebreak_visit_ratio": float(q_tiebreak_visit_ratio),
        "q_tiebreak_margin": float(q_tiebreak_margin),
        "opening_book": opening_book,
        "book_plies": int(book_plies),
        "paired_openings": True,
        "unique_start_positions": len({fen for fen, _ in specs}),
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
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--alpha-beta-depth", type=int, default=4)
    parser.add_argument("--alpha-beta-topk", type=int, default=4)
    parser.add_argument("--alpha-beta-nodes", type=int, default=20000)
    parser.add_argument("--alpha-beta-quiescence", type=int, default=3)
    parser.add_argument("--alpha-beta-margin", type=float, default=0.10)
    parser.add_argument("--alpha-beta-time-fraction", type=float, default=0.25)
    parser.add_argument("--mate-guard-plies", type=int, default=3)
    parser.add_argument("--q-tiebreak", action="store_true", default=True)
    parser.add_argument("--no-q-tiebreak", dest="q_tiebreak", action="store_false")
    parser.add_argument("--q-tiebreak-min-visits", type=int, default=32)
    parser.add_argument("--q-tiebreak-p-ratio", type=float, default=0.90)
    parser.add_argument("--q-tiebreak-visit-ratio", type=float, default=0.80)
    parser.add_argument("--q-tiebreak-margin", type=float, default=0.25)

    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--uci-depth", type=int, default=8)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--uci-multipv", type=int, default=4)
    parser.add_argument("--teacher-cache", default="data/selflearn/teacher_cache.sqlite")
    parser.add_argument("--quality-loss-cap-cp", type=int, default=1000)
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
        c_puct=args.c_puct,
        alpha_beta_depth=args.alpha_beta_depth,
        alpha_beta_topk=args.alpha_beta_topk,
        alpha_beta_nodes=args.alpha_beta_nodes,
        alpha_beta_quiescence=args.alpha_beta_quiescence,
        alpha_beta_margin=args.alpha_beta_margin,
        alpha_beta_time_fraction=args.alpha_beta_time_fraction,
        mate_guard_plies=args.mate_guard_plies,
        q_tiebreak=args.q_tiebreak,
        q_tiebreak_min_visits=args.q_tiebreak_min_visits,
        q_tiebreak_p_ratio=args.q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio=args.q_tiebreak_visit_ratio,
        q_tiebreak_margin=args.q_tiebreak_margin,
        uci=args.uci,
        uci_depth=args.uci_depth,
        uci_movetime_ms=args.uci_movetime_ms,
        uci_threads=args.uci_threads,
        uci_hash_mb=args.uci_hash_mb,
        uci_multipv=args.uci_multipv,
        teacher_cache=args.teacher_cache,
        quality_loss_cap_cp=args.quality_loss_cap_cp,
        log_every=args.log_every,
        progress=True,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
