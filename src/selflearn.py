"""Teacher-constrained self-learning with dynamic regression protection."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import time
import uuid
from typing import Dict, List

import chess
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from acceptance import attach_arena_acceptance
from arena import evaluate_models
from checkpoint_io import atomic_copy_with_backup
from chess_env import board_to_packed
from config import (
    DEVICE,
    H5_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUM_ACTIONS,
    NUM_WORKERS,
    REGRESSION_PATH,
    SELFLEARN_DIR,
    STOCKFISH_PATH,
    WEIGHT_DECAY,
)
from data import H5ChessDataset, MultiSelfLearnDataset
from model import load_model, save_model
from move_encoder import move_to_index
from opening_book import OpeningBook, unique_position_fens
from regression import load_cases, merge_cases
from search import SearchOptions, UnifiedSearch
from teacher import (
    StockfishTeacher,
    TeacherConfig,
    acceptable_moves,
    teacher_weight_from_result,
)


DEFAULT_TEACHER_CACHE = os.path.join(SELFLEARN_DIR, "teacher_cache.sqlite")
DEFAULT_DATA_RUNS_DIR = os.path.join("data", "runs")
DEFAULT_MODEL_RUNS_DIR = os.path.join(MODEL_DIR, "runs")


def ensure_parent(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def normalize_policy(policy):
    policy = np.asarray(policy, dtype=np.float32)
    total = float(policy.sum())
    return policy / total if total > 0 else policy


def one_hot_move_policy(board, move):
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if move in board.legal_moves:
        policy[move_to_index(move)] = 1.0
    return policy


def root_topk_moves(board, search_result, selected_move, topk):
    limit = max(1, int(topk))
    legal_moves = set(board.legal_moves)
    moves = []
    seen = set()

    def add(move):
        if move not in legal_moves:
            return
        uci = move.uci()
        if uci in seen:
            return
        seen.add(uci)
        moves.append(move)

    add(selected_move)
    for row in search_result.info.get("root", []):
        if len(moves) >= limit:
            break
        try:
            add(chess.Move.from_uci(str(row.get("move"))))
        except Exception:
            continue

    if len(moves) < limit:
        policy = np.asarray(search_result.policy, dtype=np.float32)
        ordered = sorted(
            board.legal_moves,
            key=lambda move: (float(policy[move_to_index(move)]), move.uci()),
            reverse=True,
        )
        for move in ordered:
            if len(moves) >= limit:
                break
            add(move)

    return moves[:limit]


def board_ply_number(board: chess.Board) -> int:
    return max(
        0,
        (int(board.fullmove_number) - 1) * 2
        + (0 if board.turn == chess.WHITE else 1),
    )


def worker_cache_path(cache_path, worker_index, worker_count):
    if not cache_path or int(worker_count) <= 1:
        return cache_path
    root, ext = os.path.splitext(cache_path)
    return f"{root}.worker{int(worker_index)}{ext or '.sqlite'}"


def create_run_id():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"selflearn_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def normalize_run_id(run_id):
    value = str(run_id or "").strip()
    if not value:
        raise ValueError("empty selflearn run id")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"run id must be a name, not a path: {value}")
    return value


def make_run_dirs(data_root, model_root, run_id):
    data_run_dir = os.path.abspath(os.path.join(data_root, run_id))
    model_run_dir = os.path.abspath(os.path.join(model_root, run_id))
    allowed_preexisting_data_files = {"info.log", "pid"}
    data_dir_exists = os.path.exists(data_run_dir)
    if data_dir_exists:
        if not os.path.isdir(data_run_dir):
            raise FileExistsError(f"selflearn data run path is not a directory: {data_run_dir}")
        unexpected_files = set(os.listdir(data_run_dir)) - allowed_preexisting_data_files
        if unexpected_files:
            raise FileExistsError(
                "selflearn data run directory already contains run data: "
                f"{data_run_dir}"
            )
    if os.path.exists(model_run_dir):
        raise FileExistsError(
            "selflearn run already exists: "
            f"data={data_run_dir} model={model_run_dir}"
        )
    if not data_dir_exists:
        os.makedirs(data_run_dir, exist_ok=False)
    try:
        os.makedirs(model_run_dir, exist_ok=False)
    except Exception:
        try:
            os.rmdir(data_run_dir)
        except OSError:
            pass
        raise
    return data_run_dir, model_run_dir


def prepare_run_paths(args):
    if args.run_id:
        run_id = normalize_run_id(args.run_id)
    elif args.run_dir:
        run_id = normalize_run_id(os.path.basename(os.path.normpath(args.run_dir)))
    else:
        run_id = create_run_id()

    run_data_dir, run_model_dir = make_run_dirs(
        args.data_runs_dir,
        args.model_runs_dir,
        run_id,
    )

    if args.teacher_cache == DEFAULT_TEACHER_CACHE:
        args.teacher_cache = os.path.join(run_data_dir, "teacher_cache.sqlite")

    if args.regression_data == REGRESSION_PATH:
        run_regression = os.path.join(run_data_dir, "regression.json")
        if os.path.exists(REGRESSION_PATH):
            shutil.copy2(REGRESSION_PATH, run_regression)
        args.regression_data = run_regression

    return {
        "run_id": run_id,
        "data_run_dir": run_data_dir,
        "model_run_dir": run_model_dir,
        "data_dir": run_data_dir,
        "model_dir": run_model_dir,
        "current_model": os.path.join(run_model_dir, "current.pth"),
    }


def adjudicate_truncated_board(board, teacher, threshold_cp):
    threshold = max(0, int(threshold_cp))
    try:
        teacher_result = teacher.analyse(board)
        side_to_move_cp = int(teacher_result.get("best_score_cp", 0))
    except Exception:
        return "1/2-1/2", 0

    white_cp = side_to_move_cp if board.turn == chess.WHITE else -side_to_move_cp
    if white_cp > threshold:
        return "1-0", 1
    if white_cp < -threshold:
        return "0-1", -1
    return "1/2-1/2", 0


def load_selflearn_openings(args, iteration):
    value = str(args.opening_book or "").strip()
    if not value or value.lower() in {"none", "off"}:
        return [chess.Board().fen()]

    required_positions = max(1, int(args.games_per_iter))
    book = OpeningBook(
        path=value,
        book_plies=args.book_plies,
        max_positions=max(int(args.max_book_positions), required_positions),
        seed=args.seed + 50000 + iteration,
    )
    return list(book.positions)


def unique_fens(fens):
    return unique_position_fens(list(fens))


def select_iteration_openings(args, iteration, opening_positions):
    needed = max(1, int(args.games_per_iter))
    unique = unique_fens(opening_positions)
    if len(unique) < needed:
        raise ValueError(
            "deterministic self-learning requires one unique opening state per game: "
            f"games_per_iter={needed}, unique_openings={len(unique)}. "
            "Generate a larger opening book or reduce --games-per-iter."
        )
    rng = random.Random(args.seed + 70000 + iteration)
    selected = list(unique)
    rng.shuffle(selected)
    return selected[:needed]


def search_arg(args, name, evaluation=False):
    if evaluation:
        value = getattr(args, f"eval_{name}", None)
        if value is not None:
            return value
    return getattr(args, name)


def write_selflearn_h5(path, rows, attrs):
    ensure_parent(path)
    if not rows:
        raise ValueError("no self-learning positions generated")

    regret_values = np.asarray(
        [max(0, int(row["regret_cp"])) for row in rows],
        dtype=np.int32,
    )
    attrs = dict(attrs)
    attrs["stored_regret_dtype"] = "int32"
    attrs["max_regret_cp"] = int(regret_values.max()) if regret_values.size else 0

    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "states",
            data=np.asarray([row["state"] for row in rows], dtype=np.uint8),
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )
        for key in ("target_policy", "mcts_policy", "teacher_policy"):
            h5.create_dataset(
                key,
                data=np.asarray([row[key] for row in rows], dtype=np.float16),
                dtype="float16",
                chunks=True,
                compression="lzf",
            )
        h5.create_dataset(
            "terminal_values",
            data=np.asarray([row["terminal_value"] for row in rows], dtype=np.int8),
            dtype="int8",
            chunks=True,
            compression="lzf",
        )
        h5.create_dataset(
            "terminal_valid",
            data=np.asarray([row["terminal_valid"] for row in rows], dtype=np.uint8),
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )
        h5.create_dataset(
            "teacher_values",
            data=np.asarray([row["teacher_value"] for row in rows], dtype=np.float16),
            dtype="float16",
            chunks=True,
            compression="lzf",
        )
        h5.create_dataset(
            "teacher_weights",
            data=np.asarray([row["teacher_weight"] for row in rows], dtype=np.float16),
            dtype="float16",
            chunks=True,
            compression="lzf",
        )
        h5.create_dataset(
            "regret_cp",
            data=regret_values,
            dtype="int32",
            chunks=True,
            compression="lzf",
        )
        for key, value in attrs.items():
            h5.attrs[key] = value


def search_options_from_args(args, evaluation=False):
    sims = args.eval_sims if evaluation else args.sims
    batch = args.eval_mcts_batch_size if evaluation else args.mcts_batch_size
    movetime_ms = args.eval_movetime_ms if evaluation else args.movetime_ms
    return SearchOptions(
        mcts_sims=sims,
        mcts_batch_size=batch,
        time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        c_puct=search_arg(args, "c_puct", evaluation),
        mate_guard_plies=search_arg(args, "mate_guard_plies", evaluation),
        mate_guard_topk=search_arg(args, "mate_guard_topk", evaluation),
        mate_guard_nodes=search_arg(args, "mate_guard_nodes", evaluation),
        mate_guard_time_fraction=search_arg(
            args,
            "mate_guard_time_fraction",
            evaluation,
        ),
        q_tiebreak=search_arg(args, "q_tiebreak", evaluation),
        q_tiebreak_min_visits=search_arg(args, "q_tiebreak_min_visits", evaluation),
        q_tiebreak_p_ratio=search_arg(args, "q_tiebreak_p_ratio", evaluation),
        q_tiebreak_visit_ratio=search_arg(args, "q_tiebreak_visit_ratio", evaluation),
        q_tiebreak_margin=search_arg(args, "q_tiebreak_margin", evaluation),
    )


def run_selflearn_worker(job):
    (
        worker_index,
        worker_count,
        args,
        model_path,
        iteration,
        start_index,
        game_count,
        assigned_openings,
    ) = job

    seed = args.seed + iteration * 10000 + worker_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_path, device=args.device)
    searcher = UnifiedSearch(
        model,
        search_options_from_args(args, evaluation=False),
        device=args.device,
    )
    teacher_config = TeacherConfig(
        uci=args.uci,
        depth=args.uci_depth,
        movetime_ms=args.uci_movetime_ms,
        multipv=args.uci_multipv,
        threads=args.uci_threads,
        hash_mb=args.uci_hash_mb,
        policy_temperature_cp=args.teacher_policy_temperature_cp,
        cache_path=worker_cache_path(
            args.teacher_cache,
            worker_index,
            worker_count,
        ),
    )

    all_rows = []
    generated_cases = {}
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    teacher_positions = 0
    truncated_games = 0
    regret_sum = 0.0
    weight_sum = 0.0
    label_weight_sum = 0.0
    label_move_sum = 0
    teacher_vetoed_moves = 0
    used_openings = set()

    with StockfishTeacher(teacher_config) as teacher:
        for local_index in range(game_count):
            game_index = start_index + local_index
            start_fen = assigned_openings[local_index]
            used_openings.add(start_fen)
            board = chess.Board(start_fen)
            start_ply = board_ply_number(board)
            game_rows = []
            ply = 0
            game_vetoes = 0

            while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
                search_result = searcher.search(board.copy(stack=False))
                move = search_result.move
                if move not in board.legal_moves:
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    move = legal[0]

                model_move = move
                model_policy = one_hot_move_policy(board, model_move)
                mcts_policy = normalize_policy(search_result.mcts_policy)
                teacher_policy = normalize_policy(search_result.policy)
                teacher_value = 0.0
                teacher_weight = 0.0
                teacher_label_weight = 0.0
                regret_cp = 0

                use_teacher = (
                    ply >= args.teacher_start_ply
                    and (ply - args.teacher_start_ply)
                    % max(1, args.teacher_every)
                    == 0
                    and random.random() <= args.teacher_sample_rate
                )
                if use_teacher:
                    label_moves = root_topk_moves(
                        board,
                        search_result,
                        model_move,
                        args.teacher_label_topk,
                    )
                    teacher_result = teacher.analyse_candidates(
                        board,
                        label_moves,
                        played_move=model_move,
                    )
                    teacher_policy = teacher.dense_policy(board, teacher_result)
                    teacher_value = float(teacher_result.get("value", 0.0))
                    teacher_weight = teacher_weight_from_result(teacher_result)
                    teacher_label_weight = max(
                        float(teacher_weight),
                        float(args.teacher_label_min_weight),
                    )
                    teacher_label_weight = min(0.95, max(0.0, teacher_label_weight))
                    regret_cp = int(teacher_result.get("regret_cp", 0))
                    teacher_positions += 1
                    regret_sum += float(regret_cp)
                    weight_sum += float(teacher_weight)
                    label_weight_sum += float(teacher_label_weight)
                    label_move_sum += int(
                        teacher_result.get("teacher_label_topk", len(label_moves))
                    )

                    answers = acceptable_moves(
                        teacher_result,
                        tolerance_cp=args.regression_answer_tolerance_cp,
                        max_answers=args.regression_max_answers,
                    )
                    if (
                        answers
                        and regret_cp >= args.regression_min_regret_cp
                        and teacher_weight >= args.regression_min_teacher_weight
                    ):
                        fen = board.fen()
                        generated_cases[fen] = {
                            "fen": fen,
                            "answers": answers,
                            "best_score_cp": int(
                                teacher_result.get("best_score_cp", 0)
                            ),
                            "regret_cp": regret_cp,
                            "teacher_weight": teacher_weight,
                            "seen": 1,
                        }

                    if (
                        args.teacher_veto
                        and regret_cp >= args.teacher_veto_regret_cp
                        and teacher_weight >= args.teacher_veto_min_weight
                    ):
                        try:
                            teacher_move = chess.Move.from_uci(
                                str(teacher_result.get("best_move"))
                            )
                        except Exception:
                            teacher_move = None
                        if (
                            teacher_move is not None
                            and teacher_move in board.legal_moves
                            and teacher_move != move
                        ):
                            move = teacher_move
                            teacher_vetoed_moves += 1
                            game_vetoes += 1

                target_policy = normalize_policy(
                    (1.0 - teacher_label_weight) * model_policy
                    + teacher_label_weight * teacher_policy
                )
                game_rows.append({
                    "state": board_to_packed(board),
                    "turn": board.turn,
                    "target_policy": target_policy,
                    "mcts_policy": mcts_policy,
                    "teacher_policy": teacher_policy,
                    "teacher_value": teacher_value,
                    "teacher_weight": teacher_weight,
                    "regret_cp": regret_cp,
                })

                board.push(move)
                ply += 1

            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                truncated_games += 1
                result_string, white_value = adjudicate_truncated_board(
                    board,
                    teacher,
                    args.truncate_adjudication_cp,
                )
            elif outcome.winner is None:
                result_string = "1/2-1/2"
                white_value = 0
            else:
                result_string = "1-0" if outcome.winner == chess.WHITE else "0-1"
                white_value = 1 if outcome.winner == chess.WHITE else -1

            results[result_string] = results.get(result_string, 0) + 1
            for row in game_rows:
                terminal_value = (
                    white_value
                    if row["turn"] == chess.WHITE
                    else -white_value
                )
                row["terminal_value"] = terminal_value
                row["terminal_valid"] = 1
                row.pop("turn", None)
                all_rows.append(row)

            print(
                f"selflearn worker {worker_index}: "
                f"game {game_index + 1}/{args.games_per_iter}: "
                f"start_ply={start_ply}, plies={ply}, "
                f"vetoes={game_vetoes}, result={result_string}",
                flush=True,
            )

    return {
        "rows": all_rows,
        "cases": generated_cases,
        "results": results,
        "teacher_positions": teacher_positions,
        "truncated_games": truncated_games,
        "regret_sum": regret_sum,
        "weight_sum": weight_sum,
        "label_weight_sum": label_weight_sum,
        "label_move_sum": label_move_sum,
        "teacher_vetoed_moves": teacher_vetoed_moves,
        "used_openings": list(used_openings),
    }


def generate_selflearn_data(args, model_path, output_path, iteration):
    random.seed(args.seed + iteration)
    np.random.seed(args.seed + iteration)
    torch.manual_seed(args.seed + iteration)

    print(
        "selflearn data generation start:",
        f"iteration={iteration}",
        f"model={model_path}",
        f"games={args.games_per_iter}",
        f"sims={args.sims}",
        f"device={args.device}",
        f"workers={args.selfplay_workers}",
        flush=True,
    )

    available_opening_positions = load_selflearn_openings(args, iteration)
    opening_positions = select_iteration_openings(
        args,
        iteration,
        available_opening_positions,
    )
    print(
        "selflearn openings:",
        f"path={args.opening_book}",
        f"available={len(unique_fens(available_opening_positions))}",
        f"assigned={len(opening_positions)}",
        f"book_plies={args.book_plies}",
        flush=True,
    )

    worker_count = max(1, min(int(args.selfplay_workers), int(args.games_per_iter)))
    splits = [int(args.games_per_iter) // worker_count] * worker_count
    for index in range(int(args.games_per_iter) % worker_count):
        splits[index] += 1

    jobs = []
    offset = 0
    for worker_index, count in enumerate(splits, 1):
        if count <= 0:
            continue
        jobs.append((
            worker_index,
            worker_count,
            args,
            model_path,
            iteration,
            offset,
            count,
            opening_positions[offset:offset + count],
        ))
        offset += count

    if len(jobs) == 1:
        outputs = [run_selflearn_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            outputs = pool.map(run_selflearn_worker, jobs)

    all_rows = []
    generated_cases = {}
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    teacher_positions = 0
    truncated_games = 0
    regret_sum = 0.0
    weight_sum = 0.0
    label_weight_sum = 0.0
    label_move_sum = 0
    teacher_vetoed_moves = 0
    used_openings = set()

    for output in outputs:
        all_rows.extend(output["rows"])
        generated_cases.update(output["cases"])
        for key, value in output["results"].items():
            results[key] = results.get(key, 0) + int(value)
        teacher_positions += int(output["teacher_positions"])
        truncated_games += int(output["truncated_games"])
        regret_sum += float(output["regret_sum"])
        weight_sum += float(output["weight_sum"])
        label_weight_sum += float(output["label_weight_sum"])
        label_move_sum += int(output["label_move_sum"])
        teacher_vetoed_moves += int(output["teacher_vetoed_moves"])
        used_openings.update(output["used_openings"])

    cases = list(generated_cases.values())
    cases.sort(
        key=lambda case: (
            float(case["teacher_weight"]),
            int(case["regret_cp"]),
        ),
        reverse=True,
    )
    cases = cases[: max(1, int(args.regression_max_new_per_iter))]

    attrs = {
        "type": "teacher_constrained_selflearning",
        "source_model": model_path,
        "iteration": int(iteration),
        "games": int(args.games_per_iter),
        "positions": int(len(all_rows)),
        "teacher_positions": int(teacher_positions),
        "truncated_games": int(truncated_games),
        "mean_regret_cp": (
            float(regret_sum / teacher_positions) if teacher_positions else 0.0
        ),
        "mean_teacher_weight": (
            float(weight_sum / teacher_positions) if teacher_positions else 0.0
        ),
        "mean_teacher_label_weight": (
            float(label_weight_sum / teacher_positions) if teacher_positions else 0.0
        ),
        "mean_teacher_label_moves": (
            float(label_move_sum / teacher_positions) if teacher_positions else 0.0
        ),
        "teacher_vetoed_moves": int(teacher_vetoed_moves),
        "teacher_veto": bool(args.teacher_veto),
        "teacher_veto_regret_cp": int(args.teacher_veto_regret_cp),
        "teacher_veto_min_weight": float(args.teacher_veto_min_weight),
        "teacher_label_topk": int(args.teacher_label_topk),
        "teacher_label_min_weight": float(args.teacher_label_min_weight),
        "truncate_adjudication_cp": int(args.truncate_adjudication_cp),
        "search_type": "mcts_mate_guard",
        "move_selection": "top1",
        "target_policy_base": "top1_teacher_labeled_topk",
        "sims_soft_cap": int(args.sims),
        "mate_guard_plies": int(args.mate_guard_plies),
        "mate_guard_topk": int(args.mate_guard_topk),
        "mate_guard_nodes": int(args.mate_guard_nodes),
        "mate_guard_time_fraction": float(args.mate_guard_time_fraction),
        "q_tiebreak": bool(args.q_tiebreak),
        "q_tiebreak_min_visits": int(args.q_tiebreak_min_visits),
        "q_tiebreak_p_ratio": float(args.q_tiebreak_p_ratio),
        "q_tiebreak_visit_ratio": float(args.q_tiebreak_visit_ratio),
        "q_tiebreak_margin": float(args.q_tiebreak_margin),
        "uci_depth": int(args.uci_depth),
        "uci_multipv": int(args.uci_multipv),
        "opening_book": str(args.opening_book),
        "book_plies": int(args.book_plies),
        "available_opening_positions": int(len(unique_fens(available_opening_positions))),
        "assigned_opening_positions": int(len(opening_positions)),
        "used_opening_positions": int(len(used_openings)),
    }
    write_selflearn_h5(output_path, all_rows, attrs)
    summary = {
        "iteration": int(iteration),
        "path": output_path,
        "games": int(args.games_per_iter),
        "positions": len(all_rows),
        "teacher_positions": teacher_positions,
        "truncated_games": truncated_games,
        "results": results,
        "regression_cases": len(cases),
        "mean_regret_cp": attrs["mean_regret_cp"],
        "mean_teacher_weight": attrs["mean_teacher_weight"],
        "mean_teacher_label_weight": attrs["mean_teacher_label_weight"],
        "mean_teacher_label_moves": attrs["mean_teacher_label_moves"],
        "teacher_vetoed_moves": teacher_vetoed_moves,
        "move_selection": attrs["move_selection"],
        "target_policy_base": attrs["target_policy_base"],
        "available_opening_positions": attrs["available_opening_positions"],
        "assigned_opening_positions": attrs["assigned_opening_positions"],
        "used_opening_positions": len(used_openings),
    }
    print("selflearn games summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return {
        "path": output_path,
        "positions": len(all_rows),
        "teacher_positions": teacher_positions,
        "truncated_games": truncated_games,
        "results": results,
        "teacher_vetoed_moves": teacher_vetoed_moves,
        "regression_cases": cases,
    }


def soft_policy_loss(logits, target):
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def kl_to_reference(student_logits, reference_logits, temperature):
    temperature = max(1e-6, float(temperature))
    student_logp = F.log_softmax(student_logits / temperature, dim=1)
    reference_p = F.softmax(reference_logits / temperature, dim=1)
    return (
        F.kl_div(student_logp, reference_p, reduction="batchmean")
        * temperature ** 2
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train_candidate(args, champion_path, replay_paths, candidate_path, iteration):
    pin_memory = str(args.device).startswith("cuda")
    dataset = MultiSelfLearnDataset(replay_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=pin_memory,
        persistent_workers=args.train_workers > 0,
    )

    supervised_iter = None
    if (
        args.supervised_data
        and os.path.exists(args.supervised_data)
        and args.supervised_weight > 0
    ):
        supervised_loader = DataLoader(
            H5ChessDataset(args.supervised_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.train_workers,
            pin_memory=pin_memory,
            persistent_workers=args.train_workers > 0,
        )
        supervised_iter = cycle(supervised_loader)

    print(
        "selflearn train start:",
        f"iteration={iteration}",
        f"champion={champion_path}",
        f"candidate={candidate_path}",
        f"replay_files={len(replay_paths)}",
        f"positions={len(dataset)}",
        f"device={args.device}",
        f"max_steps={args.train_max_steps}",
        flush=True,
    )
    student = load_model(champion_path, device=args.device)
    student.train()
    reference = load_model(champion_path, device=args.device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0
    stop = False

    for _epoch in range(args.epochs_per_iter):
        for batch in loader:
            (
                states,
                target_policy,
                terminal_values,
                terminal_valid,
                teacher_values,
                teacher_weights,
                _regret,
            ) = batch
            states = states.to(args.device, non_blocking=True)
            target_policy = target_policy.to(args.device, non_blocking=True)
            terminal_values = terminal_values.to(args.device, non_blocking=True)
            terminal_valid = terminal_valid.to(args.device, non_blocking=True)
            teacher_values = teacher_values.to(args.device, non_blocking=True)
            teacher_weights = teacher_weights.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                policy_logits, values = student(states)
                values = values.squeeze(1)
                with torch.no_grad():
                    reference_logits, _ = reference(states)

                policy_loss = soft_policy_loss(policy_logits, target_policy)

                terminal_per_item = (values - terminal_values).pow(2)
                terminal_value_loss = (
                    terminal_per_item * terminal_valid
                ).sum() / terminal_valid.sum().clamp_min(1.0)

                teacher_value_per_item = F.smooth_l1_loss(
                    values,
                    teacher_values,
                    reduction="none",
                )
                teacher_value_loss = (
                    teacher_value_per_item * teacher_weights
                ).sum() / teacher_weights.sum().clamp_min(1.0)

                kl_loss = kl_to_reference(
                    policy_logits,
                    reference_logits,
                    args.kl_temperature,
                )

                supervised_loss = torch.zeros((), device=args.device)
                if supervised_iter is not None:
                    sup_state, sup_move, sup_value = next(supervised_iter)
                    sup_state = sup_state.to(args.device, non_blocking=True)
                    sup_move = sup_move.to(args.device, non_blocking=True)
                    sup_value = sup_value.to(args.device, non_blocking=True)
                    sup_logits, sup_pred_value = student(sup_state)
                    supervised_loss = (
                        F.cross_entropy(sup_logits, sup_move)
                        + args.supervised_value_weight
                        * F.mse_loss(
                            sup_pred_value.squeeze(1),
                            sup_value,
                        )
                    )

                loss = (
                    args.policy_weight * policy_loss
                    + args.terminal_value_weight * terminal_value_loss
                    + args.teacher_value_weight * teacher_value_loss
                    + args.kl_weight * kl_loss
                    + args.supervised_weight * supervised_loss
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if (
                args.log_every > 0
                and (
                    global_step == 1
                    or global_step % args.log_every == 0
                )
            ):
                print(
                    "selflearn train step:",
                    f"iteration={iteration}",
                    f"step={global_step}",
                    f"policy={policy_loss.item():.4f}",
                    f"terminal_value={terminal_value_loss.item():.4f}",
                    f"teacher_value={teacher_value_loss.item():.4f}",
                    f"kl={kl_loss.item():.4f}",
                    f"supervised={supervised_loss.item():.4f}",
                    f"loss={loss.item():.4f}",
                    flush=True,
                )
            if args.train_max_steps and global_step >= args.train_max_steps:
                stop = True
                break
        if stop:
            break

    save_model(
        candidate_path,
        student,
        epoch=max(0, args.epochs_per_iter - 1),
        global_step=global_step,
        extra={
            "type": "teacher_constrained_selflearning",
            "source_model": champion_path,
            "iteration": int(iteration),
            "replay_files": list(replay_paths),
        },
    )
    print(
        "selflearn candidate saved:",
        f"path={candidate_path}",
        f"steps={global_step}",
        flush=True,
    )
    return {"steps": global_step, "candidate": candidate_path}


def validation_subset(dataset, samples):
    count = min(len(dataset), max(0, int(samples)))
    if count <= 0:
        return Subset(dataset, [])
    if count == len(dataset):
        return dataset
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=count,
        dtype=np.int64,
    ).tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def supervised_validation(model, path, device, samples, batch_size):
    if not path or not os.path.exists(path) or samples <= 0:
        return None
    dataset = validation_subset(H5ChessDataset(path), samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    total_loss = 0.0
    total_items = 0
    for state, move, value_target in loader:
        state = state.to(device)
        move = move.to(device)
        value_target = value_target.to(device)
        logits, value = model(state)
        per_item = F.cross_entropy(
            logits,
            move,
            reduction="none",
        ) + 0.25 * (
            value.squeeze(1) - value_target
        ).pow(2)
        total_loss += float(per_item.sum().item())
        total_items += int(state.shape[0])
    return total_loss / max(1, total_items)


@torch.no_grad()
def target_validation(model, replay_paths, device, samples, batch_size):
    dataset = validation_subset(
        MultiSelfLearnDataset(replay_paths),
        samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    total_ce = 0.0
    total_items = 0
    for batch in loader:
        state, target_policy = batch[0], batch[1]
        state = state.to(device)
        target_policy = target_policy.to(device)
        logits, _ = model(state)
        per_item = -(
            target_policy * F.log_softmax(logits, dim=1)
        ).sum(dim=1)
        total_ce += float(per_item.sum().item())
        total_items += int(state.shape[0])
    return total_ce / max(1, total_items)


def regression_validation(model, cases, args, label="model"):
    if not cases:
        return {
            "cases": 0,
            "correct": 0,
            "accuracy": None,
            "details": [],
        }

    options = SearchOptions(
        mcts_sims=args.regression_sims,
        mcts_batch_size=args.eval_mcts_batch_size,
        time_limit=(
            args.regression_movetime_ms / 1000.0
            if args.regression_movetime_ms > 0
            else None
        ),
        c_puct=search_arg(args, "c_puct", evaluation=True),
        mate_guard_plies=search_arg(args, "mate_guard_plies", evaluation=True),
        mate_guard_topk=search_arg(args, "mate_guard_topk", evaluation=True),
        mate_guard_nodes=search_arg(args, "mate_guard_nodes", evaluation=True),
        mate_guard_time_fraction=search_arg(
            args,
            "mate_guard_time_fraction",
            evaluation=True,
        ),
        q_tiebreak=search_arg(args, "q_tiebreak", evaluation=True),
        q_tiebreak_min_visits=search_arg(
            args,
            "q_tiebreak_min_visits",
            evaluation=True,
        ),
        q_tiebreak_p_ratio=search_arg(args, "q_tiebreak_p_ratio", evaluation=True),
        q_tiebreak_visit_ratio=search_arg(
            args,
            "q_tiebreak_visit_ratio",
            evaluation=True,
        ),
        q_tiebreak_margin=search_arg(args, "q_tiebreak_margin", evaluation=True),
    )
    searcher = UnifiedSearch(model, options, device=args.device)
    correct = 0
    details = []
    total_cases = len(cases)
    log_every = max(0, int(args.log_every))
    print(
        "regression validation start:",
        f"label={label}",
        f"cases={total_cases}",
        flush=True,
    )

    for index, case in enumerate(cases, 1):
        board = chess.Board(case["fen"])
        legal_answers = {
            move
            for move in case["answers"]
            if chess.Move.from_uci(move) in board.legal_moves
        }
        if not legal_answers:
            continue
        result = searcher.search(board)
        selected = result.move.uci()
        passed = selected in legal_answers
        correct += int(passed)
        details.append({
            "fen": case["fen"],
            "answers": sorted(legal_answers),
            "selected": selected,
            "correct": passed,
        })
        if log_every > 0 and (index == total_cases or index % log_every == 0):
            print(
                "regression validation:",
                f"label={label}",
                f"{index}/{total_cases}",
                f"correct={correct}",
                flush=True,
            )

    count = len(details)
    return {
        "cases": count,
        "correct": correct,
        "accuracy": correct / count if count else None,
        "details": details,
    }


def validate_candidate(
    args,
    champion_path,
    candidate_path,
    replay_paths,
    iteration,
    current_cases,
):
    print(
        "candidate validation start:",
        f"iteration={iteration}",
        f"champion={champion_path}",
        f"candidate={candidate_path}",
        flush=True,
    )
    champion = load_model(champion_path, device=args.device)
    candidate = load_model(candidate_path, device=args.device)

    print("validation stage: supervised champion", flush=True)
    champion_supervised = supervised_validation(
        champion,
        args.supervised_data,
        args.device,
        args.validation_samples,
        args.validation_batch_size,
    )
    print(
        "validation result:",
        f"champion_supervised_loss={champion_supervised}",
        flush=True,
    )
    print("validation stage: supervised candidate", flush=True)
    candidate_supervised = supervised_validation(
        candidate,
        args.supervised_data,
        args.device,
        args.validation_samples,
        args.validation_batch_size,
    )
    print(
        "validation result:",
        f"candidate_supervised_loss={candidate_supervised}",
        flush=True,
    )
    print("validation stage: target champion", flush=True)
    champion_target = target_validation(
        champion,
        replay_paths,
        args.device,
        args.validation_samples,
        args.validation_batch_size,
    )
    print(
        "validation result:",
        f"champion_target_ce={champion_target:.6f}",
        flush=True,
    )
    print("validation stage: target candidate", flush=True)
    candidate_target = target_validation(
        candidate,
        replay_paths,
        args.device,
        args.validation_samples,
        args.validation_batch_size,
    )
    print(
        "validation result:",
        f"candidate_target_ce={candidate_target:.6f}",
        flush=True,
    )

    persisted = load_cases(args.regression_data)
    combined = {case["fen"]: case for case in persisted}
    for case in current_cases:
        combined[case["fen"]] = case
    regression_cases = list(combined.values())

    champion_regression = regression_validation(
        champion,
        regression_cases,
        args,
        label="champion",
    )
    candidate_regression = regression_validation(
        candidate,
        regression_cases,
        args,
        label="candidate",
    )

    supervised_ok = (
        champion_supervised is None
        or candidate_supervised
        <= champion_supervised + args.max_supervised_loss_increase
    )
    target_ok = (
        candidate_target
        <= champion_target + args.max_target_ce_increase
    )
    candidate_accuracy = candidate_regression["accuracy"]
    regression_ok = (
        not regression_cases
        or (
            candidate_accuracy is not None
            and candidate_accuracy >= args.min_regression_accuracy
            and candidate_regression["correct"] + args.max_regression_drop
            >= champion_regression["correct"]
        )
    )

    arena_metrics = evaluate_models(
        candidate_path=candidate_path,
        baseline_path=champion_path,
        games=args.eval_games,
        sims=args.eval_sims,
        workers=args.eval_workers,
        device=args.device,
        max_plies=args.eval_max_plies,
        seed=args.seed + 10000 + iteration,
        opening_book=args.eval_opening_book,
        book_plies=args.eval_book_plies,
        max_book_positions=args.eval_max_book_positions,
        mcts_batch_size=args.eval_mcts_batch_size,
        movetime_ms=args.eval_movetime_ms,
        c_puct=search_arg(args, "c_puct", evaluation=True),
        mate_guard_plies=search_arg(args, "mate_guard_plies", evaluation=True),
        mate_guard_topk=search_arg(args, "mate_guard_topk", evaluation=True),
        mate_guard_nodes=search_arg(args, "mate_guard_nodes", evaluation=True),
        mate_guard_time_fraction=search_arg(
            args,
            "mate_guard_time_fraction",
            evaluation=True,
        ),
        q_tiebreak=search_arg(args, "q_tiebreak", evaluation=True),
        q_tiebreak_min_visits=search_arg(
            args,
            "q_tiebreak_min_visits",
            evaluation=True,
        ),
        q_tiebreak_p_ratio=search_arg(args, "q_tiebreak_p_ratio", evaluation=True),
        q_tiebreak_visit_ratio=search_arg(
            args,
            "q_tiebreak_visit_ratio",
            evaluation=True,
        ),
        q_tiebreak_margin=search_arg(args, "q_tiebreak_margin", evaluation=True),
        uci=args.uci,
        uci_depth=args.eval_uci_depth,
        uci_movetime_ms=args.eval_uci_movetime_ms,
        uci_multipv=args.eval_uci_multipv,
        uci_threads=args.uci_threads,
        uci_hash_mb=args.uci_hash_mb,
        teacher_cache=args.teacher_cache,
        log_every=args.log_every,
        progress=True,
    )
    arena_metrics = attach_arena_acceptance(
        arena_metrics,
        min_net_wins=args.eval_min_net_wins,
        min_acpl_improvement=args.eval_min_acpl_improvement,
        min_accuracy_improvement=args.eval_min_accuracy_improvement,
    )

    accepted = bool(
        arena_metrics.get("accepted")
        and supervised_ok
        and target_ok
        and regression_ok
    )
    supervised_delta = (
        None
        if champion_supervised is None or candidate_supervised is None
        else float(candidate_supervised - champion_supervised)
    )
    target_delta = float(candidate_target - champion_target)
    return {
        "accepted": accepted,
        "arena_ok": bool(arena_metrics.get("accepted")),
        "supervised_ok": bool(supervised_ok),
        "target_ok": bool(target_ok),
        "regression_ok": bool(regression_ok),
        "champion_supervised_loss": champion_supervised,
        "candidate_supervised_loss": candidate_supervised,
        "supervised_loss_delta": supervised_delta,
        "champion_target_ce": champion_target,
        "candidate_target_ce": candidate_target,
        "target_ce_delta": target_delta,
        "champion_regression": champion_regression,
        "candidate_regression": candidate_regression,
        "arena": arena_metrics,
    }


def run(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"champion model not found: {args.model}"
        )

    paths = prepare_run_paths(args)
    current_model = paths["current_model"]
    shutil.copy2(args.model, current_model)
    print("selflearn run id:", paths["run_id"], flush=True)
    print("selflearn data run directory:", paths["data_run_dir"], flush=True)
    print("selflearn model run directory:", paths["model_run_dir"], flush=True)
    print("selflearn initial model:", args.model, flush=True)
    print("selflearn current model:", current_model, flush=True)
    print("selflearn regression data:", args.regression_data, flush=True)
    print("selflearn teacher cache:", args.teacher_cache, flush=True)

    accepted_replay: List[str] = []
    for iteration in range(1, args.iterations + 1):
        print(f"self-learning iteration {iteration}")
        data_path = os.path.join(
            paths["data_dir"],
            f"selflearn_iter_{iteration}.h5",
        )
        candidate_path = os.path.join(
            paths["model_dir"],
            f"selflearn_candidate_iter_{iteration}.pth",
        )

        data_summary = generate_selflearn_data(
            args,
            model_path=current_model,
            output_path=data_path,
            iteration=iteration,
        )
        replay_paths = (
            accepted_replay + [data_path]
        )[-max(1, args.replay_window):]

        train_summary = train_candidate(
            args,
            champion_path=current_model,
            replay_paths=replay_paths,
            candidate_path=candidate_path,
            iteration=iteration,
        )
        print("selflearn train summary:", train_summary, flush=True)
        validation = validate_candidate(
            args,
            champion_path=current_model,
            candidate_path=candidate_path,
            replay_paths=replay_paths,
            iteration=iteration,
            current_cases=data_summary["regression_cases"],
        )

        print(
            "validation:",
            f"arena={validation['arena_ok']}",
            f"supervised={validation['supervised_ok']}",
            f"target={validation['target_ok']}",
            f"regression={validation['regression_ok']}",
            flush=True,
        )
        print(
            "validation details:",
            f"supervised_delta={validation['supervised_loss_delta']}",
            f"target_delta={validation['target_ce_delta']}",
            f"arena_result_ok={validation['arena'].get('result_ok')}",
            f"arena_quality_ok={validation['arena'].get('quality_ok')}",
            f"net_wins={validation['arena'].get('net_wins')}",
            flush=True,
        )
        print("arena quality:", validation["arena"]["quality"], flush=True)

        if validation["accepted"]:
            atomic_copy_with_backup(
                candidate_path,
                current_model,
                make_backup=not args.no_backup,
            )
            accepted_replay.append(data_path)
            merge_cases(
                args.regression_data,
                data_summary["regression_cases"],
                max_cases=args.regression_max_cases,
            )
            print("candidate accepted:", current_model)
        else:
            print("candidate rejected:", candidate_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Teacher-constrained AlphaZero-style self-learning"
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--supervised-data", default=H5_PATH)
    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--data-runs-dir", default=DEFAULT_DATA_RUNS_DIR)
    parser.add_argument("--model-runs-dir", default=DEFAULT_MODEL_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)

    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--selfplay-workers", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--truncate-adjudication-cp", type=int, default=200)
    parser.add_argument("--opening-book", default="data/openings.bin")
    parser.add_argument("--book-plies", type=int, default=8)
    parser.add_argument("--max-book-positions", type=int, default=50000)
    parser.add_argument("--sims", type=int, default=80)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=5000)
    parser.add_argument("--c-puct", type=float, default=1.5)

    parser.add_argument("--mate-guard-plies", type=int, default=3)
    parser.add_argument("--mate-guard-topk", type=int, default=8)
    parser.add_argument("--mate-guard-nodes", type=int, default=20000)
    parser.add_argument("--mate-guard-time-fraction", type=float, default=0.10)
    parser.add_argument("--q-tiebreak", action="store_true", default=True)
    parser.add_argument("--no-q-tiebreak", dest="q_tiebreak", action="store_false")
    parser.add_argument("--q-tiebreak-min-visits", type=int, default=32)
    parser.add_argument("--q-tiebreak-p-ratio", type=float, default=0.90)
    parser.add_argument("--q-tiebreak-visit-ratio", type=float, default=0.80)
    parser.add_argument("--q-tiebreak-margin", type=float, default=0.25)

    parser.add_argument("--uci-depth", type=int, default=10)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-multipv", type=int, default=8)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument(
        "--teacher-policy-temperature-cp",
        type=float,
        default=80.0,
    )
    parser.add_argument("--teacher-start-ply", type=int, default=4)
    parser.add_argument("--teacher-every", type=int, default=1)
    parser.add_argument("--teacher-sample-rate", type=float, default=1.0)
    parser.add_argument("--teacher-label-topk", type=int, default=4)
    parser.add_argument("--teacher-label-min-weight", type=float, default=0.20)
    parser.add_argument("--teacher-veto", action="store_true", default=True)
    parser.add_argument("--no-teacher-veto", dest="teacher_veto", action="store_false")
    parser.add_argument("--teacher-veto-regret-cp", type=int, default=300)
    parser.add_argument("--teacher-veto-min-weight", type=float, default=0.80)
    parser.add_argument(
        "--teacher-cache",
        default=DEFAULT_TEACHER_CACHE,
    )

    parser.add_argument("--epochs-per-iter", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--train-max-steps", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--replay-window", type=int, default=3)

    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--terminal-value-weight", type=float, default=0.25)
    parser.add_argument("--teacher-value-weight", type=float, default=0.20)
    parser.add_argument("--kl-weight", type=float, default=0.10)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--supervised-weight", type=float, default=0.35)
    parser.add_argument(
        "--supervised-value-weight",
        type=float,
        default=0.25,
    )

    parser.add_argument("--validation-samples", type=int, default=10000)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument(
        "--max-supervised-loss-increase",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--max-target-ce-increase",
        type=float,
        default=0.02,
    )

    parser.add_argument("--regression-data", default=REGRESSION_PATH)
    parser.add_argument(
        "--regression-answer-tolerance-cp",
        type=int,
        default=35,
    )
    parser.add_argument("--regression-max-answers", type=int, default=8)
    parser.add_argument("--regression-min-regret-cp", type=int, default=0)
    parser.add_argument(
        "--regression-min-teacher-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--regression-max-new-per-iter",
        type=int,
        default=500,
    )
    parser.add_argument("--regression-max-cases", type=int, default=2000)
    parser.add_argument("--regression-sims", type=int, default=200)
    parser.add_argument(
        "--regression-movetime-ms",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--min-regression-accuracy",
        type=float,
        default=1.0,
    )
    parser.add_argument("--max-regression-drop", type=int, default=0)

    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=80)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--eval-max-plies", type=int, default=240)
    parser.add_argument(
        "--eval-opening-book",
        default="data/openings.bin",
    )
    parser.add_argument("--eval-book-plies", type=int, default=8)
    parser.add_argument(
        "--eval-max-book-positions",
        type=int,
        default=50000,
    )
    parser.add_argument("--eval-min-net-wins", type=int, default=5)
    parser.add_argument("--eval-mcts-batch-size", type=int, default=32)
    parser.add_argument("--eval-movetime-ms", type=int, default=5000)
    parser.add_argument("--eval-c-puct", type=float, default=None)
    parser.add_argument("--eval-mate-guard-plies", type=int, default=None)
    parser.add_argument("--eval-mate-guard-topk", type=int, default=None)
    parser.add_argument("--eval-mate-guard-nodes", type=int, default=None)
    parser.add_argument("--eval-mate-guard-time-fraction", type=float, default=None)
    parser.add_argument(
        "--eval-q-tiebreak",
        dest="eval_q_tiebreak",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-eval-q-tiebreak",
        dest="eval_q_tiebreak",
        action="store_false",
    )
    parser.add_argument("--eval-q-tiebreak-min-visits", type=int, default=None)
    parser.add_argument("--eval-q-tiebreak-p-ratio", type=float, default=None)
    parser.add_argument("--eval-q-tiebreak-visit-ratio", type=float, default=None)
    parser.add_argument("--eval-q-tiebreak-margin", type=float, default=None)
    parser.add_argument("--eval-uci-depth", type=int, default=8)
    parser.add_argument(
        "--eval-uci-movetime-ms",
        type=int,
        default=0,
    )
    parser.add_argument("--eval-uci-multipv", type=int, default=4)
    parser.add_argument(
        "--eval-min-acpl-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--eval-min-accuracy-improvement",
        type=float,
        default=0.0,
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-backup", action="store_true", default=False)
    args = parser.parse_args()
    if args.parallel is not None:
        if args.selfplay_workers is None:
            args.selfplay_workers = args.parallel
        if args.eval_workers is None:
            args.eval_workers = args.parallel
    if args.selfplay_workers is None:
        args.selfplay_workers = 1
    if args.eval_workers is None:
        args.eval_workers = 1
    args.selfplay_workers = max(1, int(args.selfplay_workers))
    args.eval_workers = max(1, int(args.eval_workers))
    return args


if __name__ == "__main__":
    run(parse_args())
