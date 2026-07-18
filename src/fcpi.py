"""Architecture-dispatched Folded Counterfactual Policy Iteration.

FCPI is teacher-free.  Each registered architecture owns its behavior policy,
counterfactual target construction, training loss, and validation metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import random
import shutil
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from acceptance import attach_arena_acceptance
from architectures import RESNET_PVA_GAD, RESNET_PV_LINEAR
from arena import evaluate_models
from checkpoint_io import atomic_copy_with_backup
from config import DEVICE
from decision import profile_for_model
from evaluator import BatchedEvaluator
from model import checkpoint_metadata, load_model, save_model
from opening_book import make_sampling_specs


DATA_RUNS_DIR = os.path.join("data", "runs")
MODEL_RUNS_DIR = os.path.join("models", "runs")


@dataclass
class RawPosition:
    game_id: int
    state: np.ndarray
    fen: str
    root_value: float
    legal_indices: np.ndarray
    legal_prior: np.ndarray
    played_index: int
    candidate_indices: np.ndarray
    model_advantages: Optional[np.ndarray] = None
    value_target: float = 0.0
    policy_target: Optional[np.ndarray] = None
    candidate_q: Optional[np.ndarray] = None
    advantage_target: Optional[np.ndarray] = None


@dataclass
class Trajectory:
    game_id: int
    board: chess.Board
    positions: List[RawPosition]


@dataclass
class CounterfactualBranch:
    record_index: int
    candidate_index: int
    board: chess.Board
    depth: int
    estimates: List[float]
    current_value: Optional[float] = None
    current_policy: Optional[np.ndarray] = None
    current_payload: Optional[np.ndarray] = None
    last_residual: float = 0.0
    last_change: float = 0.0
    terminal: bool = False


def create_run_id() -> str:
    return time.strftime("fcpi_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]


def prepare_run_paths() -> Dict[str, str]:
    run_id = os.environ.get("FCPI_RUN_ID") or create_run_id()
    run_id = str(run_id).strip().replace("\\", "/").split("/")[-1]
    if not run_id.startswith("fcpi_"):
        raise ValueError("system run id must start with fcpi_")
    data_dir = os.path.join(DATA_RUNS_DIR, run_id)
    model_dir = os.path.join(MODEL_RUNS_DIR, run_id)
    if os.path.exists(model_dir):
        raise FileExistsError(f"FCPI model run already exists: {run_id}")
    if os.path.exists(data_dir):
        unexpected = set(os.listdir(data_dir)) - {"info.log", "pid"}
        if unexpected:
            raise FileExistsError(
                f"FCPI data run contains prior artifacts: {run_id}: {sorted(unexpected)}"
            )
    else:
        os.makedirs(data_dir)
    os.makedirs(model_dir)
    return {
        "run_id": run_id,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "current_model": os.path.join(model_dir, "current.pth"),
    }


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float64)
    return values / total


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - float(np.max(logits))
    return normalize(np.exp(np.clip(logits, -80.0, 0.0)))


def sample_local_index(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probabilities = normalize(probabilities)
    return int(rng.choice(len(probabilities), p=probabilities))


def terminal_value(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def result_label(board: chess.Board, truncated: bool) -> str:
    if truncated and not board.is_game_over(claim_draw=True):
        return "bootstrap"
    return board.result(claim_draw=True)


def choose_top_actions(
    legal_indices: np.ndarray,
    scores: np.ndarray,
    played_index: int,
    topk: int,
) -> np.ndarray:
    count = min(max(1, int(topk)), len(legal_indices))
    order = np.argsort(-np.asarray(scores), kind="stable")[:count]
    selected = [int(legal_indices[index]) for index in order]
    if int(played_index) not in selected:
        selected[-1] = int(played_index)
    return np.asarray(list(dict.fromkeys(selected)), dtype=np.int64)


def mixed_depth_q(estimates: Sequence[float], trace_lambda: float) -> float:
    if not estimates:
        raise ValueError("counterfactual branch has no value estimates")
    if len(estimates) == 1:
        return float(estimates[0])
    lam = float(np.clip(trace_lambda, 0.0, 1.0))
    total = 0.0
    for depth, estimate in enumerate(estimates[:-1]):
        total += (1.0 - lam) * (lam ** depth) * float(estimate)
    total += (lam ** (len(estimates) - 1)) * float(estimates[-1])
    return float(np.clip(total, -1.0, 1.0))


def evaluate_branch_frontier(
    branches: Sequence[CounterfactualBranch],
    evaluator: BatchedEvaluator,
):
    pending = [branch for branch in branches if not branch.terminal]
    if not pending:
        return
    boards = [branch.board for branch in pending]
    batch = evaluator.evaluate_boards_full(boards)
    profile = evaluator.profile
    for index, branch in enumerate(pending):
        value = float(batch.values[index])
        branch.current_value = value
        branch.current_policy = batch.policies[index]
        branch.current_payload = profile.payload_for_index(batch.expansion_payload, index)
        branch.estimates.append(float(((-1.0) ** branch.depth) * value))


def adaptive_successor_q_batches(
    records: Sequence[RawPosition],
    evaluator: BatchedEvaluator,
    evolution,
    args,
):
    output: List[np.ndarray] = []
    all_depths = []
    all_residuals = []
    all_changes = []
    terminal_branches = 0
    requested_depth_total = 0
    records_per_batch = max(1, int(args.target_records_per_batch))
    min_plies = max(1, int(evolution.counterfactual_min_plies(args)))
    max_plies = max(min_plies, int(evolution.counterfactual_max_plies(args)))
    target_average_plies = float(
        np.clip(
            evolution.counterfactual_target_average_plies(args),
            min_plies,
            max_plies,
        )
    )
    trace_lambda = float(evolution.counterfactual_lambda(args))
    total_records = len(records)
    estimated_branches = sum(len(record.candidate_indices) for record in records)
    estimated_branch_plies = int(round(estimated_branches * target_average_plies))
    progress_started = time.perf_counter()
    last_logged_records = 0
    log_every = max(1, int(args.log_every))
    print(
        "fcpi counterfactual start:",
        f"positions={total_records}",
        f"branches={estimated_branches}",
        f"target_average_plies={target_average_plies:.2f}",
        f"estimated_branch_plies={estimated_branch_plies}",
        flush=True,
    )

    for start in range(0, len(records), records_per_batch):
        subset = records[start:start + records_per_batch]
        branches: List[CounterfactualBranch] = []
        profile = evaluator.profile
        for record_index, row in enumerate(subset):
            board = chess.Board(row.fen)
            for candidate_index, encoded_move in enumerate(row.candidate_indices):
                move = profile.move_codec.index_to_move(int(encoded_move), board)
                if move is None:
                    raise RuntimeError(
                        f"FCPI candidate action is illegal: action={encoded_move} fen={row.fen}"
                    )
                child = board.copy(stack=False)
                child.push(move)
                branch = CounterfactualBranch(
                    record_index=record_index,
                    candidate_index=candidate_index,
                    board=child,
                    depth=1,
                    estimates=[],
                )
                if child.is_game_over(claim_draw=True):
                    value = terminal_value(child)
                    branch.estimates.append(-float(value))
                    branch.current_value = float(value)
                    branch.terminal = True
                    terminal_branches += 1
                branches.append(branch)

        evaluate_branch_frontier(branches, evaluator)
        for branch in branches:
            if branch.terminal:
                continue
            row = subset[branch.record_index]
            branch.last_residual = abs(float(row.root_value) + float(branch.current_value))
            branch.last_change = abs(
                float(branch.estimates[-1])
                - float(evolution.candidate_reference_q(row, branch.candidate_index))
            )
            all_residuals.append(branch.last_residual)
            all_changes.append(branch.last_change)

        requested_depth_total += int(round(len(branches) * target_average_plies))

        while True:
            expandable = [
                branch
                for branch in branches
                if not branch.terminal and branch.depth < max_plies
            ]
            if not expandable:
                break

            best_by_record = {}
            for branch in branches:
                latest = float(branch.estimates[-1])
                best_by_record[branch.record_index] = max(
                    latest,
                    best_by_record.get(branch.record_index, -float("inf")),
                )

            required = [branch for branch in expandable if branch.depth < min_plies]
            if required:
                active = required
            else:
                target_depth_total = int(round(len(branches) * target_average_plies))
                remaining_budget = target_depth_total - sum(branch.depth for branch in branches)
                if remaining_budget <= 0:
                    break

                ranked = sorted(
                    expandable,
                    key=lambda branch: evolution.counterfactual_priority(
                        branch,
                        best_by_record[branch.record_index],
                    ),
                    reverse=True,
                )
                active = []
                future_capacity = 0
                for branch in ranked:
                    active.append(branch)
                    future_capacity += max_plies - branch.depth
                    if future_capacity >= remaining_budget:
                        break
                active = active[:remaining_budget]
            if not active:
                break

            previous_values = {}
            previous_estimates = {}
            for branch in active:
                if branch.current_policy is None or branch.current_value is None:
                    raise RuntimeError("FCPI adaptive rollout branch is missing an evaluation")
                previous_values[id(branch)] = float(branch.current_value)
                previous_estimates[id(branch)] = float(branch.estimates[-1])
                move = evolution.rollout_move(
                    branch.board,
                    branch.current_policy,
                    branch.current_payload,
                    args,
                    profile,
                )
                if move not in branch.board.legal_moves:
                    raise RuntimeError(
                        f"FCPI rollout returned illegal move: {move} fen={branch.board.fen()}"
                    )
                branch.board.push(move)
                branch.depth += 1
                branch.current_policy = None
                branch.current_payload = None
                if branch.board.is_game_over(claim_draw=True):
                    value = terminal_value(branch.board)
                    branch.current_value = float(value)
                    branch.estimates.append(float(((-1.0) ** branch.depth) * value))
                    branch.terminal = True
                    terminal_branches += 1

            evaluate_branch_frontier(active, evaluator)
            for branch in active:
                previous_value = previous_values[id(branch)]
                current_value = float(branch.current_value)
                branch.last_residual = abs(previous_value + current_value)
                branch.last_change = abs(
                    float(branch.estimates[-1]) - previous_estimates[id(branch)]
                )
                all_residuals.append(branch.last_residual)
                all_changes.append(branch.last_change)

        subset_q = [
            np.zeros(len(row.candidate_indices), dtype=np.float32)
            for row in subset
        ]
        for branch in branches:
            subset_q[branch.record_index][branch.candidate_index] = mixed_depth_q(
                branch.estimates,
                trace_lambda,
            )
            all_depths.append(branch.depth)
        output.extend(subset_q)

        processed_records = min(start + len(subset), total_records)
        if (
            processed_records == total_records
            or processed_records - last_logged_records >= log_every
        ):
            elapsed = max(1e-9, time.perf_counter() - progress_started)
            print(
                "fcpi counterfactual:",
                f"positions={processed_records}/{total_records}",
                f"branches={len(all_depths)}/{estimated_branches}",
                f"branch_plies={sum(all_depths)}/{estimated_branch_plies}",
                f"positions_per_sec={processed_records / elapsed:.2f}",
                flush=True,
            )
            last_logged_records = processed_records

    summary = {
        "branches": len(all_depths),
        "average_depth": float(np.mean(all_depths)) if all_depths else 0.0,
        "max_depth": int(max(all_depths)) if all_depths else 0,
        "depth_histogram": {
            str(depth): int(sum(value == depth for value in all_depths))
            for depth in sorted(set(all_depths))
        },
        "terminal_branches": int(terminal_branches),
        "mean_residual": float(np.mean(all_residuals)) if all_residuals else 0.0,
        "p90_residual": float(np.percentile(all_residuals, 90)) if all_residuals else 0.0,
        "mean_change": float(np.mean(all_changes)) if all_changes else 0.0,
        "p90_change": float(np.percentile(all_changes, 90)) if all_changes else 0.0,
        "min_plies": int(min_plies),
        "max_plies": int(max_plies),
        "target_average_plies": float(target_average_plies),
        "requested_branch_plies": int(requested_depth_total),
        "actual_branch_plies": int(sum(all_depths)),
        "budget_utilization": (
            float(sum(all_depths) / requested_depth_total)
            if requested_depth_total > 0
            else 0.0
        ),
        "trace_lambda": float(trace_lambda),
    }
    return output, summary


def assign_td_lambda_returns(
    trajectories: Sequence[Trajectory],
    evaluator: BatchedEvaluator,
    td_lambda: float,
):
    truncated = [
        trajectory.board
        for trajectory in trajectories
        if not trajectory.board.is_game_over(claim_draw=True)
    ]
    truncated_values = iter(
        evaluator.evaluate_boards_full(truncated).values.tolist() if truncated else []
    )
    lam = float(np.clip(td_lambda, 0.0, 1.0))
    for trajectory in trajectories:
        if trajectory.board.is_game_over(claim_draw=True):
            next_return = terminal_value(trajectory.board)
        else:
            next_return = float(next(truncated_values))
        final_value = next_return
        positions = trajectory.positions
        for index in range(len(positions) - 1, -1, -1):
            next_value = (
                final_value
                if index + 1 == len(positions)
                else float(positions[index + 1].root_value)
            )
            current_return = -((1.0 - lam) * next_value + lam * next_return)
            positions[index].value_target = float(np.clip(current_return, -1.0, 1.0))
            next_return = current_return


def sample_trajectory_positions(
    trajectories: Sequence[Trajectory],
    positions_per_game: int,
    rng: np.random.Generator,
) -> Tuple[List[RawPosition], Dict[str, int]]:
    limit = max(1, int(positions_per_game))
    selected: List[RawPosition] = []
    source_positions = 0
    unique_positions = 0
    capped_games = 0
    for trajectory in trajectories:
        source_positions += len(trajectory.positions)
        unique = []
        seen_states = set()
        for position in trajectory.positions:
            state_key = np.ascontiguousarray(position.state).tobytes()
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            unique.append(position)
        unique_positions += len(unique)
        if len(unique) > limit:
            indices = np.sort(rng.choice(len(unique), size=limit, replace=False))
            selected.extend(unique[int(index)] for index in indices)
            capped_games += 1
        else:
            selected.extend(unique)
    return selected, {
        "games": len(trajectories),
        "source_positions": source_positions,
        "unique_positions": unique_positions,
        "selected_positions": len(selected),
        "positions_per_game": limit,
        "capped_games": capped_games,
    }


def collect_selfplay(
    evolution,
    model_path: str,
    args,
    iteration: int,
) -> List[RawPosition]:
    model = load_model(model_path, device=args.device)
    evaluator = BatchedEvaluator(model, device=args.device, batch_size=args.inference_batch_size)
    profile = profile_for_model(model)
    specs, opening_summary = make_sampling_specs(
        games=args.games_per_iter,
        seed=args.seed + iteration,
        opening_book=args.opening_book,
        book_plies=args.book_plies,
        max_positions=args.max_book_positions,
        startpos_fraction=args.startpos_fraction,
    )
    rng = np.random.default_rng(args.seed + iteration)
    trajectories: List[Trajectory] = []
    game_id = 0
    print(
        "fcpi self-play start:",
        f"iteration={iteration}",
        f"arch_type={evolution.arch_type}",
        f"games={len(specs)}",
        f"max_plies={args.max_plies}",
        f"device={args.device}",
        flush=True,
    )
    print(
        "fcpi starts:",
        json.dumps(opening_summary, ensure_ascii=False),
        flush=True,
    )

    in_flight = max(1, int(args.games_in_flight))
    for group_start in range(0, len(specs), in_flight):
        group = specs[group_start:group_start + in_flight]
        active = []
        for fen, _ in group:
            game_id += 1
            active.append(Trajectory(game_id, chess.Board(fen), []))

        while active:
            boards = [trajectory.board for trajectory in active]
            batch = evaluator.evaluate_boards_full(boards)
            next_active = []
            for batch_index, trajectory in enumerate(active):
                board = trajectory.board
                full_policy = batch.policies[batch_index]
                root_value = float(batch.values[batch_index])
                payload = profile.payload_for_index(batch.expansion_payload, batch_index)
                legal_moves = list(board.legal_moves)
                legal_indices = np.asarray(
                    [profile.move_codec.move_to_index(move) for move in legal_moves],
                    dtype=np.int64,
                )
                legal_prior = normalize(full_policy[legal_indices]).astype(np.float32)
                behavior, model_advantages, ranking_scores = evolution.behavior_distribution(
                    legal_indices,
                    legal_prior,
                    payload,
                    args,
                )
                local_choice = sample_local_index(behavior, rng)
                move = legal_moves[local_choice]
                played_index = int(legal_indices[local_choice])
                candidate_indices = choose_top_actions(
                    legal_indices,
                    ranking_scores,
                    played_index,
                    evolution.counterfactual_topk(args),
                )
                trajectory.positions.append(
                    RawPosition(
                        game_id=trajectory.game_id,
                        state=profile.state_codec.encode_board(board),
                        fen=board.fen(),
                        root_value=root_value,
                        legal_indices=legal_indices,
                        legal_prior=legal_prior,
                        played_index=played_index,
                        candidate_indices=candidate_indices,
                        model_advantages=model_advantages,
                    )
                )
                board.push(move)
                game_over = board.is_game_over(claim_draw=True)
                reached_limit = len(trajectory.positions) >= args.max_plies
                if game_over or reached_limit:
                    trajectories.append(trajectory)
                    truncated = reached_limit and not game_over
                    print(
                        "fcpi game:",
                        f"completed={len(trajectories)}/{len(specs)}",
                        f"game_id={trajectory.game_id}",
                        f"plies={len(trajectory.positions)}",
                        f"result={result_label(board, truncated)}",
                        f"truncated={str(truncated).lower()}",
                        flush=True,
                    )
                else:
                    next_active.append(trajectory)
            active = next_active

    assign_td_lambda_returns(
        trajectories,
        evaluator,
        evolution.td_lambda(args),
    )
    sample_rng = np.random.default_rng(args.seed + iteration + 1_000_003)
    records, sampling_summary = sample_trajectory_positions(
        trajectories,
        args.positions_per_game,
        sample_rng,
    )
    sampling_summary["starts"] = opening_summary
    evolution.last_sampling_summary = sampling_summary
    print(
        "fcpi position sampling:",
        json.dumps(sampling_summary, ensure_ascii=False),
        flush=True,
    )
    evolution.construct_targets(records, evaluator, args)
    print(
        "fcpi self-play finished:",
        f"games={len(trajectories)}",
        f"positions={len(records)}",
        flush=True,
    )
    return records


def padded_rows(records, field: str, dtype, width: int):
    output = np.zeros((len(records), width), dtype=dtype)
    for row, record in enumerate(records):
        values = np.asarray(getattr(record, field), dtype=dtype)
        output[row, :len(values)] = values
    return output


def merge_target_record_group(group: Sequence[RawPosition]) -> RawPosition:
    merged = group[0]
    if len(group) == 1:
        return merged

    for record in group[1:]:
        if not np.array_equal(record.legal_indices, merged.legal_indices):
            raise RuntimeError("identical encoded states produced different legal actions")

    merged.legal_prior = normalize(
        np.mean(np.stack([record.legal_prior for record in group]), axis=0)
    ).astype(np.float32)
    merged.policy_target = normalize(
        np.mean(np.stack([record.policy_target for record in group]), axis=0)
    ).astype(np.float32)
    merged.value_target = float(np.mean([record.value_target for record in group]))

    candidate_q = {}
    advantage = {}
    for record in group:
        for action, value in zip(record.candidate_indices, record.candidate_q):
            candidate_q.setdefault(int(action), []).append(float(value))
        if record.advantage_target is not None:
            for action, value in zip(record.candidate_indices, record.advantage_target):
                advantage.setdefault(int(action), []).append(float(value))

    candidate_set = set(candidate_q)
    candidate_indices = [
        int(action) for action in merged.legal_indices if int(action) in candidate_set
    ]
    merged.candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    merged.candidate_q = np.asarray(
        [np.mean(candidate_q[action]) for action in candidate_indices],
        dtype=np.float32,
    )
    if advantage:
        merged.advantage_target = np.asarray(
            [np.mean(advantage[action]) for action in candidate_indices],
            dtype=np.float32,
        )
    return merged


def aggregate_target_records(
    records: Sequence[RawPosition],
    split_game: int,
) -> Tuple[List[RawPosition], Dict[str, int]]:
    groups = {}
    for record in records:
        split = 1 if record.game_id > split_game else 0
        state_key = np.ascontiguousarray(record.state).tobytes()
        groups.setdefault((split, state_key), []).append(record)

    merged = [merge_target_record_group(group) for group in groups.values()]
    multiplicities = [len(group) for group in groups.values()]
    return merged, {
        "source_positions": len(records),
        "aggregated_positions": len(merged),
        "merged_positions": len(records) - len(merged),
        "duplicate_groups": sum(size > 1 for size in multiplicities),
        "max_group_size": max(multiplicities, default=0),
    }


def write_base_fcpi_h5(path: str, records: Sequence[RawPosition], evolution, args):
    if not records:
        raise RuntimeError("FCPI generated no positions")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    split_game = max(1, int(args.games_per_iter * (1.0 - args.validation_fraction)))
    records, aggregation_summary = aggregate_target_records(records, split_game)
    print(
        "fcpi position aggregation:",
        json.dumps(aggregation_summary, ensure_ascii=False),
        flush=True,
    )
    legal_width = max(len(record.legal_indices) for record in records)
    candidate_width = max(len(record.candidate_indices) for record in records)
    with h5py.File(path, "w") as h5:
        h5.attrs["arch_type"] = evolution.arch_type
        h5.attrs["fcpi_formula"] = evolution.formula_name
        h5.create_dataset(
            "states",
            data=np.stack([record.state for record in records]),
            compression="lzf",
        )
        h5.create_dataset(
            "legal_indices",
            data=padded_rows(records, "legal_indices", np.int32, legal_width),
            compression="lzf",
        )
        h5.create_dataset(
            "legal_priors",
            data=padded_rows(records, "legal_prior", np.float32, legal_width),
            compression="lzf",
        )
        h5.create_dataset(
            "policy_targets",
            data=padded_rows(records, "policy_target", np.float32, legal_width),
            compression="lzf",
        )
        h5.create_dataset(
            "legal_counts",
            data=np.asarray([len(record.legal_indices) for record in records], dtype=np.uint8),
        )
        h5.create_dataset(
            "value_targets",
            data=np.asarray([record.value_target for record in records], dtype=np.float32),
        )
        h5.create_dataset(
            "candidate_indices",
            data=padded_rows(records, "candidate_indices", np.int32, candidate_width),
            compression="lzf",
        )
        h5.create_dataset(
            "candidate_q",
            data=padded_rows(records, "candidate_q", np.float32, candidate_width),
            compression="lzf",
        )
        h5.create_dataset(
            "candidate_counts",
            data=np.asarray([len(record.candidate_indices) for record in records], dtype=np.uint8),
        )
        h5.create_dataset(
            "split",
            data=np.asarray([1 if record.game_id > split_game else 0 for record in records], dtype=np.uint8),
        )
        evolution.write_architecture_targets(h5, records, candidate_width)
    return {
        "path": path,
        "positions": len(records),
        "train_positions": sum(record.game_id <= split_game for record in records),
        "validation_positions": sum(record.game_id > split_game for record in records),
        "legal_width": legal_width,
        "counterfactual_width": candidate_width,
        "formula": evolution.formula_name,
        "sampling": dict(getattr(evolution, "last_sampling_summary", {})),
        "aggregation": aggregation_summary,
        "counterfactual": dict(getattr(evolution, "last_target_summary", {})),
    }


class FCPIH5Dataset(Dataset):
    def __init__(self, path: str, split: int, arch_type: str):
        self.path = path
        self.arch_type = arch_type
        with h5py.File(path, "r") as h5:
            stored_arch = str(h5.attrs["arch_type"])
            if stored_arch != arch_type:
                raise ValueError(f"FCPI H5 arch mismatch: expected {arch_type}, got {stored_arch}")
            self.indices = np.flatnonzero(np.asarray(h5["split"]) == int(split)).astype(np.int64)
        self._file = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        return state

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.read_row(self._open(), int(self.indices[item]))

    def read_row(self, h5, index: int):
        raise NotImplementedError


class PVLinearDataset(FCPIH5Dataset):
    def read_row(self, h5, index: int):
        count = int(h5["legal_counts"][index])
        return (
            torch.from_numpy(np.unpackbits(h5["states"][index][..., None], axis=-1).astype(np.float32)),
            torch.from_numpy(np.asarray(h5["legal_indices"][index], dtype=np.int64)),
            torch.from_numpy(np.asarray(h5["legal_priors"][index], dtype=np.float32)),
            torch.from_numpy(np.asarray(h5["policy_targets"][index], dtype=np.float32)),
            torch.tensor(count, dtype=torch.long),
            torch.tensor(float(h5["value_targets"][index]), dtype=torch.float32),
        )


class PVAGadDataset(FCPIH5Dataset):
    def read_row(self, h5, index: int):
        legal_count = int(h5["legal_counts"][index])
        candidate_count = int(h5["candidate_counts"][index])
        return (
            torch.from_numpy(np.asarray(h5["states"][index], dtype=np.int64)),
            torch.from_numpy(np.asarray(h5["legal_indices"][index], dtype=np.int64)),
            torch.from_numpy(np.asarray(h5["legal_priors"][index], dtype=np.float32)),
            torch.from_numpy(np.asarray(h5["policy_targets"][index], dtype=np.float32)),
            torch.tensor(legal_count, dtype=torch.long),
            torch.tensor(float(h5["value_targets"][index]), dtype=torch.float32),
            torch.from_numpy(np.asarray(h5["candidate_indices"][index], dtype=np.int64)),
            torch.from_numpy(np.asarray(h5["advantage_targets"][index], dtype=np.float32)),
            torch.tensor(candidate_count, dtype=torch.long),
        )


def masked_policy_terms(logits, indices, priors, targets, counts):
    # Policy masking and normalization stay in float32 under CUDA autocast.
    selected = logits.gather(1, indices).float()
    width = selected.shape[1]
    mask = torch.arange(width, device=selected.device).unsqueeze(0) < counts.unsqueeze(1)
    selected = selected.masked_fill(~mask, -1e9)
    log_prob = torch.log_softmax(selected, dim=1)
    targets = targets * mask
    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-8)
    priors = priors.clamp_min(1e-8) * mask
    priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-8)
    policy_loss = -(targets * log_prob).sum(dim=1).mean()
    kl = (torch.exp(log_prob) * (log_prob - torch.log(priors.clamp_min(1e-8)))).sum(dim=1).mean()
    entropy = -(torch.exp(log_prob) * log_prob).sum(dim=1).mean()
    return policy_loss, kl, entropy


class ResNetPVLinearFCPI:
    arch_type = RESNET_PV_LINEAR
    formula_name = "pv_linear_adaptive_value_expansion_td_kl"
    dataset_type = PVLinearDataset

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("resnet_pv_linear FCPI")
        group.add_argument("--td-lambda", dest="resnet_pv_linear_td_lambda", type=float, default=0.80)
        group.add_argument("--counterfactual-topk", dest="resnet_pv_linear_counterfactual_topk", type=int, default=6)
        group.add_argument("--counterfactual-min-plies", dest="resnet_pv_linear_counterfactual_min_plies", type=int, default=2)
        group.add_argument("--counterfactual-max-plies", dest="resnet_pv_linear_counterfactual_max_plies", type=int, default=6)
        group.add_argument(
            "--counterfactual-target-average-plies",
            dest="resnet_pv_linear_counterfactual_target_average_plies",
            type=float,
            default=4.0,
        )
        group.add_argument("--counterfactual-lambda", dest="resnet_pv_linear_counterfactual_lambda", type=float, default=0.80)
        group.add_argument("--behavior-temperature", dest="resnet_pv_linear_behavior_temperature", type=float, default=0.80)
        group.add_argument("--uniform-mix", dest="resnet_pv_linear_uniform_mix", type=float, default=0.02)
        group.add_argument("--policy-temperature", dest="resnet_pv_linear_policy_temperature", type=float, default=0.25)
        group.add_argument("--prior-power", dest="resnet_pv_linear_prior_power", type=float, default=1.0)
        group.add_argument("--played-return-weight", dest="resnet_pv_linear_played_return_weight", type=float, default=0.50)
        group.add_argument("--policy-weight", dest="resnet_pv_linear_policy_weight", type=float, default=1.0)
        group.add_argument("--value-weight", dest="resnet_pv_linear_value_weight", type=float, default=1.0)
        group.add_argument("--kl-weight", dest="resnet_pv_linear_kl_weight", type=float, default=0.05)
        group.add_argument("--entropy-weight", dest="resnet_pv_linear_entropy_weight", type=float, default=0.001)

    @staticmethod
    def td_lambda(args):
        return args.resnet_pv_linear_td_lambda

    @staticmethod
    def counterfactual_topk(args):
        return args.resnet_pv_linear_counterfactual_topk

    @staticmethod
    def counterfactual_min_plies(args):
        return args.resnet_pv_linear_counterfactual_min_plies

    @staticmethod
    def counterfactual_max_plies(args):
        return args.resnet_pv_linear_counterfactual_max_plies

    @staticmethod
    def counterfactual_target_average_plies(args):
        return args.resnet_pv_linear_counterfactual_target_average_plies

    @staticmethod
    def counterfactual_lambda(args):
        return args.resnet_pv_linear_counterfactual_lambda

    @staticmethod
    def candidate_reference_q(record, candidate_index):
        del candidate_index
        return float(record.root_value)

    @staticmethod
    def counterfactual_priority(branch, best_q):
        competitiveness = 1.0 - min(1.0, max(0.0, best_q - branch.estimates[-1]) / 2.0)
        return max(branch.last_residual, branch.last_change) + 0.05 * competitiveness

    def behavior_distribution(self, legal_indices, legal_prior, payload, args):
        del legal_indices, payload
        temperature = max(1e-4, float(args.resnet_pv_linear_behavior_temperature))
        behavior = normalize(np.power(np.clip(legal_prior, 1e-12, 1.0), 1.0 / temperature))
        mix = float(np.clip(args.resnet_pv_linear_uniform_mix, 0.0, 1.0))
        behavior = (1.0 - mix) * behavior + mix / len(behavior)
        return behavior, None, legal_prior

    @staticmethod
    def rollout_move(board, policy, payload, args, profile):
        del payload, args
        legal_moves = list(board.legal_moves)
        legal_indices = np.asarray(
            [profile.move_codec.move_to_index(move) for move in legal_moves],
            dtype=np.int64,
        )
        return legal_moves[int(np.argmax(policy[legal_indices]))]

    def construct_targets(self, records, evaluator, args):
        successor, self.last_target_summary = adaptive_successor_q_batches(
            records,
            evaluator,
            self,
            args,
        )
        print(
            "fcpi counterfactual summary:",
            json.dumps(self.last_target_summary, ensure_ascii=False),
            flush=True,
        )
        played_weight = float(
            np.clip(args.resnet_pv_linear_played_return_weight, 0.0, 1.0)
        )
        temperature = max(1e-4, float(args.resnet_pv_linear_policy_temperature))
        for record, candidate_q in zip(records, successor):
            candidate_q = np.asarray(candidate_q, dtype=np.float32)
            played = np.flatnonzero(record.candidate_indices == record.played_index)
            if played.size:
                index = int(played[0])
                candidate_q[index] = (
                    (1.0 - played_weight) * candidate_q[index]
                    + played_weight * record.value_target
                )
            q_all = np.full(len(record.legal_indices), record.root_value, dtype=np.float32)
            positions = {int(action): index for index, action in enumerate(record.legal_indices)}
            for action, q_value in zip(record.candidate_indices, candidate_q):
                q_all[positions[int(action)]] = q_value
            logits = (
                float(args.resnet_pv_linear_prior_power)
                * np.log(np.clip(record.legal_prior, 1e-12, 1.0))
                + (q_all - record.root_value) / temperature
            )
            record.policy_target = stable_softmax(logits).astype(np.float32)
            record.candidate_q = candidate_q

    @staticmethod
    def write_architecture_targets(h5, records, candidate_width):
        del h5, records, candidate_width

    def train(self, source_model, data_path, candidate_path, args):
        return train_fcpi_model(self, source_model, data_path, candidate_path, args)

    def batch_loss(self, model, batch, args, device):
        state, legal, priors, targets, counts, value_target = [item.to(device) for item in batch]
        heads = model.forward_heads(state)
        policy, kl, entropy = masked_policy_terms(
            heads["policy_logits"], legal, priors, targets, counts
        )
        value = torch.nn.functional.smooth_l1_loss(
            heads["value"].squeeze(1), value_target
        )
        loss = (
            args.resnet_pv_linear_policy_weight * policy
            + args.resnet_pv_linear_value_weight * value
            + args.resnet_pv_linear_kl_weight * kl
            - args.resnet_pv_linear_entropy_weight * entropy
        )
        return loss, {"policy": policy, "value": value, "kl": kl, "entropy": entropy}


class ResNetPVAGadFCPI:
    arch_type = RESNET_PVA_GAD
    formula_name = "pva_gad_adaptive_dueling_value_expansion_td_kl"
    dataset_type = PVAGadDataset

    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("resnet_pva_gad FCPI")
        group.add_argument("--td-lambda", dest="resnet_pva_gad_td_lambda", type=float, default=0.85)
        group.add_argument("--counterfactual-topk", dest="resnet_pva_gad_counterfactual_topk", type=int, default=8)
        group.add_argument("--counterfactual-min-plies", dest="resnet_pva_gad_counterfactual_min_plies", type=int, default=2)
        group.add_argument("--counterfactual-max-plies", dest="resnet_pva_gad_counterfactual_max_plies", type=int, default=6)
        group.add_argument(
            "--counterfactual-target-average-plies",
            dest="resnet_pva_gad_counterfactual_target_average_plies",
            type=float,
            default=4.0,
        )
        group.add_argument("--counterfactual-lambda", dest="resnet_pva_gad_counterfactual_lambda", type=float, default=0.85)
        group.add_argument("--behavior-temperature", dest="resnet_pva_gad_behavior_temperature", type=float, default=0.85)
        group.add_argument("--uniform-mix", dest="resnet_pva_gad_uniform_mix", type=float, default=0.02)
        group.add_argument("--behavior-advantage-weight", dest="resnet_pva_gad_behavior_advantage_weight", type=float, default=0.50)
        group.add_argument("--policy-temperature", dest="resnet_pva_gad_policy_temperature", type=float, default=0.25)
        group.add_argument("--prior-power", dest="resnet_pva_gad_prior_power", type=float, default=1.0)
        group.add_argument("--successor-weight", dest="resnet_pva_gad_successor_weight", type=float, default=0.75)
        group.add_argument("--played-return-weight", dest="resnet_pva_gad_played_return_weight", type=float, default=0.50)
        group.add_argument("--policy-weight", dest="resnet_pva_gad_policy_weight", type=float, default=1.0)
        group.add_argument("--value-weight", dest="resnet_pva_gad_value_weight", type=float, default=1.0)
        group.add_argument("--advantage-weight", dest="resnet_pva_gad_advantage_weight", type=float, default=0.50)
        group.add_argument("--kl-weight", dest="resnet_pva_gad_kl_weight", type=float, default=0.05)
        group.add_argument("--entropy-weight", dest="resnet_pva_gad_entropy_weight", type=float, default=0.001)

    @staticmethod
    def td_lambda(args):
        return args.resnet_pva_gad_td_lambda

    @staticmethod
    def counterfactual_topk(args):
        return args.resnet_pva_gad_counterfactual_topk

    @staticmethod
    def counterfactual_min_plies(args):
        return args.resnet_pva_gad_counterfactual_min_plies

    @staticmethod
    def counterfactual_max_plies(args):
        return args.resnet_pva_gad_counterfactual_max_plies

    @staticmethod
    def counterfactual_target_average_plies(args):
        return args.resnet_pva_gad_counterfactual_target_average_plies

    @staticmethod
    def counterfactual_lambda(args):
        return args.resnet_pva_gad_counterfactual_lambda

    @staticmethod
    def candidate_reference_q(record, candidate_index):
        if record.model_advantages is None:
            raise RuntimeError("resnet_pva_gad FCPI record is missing advantages")
        action = int(record.candidate_indices[candidate_index])
        local = np.flatnonzero(record.legal_indices == action)
        if local.size != 1:
            raise RuntimeError(f"candidate action is missing from legal actions: {action}")
        return float(
            np.clip(
                record.root_value + record.model_advantages[int(local[0])],
                -1.0,
                1.0,
            )
        )

    @staticmethod
    def counterfactual_priority(branch, best_q):
        competitiveness = 1.0 - min(1.0, max(0.0, best_q - branch.estimates[-1]) / 2.0)
        return max(branch.last_residual, branch.last_change) + 0.05 * competitiveness

    def behavior_distribution(self, legal_indices, legal_prior, payload, args):
        if payload is None:
            raise RuntimeError("resnet_pva_gad FCPI requires advantage output")
        legal_advantages = np.asarray(payload[legal_indices], dtype=np.float32)
        temperature = max(1e-4, float(args.resnet_pva_gad_behavior_temperature))
        ranking = (
            np.log(np.clip(legal_prior, 1e-12, 1.0))
            + float(args.resnet_pva_gad_behavior_advantage_weight) * legal_advantages
        )
        behavior = stable_softmax(ranking / temperature)
        mix = float(np.clip(args.resnet_pva_gad_uniform_mix, 0.0, 1.0))
        behavior = (1.0 - mix) * behavior + mix / len(behavior)
        return behavior, legal_advantages, ranking

    @staticmethod
    def rollout_move(board, policy, payload, args, profile):
        if payload is None:
            raise RuntimeError("resnet_pva_gad FCPI rollout requires advantage output")
        legal_moves = list(board.legal_moves)
        legal_indices = np.asarray(
            [profile.move_codec.move_to_index(move) for move in legal_moves],
            dtype=np.int64,
        )
        legal_prior = normalize(policy[legal_indices])
        legal_advantage = np.asarray(payload[legal_indices], dtype=np.float32)
        ranking = (
            np.log(np.clip(legal_prior, 1e-12, 1.0))
            + float(args.resnet_pva_gad_behavior_advantage_weight) * legal_advantage
        )
        return legal_moves[int(np.argmax(ranking))]

    def construct_targets(self, records, evaluator, args):
        successor, self.last_target_summary = adaptive_successor_q_batches(
            records,
            evaluator,
            self,
            args,
        )
        print(
            "fcpi counterfactual summary:",
            json.dumps(self.last_target_summary, ensure_ascii=False),
            flush=True,
        )
        successor_weight = float(
            np.clip(args.resnet_pva_gad_successor_weight, 0.0, 1.0)
        )
        played_weight = float(
            np.clip(args.resnet_pva_gad_played_return_weight, 0.0, 1.0)
        )
        temperature = max(1e-4, float(args.resnet_pva_gad_policy_temperature))
        for record, successor_q in zip(records, successor):
            if record.model_advantages is None:
                raise RuntimeError("resnet_pva_gad FCPI record is missing advantages")
            positions = {int(action): index for index, action in enumerate(record.legal_indices)}
            candidate_local = np.asarray(
                [positions[int(action)] for action in record.candidate_indices],
                dtype=np.int64,
            )
            dueling_q = np.clip(
                record.root_value + record.model_advantages[candidate_local],
                -1.0,
                1.0,
            )
            candidate_q = (
                successor_weight * np.asarray(successor_q, dtype=np.float32)
                + (1.0 - successor_weight) * dueling_q
            )
            played = np.flatnonzero(record.candidate_indices == record.played_index)
            if played.size:
                index = int(played[0])
                candidate_q[index] = (
                    (1.0 - played_weight) * candidate_q[index]
                    + played_weight * record.value_target
                )
            q_all = np.clip(
                record.root_value + record.model_advantages,
                -1.0,
                1.0,
            ).astype(np.float32)
            q_all[candidate_local] = candidate_q
            logits = (
                float(args.resnet_pva_gad_prior_power)
                * np.log(np.clip(record.legal_prior, 1e-12, 1.0))
                + (q_all - record.root_value) / temperature
            )
            record.policy_target = stable_softmax(logits).astype(np.float32)
            record.candidate_q = candidate_q.astype(np.float32)
            record.advantage_target = np.clip(
                candidate_q - record.value_target,
                -1.0,
                1.0,
            ).astype(np.float32)

    @staticmethod
    def write_architecture_targets(h5, records, candidate_width):
        h5.create_dataset(
            "advantage_targets",
            data=padded_rows(records, "advantage_target", np.float32, candidate_width),
            compression="lzf",
        )

    def train(self, source_model, data_path, candidate_path, args):
        return train_fcpi_model(self, source_model, data_path, candidate_path, args)

    def batch_loss(self, model, batch, args, device):
        (
            state,
            legal,
            priors,
            targets,
            legal_counts,
            value_target,
            candidate_indices,
            advantage_target,
            candidate_counts,
        ) = [item.to(device) for item in batch]
        heads = model.forward_heads(state)
        policy, kl, entropy = masked_policy_terms(
            heads["policy_logits"], legal, priors, targets, legal_counts
        )
        value = torch.nn.functional.smooth_l1_loss(
            heads["value"].squeeze(1), value_target
        )
        selected_advantage = heads["advantages"].gather(1, candidate_indices)
        width = selected_advantage.shape[1]
        mask = (
            torch.arange(width, device=device).unsqueeze(0)
            < candidate_counts.unsqueeze(1)
        )
        advantage = torch.nn.functional.smooth_l1_loss(
            selected_advantage[mask], advantage_target[mask]
        )
        loss = (
            args.resnet_pva_gad_policy_weight * policy
            + args.resnet_pva_gad_value_weight * value
            + args.resnet_pva_gad_advantage_weight * advantage
            + args.resnet_pva_gad_kl_weight * kl
            - args.resnet_pva_gad_entropy_weight * entropy
        )
        return loss, {
            "policy": policy,
            "value": value,
            "advantage": advantage,
            "kl": kl,
            "entropy": entropy,
        }


EVOLUTIONS = {
    RESNET_PV_LINEAR: ResNetPVLinearFCPI(),
    RESNET_PVA_GAD: ResNetPVAGadFCPI(),
}


def train_fcpi_model(evolution, source_model, data_path, candidate_path, args):
    dataset = evolution.dataset_type(data_path, split=0, arch_type=evolution.arch_type)
    if len(dataset) == 0:
        raise RuntimeError("FCPI training split is empty")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.train_workers > 0,
    )
    model = load_model(source_model, device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and str(args.device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    totals: Dict[str, float] = {}
    steps = 0
    model.train()
    for epoch in range(max(0, int(args.epochs))):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, metrics = evolution.batch_loss(model, batch, args, args.device)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            totals["loss"] = totals.get("loss", 0.0) + float(loss.item())
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value.item())
            if args.log_every > 0 and (steps == 1 or steps % args.log_every == 0):
                print(
                    "fcpi train:",
                    f"step={steps}",
                    *(f"{name}={float(value.item()):.5f}" for name, value in metrics.items()),
                    f"loss={float(loss.item()):.5f}",
                    flush=True,
                )
            if args.train_max_steps > 0 and steps >= args.train_max_steps:
                break
        if args.train_max_steps > 0 and steps >= args.train_max_steps:
            break

    source_epoch, source_step, _ = checkpoint_metadata(source_model, device="cpu")
    save_model(
        candidate_path,
        model,
        epoch=source_epoch + max(0, int(args.epochs)),
        global_step=source_step + steps,
        extra={"type": "fcpi", "formula": evolution.formula_name},
    )
    return {
        "steps": steps,
        "epochs_requested": int(args.epochs),
        "metrics": {name: value / max(1, steps) for name, value in totals.items()},
        "candidate": candidate_path,
    }


@torch.no_grad()
def validate_fcpi_model(evolution, model_path, data_path, args):
    dataset = evolution.dataset_type(data_path, split=1, arch_type=evolution.arch_type)
    if len(dataset) == 0:
        return {"positions": 0, "metrics": {}}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model(model_path, device=args.device)
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for batch in loader:
        loss, metrics = evolution.batch_loss(model, batch, args, args.device)
        totals["loss"] = totals.get("loss", 0.0) + float(loss.item())
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + float(value.item())
        batches += 1
    return {
        "positions": len(dataset),
        "metrics": {name: value / max(1, batches) for name, value in totals.items()},
    }


def validation_metric_delta(candidate_validation, current_validation):
    candidate_metrics = dict(candidate_validation.get("metrics", {}))
    current_metrics = dict(current_validation.get("metrics", {}))
    keys = sorted(set(candidate_metrics) | set(current_metrics))
    return {
        key: float(candidate_metrics.get(key, 0.0) - current_metrics.get(key, 0.0))
        for key in keys
    }


def run(args, evolution):
    paths = prepare_run_paths()
    shutil.copy2(args.model, paths["current_model"])
    print("fcpi run id:", paths["run_id"], flush=True)
    print("fcpi architecture:", evolution.arch_type, flush=True)
    print("fcpi formula:", evolution.formula_name, flush=True)
    print("fcpi current model:", paths["current_model"], flush=True)
    summaries = []
    for iteration in range(1, args.iterations + 1):
        print(f"fcpi iteration {iteration}", flush=True)
        records = collect_selfplay(evolution, paths["current_model"], args, iteration)
        data_path = os.path.join(paths["data_dir"], f"fcpi_iter_{iteration:03d}.h5")
        data_summary = write_base_fcpi_h5(data_path, records, evolution, args)
        candidate_path = os.path.join(
            paths["model_dir"], f"candidate_iter_{iteration:03d}.pth"
        )
        train_summary = evolution.train(
            paths["current_model"], data_path, candidate_path, args
        )
        validation = validate_fcpi_model(evolution, candidate_path, data_path, args)
        current_validation = validate_fcpi_model(
            evolution, paths["current_model"], data_path, args
        )
        delta = validation_metric_delta(validation, current_validation)
        arena = evaluate_models(
            candidate_path=candidate_path,
            baseline_path=paths["current_model"],
            games=args.eval_games,
            sims=args.eval_sims,
            games_in_flight=args.eval_games_in_flight,
            device=args.device,
            max_plies=args.eval_max_plies,
            seed=args.seed + iteration,
            opening_book=args.eval_opening_book,
            book_plies=args.eval_book_plies,
            max_book_positions=args.eval_max_book_positions,
            mcts_batch_size=args.eval_mcts_batch_size,
            movetime_ms=args.eval_movetime_ms,
            search_type=args.eval_search_type,
            c_puct=args.eval_c_puct,
            c_puct_base=args.eval_c_puct_base,
            c_puct_factor=args.eval_c_puct_factor,
            fpu_reduction=args.eval_fpu_reduction,
            progress=True,
        )
        arena = attach_arena_acceptance(arena, args.eval_min_net_wins)
        accepted = bool(arena["accepted"])
        if accepted:
            arena["promotion"] = atomic_copy_with_backup(
                candidate_path, paths["current_model"], make_backup=False
            )
            print("fcpi promoted:", paths["current_model"], flush=True)
        else:
            print("fcpi candidate rejected:", candidate_path, flush=True)
        summary = {
            "iteration": iteration,
            "architecture": evolution.arch_type,
            "formula": evolution.formula_name,
            "data": data_summary,
            "train": train_summary,
            "validation": validation,
            "delta": delta,
            "arena": arena,
            "accepted": accepted,
        }
        summaries.append(summary)
        with open(os.path.join(paths["data_dir"], "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": paths["run_id"],
                    "current_model": paths["current_model"],
                    "summaries": summaries,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    return summaries


def common_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="Architecture-dispatched Folded Counterfactual Policy Iteration",
        add_help=add_help,
    )
    parser.add_argument("--model", default="models/candidate.pth")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iter", type=int, default=200)
    parser.add_argument("--games-in-flight", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--positions-per-game", type=int, default=64)
    parser.add_argument("--opening-book", default="data/openings.gen.bin")
    parser.add_argument("--startpos-fraction", type=float, default=0.20)
    parser.add_argument("--book-plies", type=int, default=8)
    parser.add_argument("--max-book-positions", type=int, default=50000)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--target-records-per-batch", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=0)
    parser.add_argument("--eval-games-in-flight", type=int, default=32)
    parser.add_argument("--eval-max-plies", type=int, default=240)
    parser.add_argument("--eval-opening-book", default="data/openings.gen.bin")
    parser.add_argument("--eval-book-plies", type=int, default=8)
    parser.add_argument("--eval-max-book-positions", type=int, default=50000)
    parser.add_argument("--eval-mcts-batch-size", type=int, default=64)
    parser.add_argument("--eval-movetime-ms", type=int, default=0)
    parser.add_argument("--eval-search-type", choices=("closed", "only-mcts"), default="closed")
    parser.add_argument("--eval-c-puct", type=float, default=1.5)
    parser.add_argument("--eval-c-puct-base", type=float, default=19652.0)
    parser.add_argument("--eval-c-puct-factor", type=float, default=1.0)
    parser.add_argument("--eval-fpu-reduction", type=float, default=0.15)
    parser.add_argument("--eval-min-net-wins", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def parse_args():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model", default="models/candidate.pth")
    probe_args, _ = probe.parse_known_args()
    if not os.path.exists(probe_args.model):
        raise FileNotFoundError(f"model not found: {probe_args.model}")
    probe_model = load_model(probe_args.model, device="cpu")
    arch = probe_model.arch()
    arch_type = arch.get("type") if isinstance(arch, dict) else None
    try:
        evolution = EVOLUTIONS[arch_type]
    except KeyError as exc:
        raise RuntimeError(f"no FCPI evolution registered for {arch_type!r}") from exc
    parser = common_parser()
    evolution.add_arguments(parser)
    args = parser.parse_args()
    args.iterations = max(1, int(args.iterations))
    args.games_per_iter = max(1, int(args.games_per_iter))
    args.max_plies = max(1, int(args.max_plies))
    args.positions_per_game = max(1, int(args.positions_per_game))
    args.startpos_fraction = float(np.clip(args.startpos_fraction, 0.0, 1.0))
    args.validation_fraction = float(np.clip(args.validation_fraction, 0.0, 0.9))
    return args, evolution


def main():
    args, evolution = parse_args()
    run(args, evolution)


if __name__ == "__main__":
    main()
