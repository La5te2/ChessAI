from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from acceptance import attach_arena_acceptance
from arena import evaluate_models, worker_cache_path
from checkpoint_io import atomic_copy_with_backup
from chess_env import board_to_packed, packed_to_tensor
from config import (
    DEVICE,
    H5_PATH,
    MODEL_PATH,
    NUM_ACTIONS,
    STOCKFISH_PATH,
    WEIGHT_DECAY,
)
from data import H5ChessDataset
from model import load_model, save_model
from move_encoder import legal_move_map
from opening_book import OpeningBook, unique_position_fens
from teacher import StockfishTeacher, TeacherConfig


LEGAL_MASK_BYTES = (NUM_ACTIONS + 7) // 8
DEFAULT_DATA_RUNS_DIR = "data/runs"
DEFAULT_MODEL_RUNS_DIR = "models/runs"


def create_run_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"reinforce_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def copy_if_needed(src: str, dst: str, make_backup: bool = True):
    ensure_dir(os.path.dirname(dst) or ".")
    if os.path.abspath(src) == os.path.abspath(dst):
        return {"dst": dst, "backup": None, "skipped": True}
    return atomic_copy_with_backup(src, dst, make_backup=make_backup)


def cp_to_value(cp: Optional[float]) -> float:
    if cp is None:
        return 0.0
    return float(np.tanh(float(cp) / 600.0))


def result_value_for_color(board: chess.Board, color: chess.Color) -> Optional[float]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == color else -1.0


def result_string_from_white_value(value: float, threshold: float = 0.25) -> str:
    if value > float(threshold):
        return "1-0"
    if value < -float(threshold):
        return "0-1"
    return "1/2-1/2"


def final_white_value_and_result(
    board: chess.Board,
    teacher: StockfishTeacher,
    adjudication_threshold: float,
) -> Tuple[float, str]:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0, "1/2-1/2"
        if outcome.winner == chess.WHITE:
            return 1.0, "1-0"
        return -1.0, "0-1"

    result = teacher.analyse(board)
    stm_value = float(result.get("value", 0.0))
    white_value = stm_value if board.turn == chess.WHITE else -stm_value
    return white_value, result_string_from_white_value(
        white_value,
        threshold=adjudication_threshold,
    )


def packed_legal_mask(board: chess.Board, move_map: Dict[int, chess.Move]) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    for index in move_map:
        if 0 <= int(index) < NUM_ACTIONS:
            mask[int(index)] = 1
    packed = np.packbits(mask).astype(np.uint8)
    if packed.shape[0] != LEGAL_MASK_BYTES:
        raise ValueError(f"unexpected legal mask bytes: {packed.shape[0]}")
    return packed


def packed_action_mask(indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    for index in indices:
        if 0 <= int(index) < NUM_ACTIONS:
            mask[int(index)] = 1
    packed = np.packbits(mask).astype(np.uint8)
    if packed.shape[0] != LEGAL_MASK_BYTES:
        raise ValueError(f"unexpected action mask bytes: {packed.shape[0]}")
    return packed


def select_model_move(
    model: torch.nn.Module,
    board: chess.Board,
    device: str,
    temperature: float,
    topk: int,
) -> Optional[Dict]:
    move_map = legal_move_map(board)
    if not move_map:
        return None

    indices = np.asarray(sorted(move_map.keys()), dtype=np.int64)
    state_np = packed_to_tensor(board_to_packed(board))
    state = torch.from_numpy(state_np).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = model(state)
        logits = logits[0].detach().float().cpu()
        value = float(value.squeeze().detach().float().cpu().item())

    legal_logits = logits[torch.from_numpy(indices)]
    sample_logits = legal_logits / max(1e-6, float(temperature))
    if int(topk) > 0 and int(topk) < len(indices):
        top_values, top_offsets = torch.topk(sample_logits, k=int(topk))
        sample_log_probs = F.log_softmax(top_values, dim=0)
        sample_probs = sample_log_probs.exp()
        picked_local = int(torch.multinomial(sample_probs, 1).item())
        picked_offset = int(top_offsets[picked_local].item())
        old_log_prob = float(sample_log_probs[picked_local].item())
        behavior_indices = indices[top_offsets.numpy()]
    else:
        sample_log_probs = F.log_softmax(sample_logits, dim=0)
        sample_probs = sample_log_probs.exp()
        picked_offset = int(torch.multinomial(sample_probs, 1).item())
        old_log_prob = float(sample_log_probs[picked_offset].item())
        behavior_indices = indices

    action = int(indices[picked_offset])
    move = move_map[action]
    return {
        "move": move,
        "action": action,
        "old_log_prob": old_log_prob,
        "old_value": value,
        "policy_prob": float(np.exp(old_log_prob)),
        "state": board_to_packed(board),
        "legal_mask": packed_legal_mask(board, move_map),
        "behavior_mask": packed_action_mask(behavior_indices),
        "behavior_temperature": float(temperature),
    }


def sampling_params_for_position(args: Dict, teacher_result: Optional[Dict]) -> Tuple[float, int]:
    temperature = float(args["sample_temperature"])
    topk = int(args["sample_topk"])
    if not teacher_result:
        return temperature, topk

    margin = float(teacher_result.get("margin_cp", 0.0))
    if margin >= float(args["sharp_gap_cp"]):
        temperature = min(temperature, float(args["sharp_temperature"]))
        sharp_topk = int(args["sharp_topk"])
        if sharp_topk > 0:
            topk = sharp_topk if topk <= 0 else min(topk, sharp_topk)
    return temperature, topk


def teacher_result_for_played_move(
    teacher: StockfishTeacher,
    board: chess.Board,
    move: chess.Move,
    sharp_result: Optional[Dict],
) -> Dict:
    if sharp_result is None:
        return teacher.analyse(board, played_move=move)

    move_scores = sharp_result.get("move_scores_cp") or {}
    uci = move.uci()
    if uci not in move_scores:
        return teacher.analyse(board, played_move=move)

    result = dict(sharp_result)
    best_score = int(result.get("best_score_cp", 0))
    played_score = int(move_scores[uci])
    result["played_score_cp"] = played_score
    result["regret_cp"] = int(max(0, best_score - played_score))
    return result


def step_feedback(args: Dict, result: Dict) -> Dict:
    best_cp = int(result.get("best_score_cp", 0))
    played_cp = int(result.get("played_score_cp", best_cp))
    regret_cp = int(max(0, result.get("regret_cp", best_cp - played_cp)))
    best_value = cp_to_value(best_cp)
    played_value = cp_to_value(played_cp)

    regret_scale = max(1.0, float(args["regret_scale_cp"]))
    regret_norm = min(
        1.0,
        regret_cp / regret_scale,
    )
    accuracy = float(np.exp(-float(regret_cp) / regret_scale))
    quality_reward = float(args["regret_weight"]) * (2.0 * accuracy - 1.0)
    value_delta = float(args["delta_weight"]) * (played_value - best_value)
    blunder_norm = min(
        1.0,
        max(0.0, float(regret_cp) - float(args["blunder_cp"])) / regret_scale,
    )
    blunder_penalty = float(args["blunder_weight"]) * blunder_norm
    reward = quality_reward + value_delta - blunder_penalty
    reward = max(
        -float(args["reward_clip"]),
        min(float(args["reward_clip"]), float(reward)),
    )
    return {
        "teacher_value": float(result.get("value", best_value)),
        "best_value": float(best_value),
        "played_value": float(played_value),
        "regret_cp": float(regret_cp),
        "accuracy": float(accuracy),
        "regret_norm": float(regret_norm),
        "reward": float(reward),
    }


def finalize_episode(args: Dict, episode: List[Dict], final_white_value: float) -> List[Dict]:
    if not episode:
        return []

    values = [float(row["old_value"]) for row in episode]
    rewards = [float(row["reward"]) for row in episode]
    movers = [bool(row["mover_white"]) for row in episode]
    terminal_values = [
        float(final_white_value if mover == chess.WHITE else -final_white_value)
        for mover in movers
    ]

    rewards[-1] += float(args["terminal_weight"]) * terminal_values[-1]

    gamma = float(args["gamma"])
    lam = float(args["gae_lambda"])
    gae = 0.0
    advantages = [0.0 for _ in episode]
    returns = [0.0 for _ in episode]

    for index in range(len(episode) - 1, -1, -1):
        next_value_same_pov = 0.0
        if index + 1 < len(episode):
            next_value_same_pov = -values[index + 1]
        delta = rewards[index] + gamma * next_value_same_pov - values[index]
        gae = delta - gamma * lam * gae
        advantages[index] = float(gae)
        returns[index] = float(values[index] + gae)

    out = []
    for index, row in enumerate(episode):
        item = dict(row)
        item["reward"] = float(rewards[index])
        item["terminal_value"] = float(terminal_values[index])
        item["advantage"] = float(advantages[index])
        item["return"] = float(
            max(-float(args["value_clip"]), min(float(args["value_clip"]), returns[index]))
        )
        out.append(item)
    return out


def rollout_worker(job):
    args, worker_index, worker_count, model_path, game_specs = job
    seed = int(args["seed"]) + 100003 * int(worker_index)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)

    device = args.get("rollout_device") or args.get("device") or "cpu"
    model = load_model(model_path, device=device)
    model.eval()

    teacher_config = TeacherConfig(
        uci=args["uci"],
        depth=int(args["uci_depth"]),
        movetime_ms=int(args["uci_movetime_ms"]),
        multipv=int(args["uci_multipv"]),
        threads=int(args["uci_threads"]),
        hash_mb=int(args["uci_hash_mb"]),
        cache_path=worker_cache_path(
            args["teacher_cache"],
            worker_index,
            worker_count,
        ),
    )

    rows = {
        "states": [],
        "legal_masks": [],
        "behavior_masks": [],
        "behavior_temperatures": [],
        "actions": [],
        "old_log_probs": [],
        "old_values": [],
        "advantages": [],
        "returns": [],
        "teacher_values": [],
        "played_values": [],
        "terminal_values": [],
        "rewards": [],
        "regret_cp": [],
    }
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    total_regret = 0.0
    total_reward = 0.0
    positions = 0
    sharp_positions = 0

    with StockfishTeacher(teacher_config) as teacher:
        total_games = len(game_specs)
        for local_index, (game_number, fen) in enumerate(game_specs, 1):
            board = chess.Board(fen)
            start_ply = board.ply()
            episode = []

            for _ply in range(int(args["max_plies"])):
                if board.is_game_over(claim_draw=True):
                    break

                sharp_result = None
                if bool(args["sharp_check"]):
                    sharp_result = teacher.analyse(board)
                    sharp_positions += 1

                temperature, topk = sampling_params_for_position(args, sharp_result)
                choice = select_model_move(
                    model,
                    board,
                    device=device,
                    temperature=temperature,
                    topk=topk,
                )
                if choice is None:
                    break

                teacher_result = teacher_result_for_played_move(
                    teacher,
                    board,
                    choice["move"],
                    sharp_result,
                )
                feedback = step_feedback(args, teacher_result)

                episode.append({
                    "state": choice["state"],
                    "legal_mask": choice["legal_mask"],
                    "behavior_mask": choice["behavior_mask"],
                    "behavior_temperature": float(choice["behavior_temperature"]),
                    "action": int(choice["action"]),
                    "old_log_prob": float(choice["old_log_prob"]),
                    "old_value": float(choice["old_value"]),
                    "teacher_value": float(feedback["teacher_value"]),
                    "played_value": float(feedback["played_value"]),
                    "regret_cp": float(feedback["regret_cp"]),
                    "reward": float(feedback["reward"]),
                    "mover_white": bool(board.turn == chess.WHITE),
                })
                board.push(choice["move"])

            final_white_value, result = final_white_value_and_result(
                board,
                teacher,
                adjudication_threshold=float(args["adjudication_threshold"]),
            )
            finalized = finalize_episode(args, episode, final_white_value)
            for row in finalized:
                rows["states"].append(row["state"])
                rows["legal_masks"].append(row["legal_mask"])
                rows["behavior_masks"].append(row["behavior_mask"])
                rows["behavior_temperatures"].append(float(row["behavior_temperature"]))
                rows["actions"].append(int(row["action"]))
                rows["old_log_probs"].append(float(row["old_log_prob"]))
                rows["old_values"].append(float(row["old_value"]))
                rows["advantages"].append(float(row["advantage"]))
                rows["returns"].append(float(row["return"]))
                rows["teacher_values"].append(float(row["teacher_value"]))
                rows["played_values"].append(float(row["played_value"]))
                rows["terminal_values"].append(float(row["terminal_value"]))
                rows["rewards"].append(float(row["reward"]))
                rows["regret_cp"].append(float(row["regret_cp"]))
                total_regret += float(row["regret_cp"])
                total_reward += float(row["reward"])
            positions += len(finalized)
            results[result] = results.get(result, 0) + 1

            if bool(args["progress"]):
                mean_regret = total_regret / max(1, positions)
                print(
                    "reinforce game",
                    f"{game_number}/{args['games_per_iter']}:",
                    f"worker={worker_index}",
                    f"start_ply={start_ply}",
                    f"plies={len(finalized)}",
                    f"mean_regret_cp={mean_regret:.1f}",
                    f"result={result}",
                    flush=True,
                )

    return {
        "rows": rows,
        "summary": {
            "worker": worker_index,
            "games": len(game_specs),
            "positions": positions,
            "results": results,
            "mean_regret_cp": total_regret / max(1, positions),
            "mean_reward": total_reward / max(1, positions),
            "sharp_positions": sharp_positions,
        },
    }


def load_openings(args) -> List[str]:
    if not args.opening_book:
        return [chess.Board().fen() for _ in range(int(args.games_per_iter))]

    book = OpeningBook(
        path=args.opening_book,
        book_plies=args.book_plies,
        max_positions=max(int(args.max_book_positions), int(args.games_per_iter)),
        seed=args.seed,
    )
    fens = unique_position_fens(list(book.positions))
    if len(fens) < int(args.games_per_iter):
        raise ValueError(
            "reinforce requires enough unique opening states: "
            f"games_per_iter={args.games_per_iter}, unique_openings={len(fens)}"
        )
    rng = random.Random(int(args.seed))
    rng.shuffle(fens)
    return fens[:int(args.games_per_iter)]


def merge_rollout_outputs(outputs: List[Dict]) -> Tuple[Dict, Dict]:
    rows = {
        "states": [],
        "legal_masks": [],
        "behavior_masks": [],
        "behavior_temperatures": [],
        "actions": [],
        "old_log_probs": [],
        "old_values": [],
        "advantages": [],
        "returns": [],
        "teacher_values": [],
        "played_values": [],
        "terminal_values": [],
        "rewards": [],
        "regret_cp": [],
    }
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    summaries = []
    for output in outputs:
        summaries.append(output["summary"])
        for key in rows:
            rows[key].extend(output["rows"][key])
        for result, count in output["summary"]["results"].items():
            results[result] = results.get(result, 0) + int(count)

    positions = len(rows["actions"])
    regret_array = np.asarray(rows["regret_cp"], dtype=np.float32)
    summary = {
        "games": int(sum(item["games"] for item in summaries)),
        "positions": int(positions),
        "results": results,
        "mean_regret_cp": float(np.mean(regret_array)) if positions else 0.0,
        "excellent_rate": float(np.mean(regret_array <= 30.0)) if positions else 0.0,
        "inaccuracy_rate": float(np.mean(regret_array > 50.0)) if positions else 0.0,
        "mistake_rate": float(np.mean(regret_array > 150.0)) if positions else 0.0,
        "blunder_rate": float(np.mean(regret_array > 300.0)) if positions else 0.0,
        "mean_reward": float(np.mean(rows["rewards"])) if positions else 0.0,
        "mean_advantage": float(np.mean(rows["advantages"])) if positions else 0.0,
        "std_advantage": float(np.std(rows["advantages"])) if positions else 0.0,
        "sharp_positions": int(sum(item["sharp_positions"] for item in summaries)),
        "workers": summaries,
    }
    return rows, summary


def write_rollout_h5(path: str, rows: Dict, summary: Dict, args):
    ensure_dir(os.path.dirname(path) or ".")
    count = len(rows["actions"])
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "states",
            data=np.asarray(rows["states"], dtype=np.uint8).reshape(count, 18, 8),
            compression="gzip",
            compression_opts=3,
        )
        h5.create_dataset(
            "legal_masks",
            data=np.asarray(rows["legal_masks"], dtype=np.uint8).reshape(
                count,
                LEGAL_MASK_BYTES,
            ),
            compression="gzip",
            compression_opts=3,
        )
        h5.create_dataset(
            "behavior_masks",
            data=np.asarray(rows["behavior_masks"], dtype=np.uint8).reshape(
                count,
                LEGAL_MASK_BYTES,
            ),
            compression="gzip",
            compression_opts=3,
        )
        for key, dtype in [
            ("behavior_temperatures", np.float32),
            ("actions", np.int64),
            ("old_log_probs", np.float32),
            ("old_values", np.float32),
            ("advantages", np.float32),
            ("returns", np.float32),
            ("teacher_values", np.float32),
            ("played_values", np.float32),
            ("terminal_values", np.float32),
            ("rewards", np.float32),
            ("regret_cp", np.float32),
        ]:
            h5.create_dataset(key, data=np.asarray(rows[key], dtype=dtype))
        h5.attrs["summary_json"] = json.dumps(summary, ensure_ascii=False)
        h5.attrs["run_id"] = args.run_id
        h5.attrs["generator"] = "reinforce"


def read_rollout_summary(path: str) -> Dict:
    with h5py.File(path, "r") as h5:
        raw = h5.attrs.get("summary_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw:
            return json.loads(raw)
        return {
            "games": None,
            "positions": int(h5["actions"].shape[0]),
            "results": {},
        }


def generate_rollout(args, model_path: str, output_path: str) -> Dict:
    print(
        "reinforce rollout start:",
        f"model={model_path}",
        f"games={args.games_per_iter}",
        f"parallel={args.parallel}",
        f"device={args.rollout_device or args.device}",
        f"teacher_depth={args.uci_depth}",
        f"teacher_multipv={args.uci_multipv}",
        flush=True,
    )
    fens = load_openings(args)
    print(
        "reinforce openings:",
        f"path={args.opening_book or 'startpos'}",
        f"positions={len(fens)}",
        f"book_plies={args.book_plies}",
        flush=True,
    )

    specs = [(index + 1, fen) for index, fen in enumerate(fens)]
    workers = max(1, min(int(args.parallel), len(specs)))
    splits = [[] for _ in range(workers)]
    for index, spec in enumerate(specs):
        splits[index % workers].append(spec)

    args_dict = vars(args).copy()
    args_dict["progress"] = True
    jobs = [
        (args_dict, worker_index + 1, workers, model_path, chunk)
        for worker_index, chunk in enumerate(splits)
        if chunk
    ]
    if len(jobs) == 1:
        outputs = [rollout_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            outputs = pool.map(rollout_worker, jobs)

    rows, summary = merge_rollout_outputs(outputs)
    write_rollout_h5(output_path, rows, summary, args)
    print("reinforce rollout summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


class RolloutDataset(Dataset):
    def __init__(self, path: str, normalize_advantages: bool = True, advantage_clip: float = 5.0):
        with h5py.File(path, "r") as h5:
            self.states = np.asarray(h5["states"], dtype=np.uint8)
            self.legal_masks = np.asarray(h5["legal_masks"], dtype=np.uint8)
            if "behavior_masks" in h5:
                self.behavior_masks = np.asarray(h5["behavior_masks"], dtype=np.uint8)
            else:
                self.behavior_masks = np.asarray(h5["legal_masks"], dtype=np.uint8)
            if "behavior_temperatures" in h5:
                self.behavior_temperatures = np.asarray(
                    h5["behavior_temperatures"],
                    dtype=np.float32,
                )
            else:
                self.behavior_temperatures = np.ones(
                    int(h5["actions"].shape[0]),
                    dtype=np.float32,
                )
            self.actions = np.asarray(h5["actions"], dtype=np.int64)
            self.old_log_probs = np.asarray(h5["old_log_probs"], dtype=np.float32)
            self.advantages = np.asarray(h5["advantages"], dtype=np.float32)
            self.returns = np.asarray(h5["returns"], dtype=np.float32)
            self.teacher_values = np.asarray(h5["teacher_values"], dtype=np.float32)

        if normalize_advantages and len(self.advantages):
            mean = float(np.mean(self.advantages))
            std = float(np.std(self.advantages))
            if std > 1e-6:
                self.advantages = (self.advantages - mean) / std
        if advantage_clip and advantage_clip > 0:
            self.advantages = np.clip(
                self.advantages,
                -float(advantage_clip),
                float(advantage_clip),
            ).astype(np.float32)

    def __len__(self):
        return int(self.actions.shape[0])

    def __getitem__(self, index):
        legal = np.unpackbits(self.legal_masks[index])[:NUM_ACTIONS].astype(np.bool_)
        behavior = np.unpackbits(self.behavior_masks[index])[:NUM_ACTIONS].astype(np.bool_)
        return (
            torch.from_numpy(packed_to_tensor(self.states[index])),
            torch.tensor(int(self.actions[index]), dtype=torch.long),
            torch.tensor(float(self.old_log_probs[index]), dtype=torch.float32),
            torch.tensor(float(self.advantages[index]), dtype=torch.float32),
            torch.tensor(float(self.returns[index]), dtype=torch.float32),
            torch.tensor(float(self.teacher_values[index]), dtype=torch.float32),
            torch.from_numpy(legal),
            torch.from_numpy(behavior),
            torch.tensor(float(self.behavior_temperatures[index]), dtype=torch.float32),
        )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def masked_log_probs(logits, legal_masks, temperature: float = 1.0):
    logits = logits.float()
    if torch.is_tensor(temperature):
        temperature = temperature.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-6)
        if temperature.ndim == 1:
            temperature = temperature.view(-1, 1)
        scaled = logits / temperature
    else:
        scaled = logits / max(1e-6, float(temperature))
    legal_masks = legal_masks.bool()
    masked = scaled.masked_fill(~legal_masks, -1e9)
    return F.log_softmax(masked, dim=1)


def masked_entropy(log_probs, legal_masks):
    probs = log_probs.exp()
    entropy = -(probs * log_probs).masked_fill(~legal_masks, 0.0).sum(dim=1)
    return entropy.mean()


def masked_reference_kl(student_logits, reference_logits, legal_masks, temperature: float):
    student_logp = masked_log_probs(student_logits, legal_masks, temperature)
    with torch.no_grad():
        reference_logp = masked_log_probs(reference_logits, legal_masks, temperature)
        reference_p = reference_logp.exp()
    return (reference_p * (reference_logp - student_logp)).masked_fill(
        ~legal_masks,
        0.0,
    ).sum(dim=1).mean() * (max(1e-6, float(temperature)) ** 2)


def supervised_loss_from_batch(model, batch, args):
    states, target, values = batch
    states = states.to(args.device, non_blocking=True)
    target = target.to(args.device, non_blocking=True)
    values = values.to(args.device, non_blocking=True)
    logits, predicted_values = model(states)
    if target.ndim == 2:
        policy_loss = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    else:
        policy_loss = F.cross_entropy(logits, target.long())
    value_loss = F.mse_loss(predicted_values.squeeze(1), values)
    return policy_loss + float(args.supervised_value_weight) * value_loss


def train_actor_critic(args, source_model: str, rollout_path: str, candidate_path: str) -> Dict:
    dataset = RolloutDataset(
        rollout_path,
        normalize_advantages=True,
        advantage_clip=args.advantage_clip,
    )
    if len(dataset) == 0:
        raise RuntimeError("rollout produced no positions")

    pin_memory = str(args.device).startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=pin_memory,
        persistent_workers=args.train_workers > 0,
    )

    supervised_iter = None
    if args.supervised_data and os.path.exists(args.supervised_data) and args.supervised_weight > 0:
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
        "reinforce train start:",
        f"source={source_model}",
        f"candidate={candidate_path}",
        f"positions={len(dataset)}",
        f"ppo_epochs={args.ppo_epochs}",
        f"max_steps={args.train_max_steps}",
        f"critic_target={args.critic_target}",
        flush=True,
    )

    student = load_model(source_model, device=args.device)
    student.train()
    reference = load_model(source_model, device=args.device)
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

    step = 0
    stop = False
    last_metrics = {}
    for epoch in range(int(args.ppo_epochs)):
        for batch in loader:
            (
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                teacher_values,
                legal_masks,
                behavior_masks,
                behavior_temperatures,
            ) = batch
            states = states.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)
            old_log_probs = old_log_probs.to(args.device, non_blocking=True)
            advantages = advantages.to(args.device, non_blocking=True)
            returns = returns.to(args.device, non_blocking=True)
            teacher_values = teacher_values.to(args.device, non_blocking=True)
            legal_masks = legal_masks.to(args.device, non_blocking=True)
            behavior_masks = behavior_masks.to(args.device, non_blocking=True)
            behavior_temperatures = behavior_temperatures.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, values = student(states)
                values = values.squeeze(1)
                log_probs = masked_log_probs(logits, legal_masks)
                behavior_log_probs = masked_log_probs(
                    logits,
                    behavior_masks,
                    behavior_temperatures,
                )
                action_log_probs = behavior_log_probs.gather(
                    1,
                    actions.unsqueeze(1),
                ).squeeze(1)
                ratio = torch.exp(action_log_probs - old_log_probs)
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(args.ppo_clip),
                    1.0 + float(args.ppo_clip),
                ) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                target_values = returns
                if args.critic_target == "teacher":
                    target_values = teacher_values
                value_loss = F.smooth_l1_loss(values, target_values)

                entropy = masked_entropy(log_probs, legal_masks)
                with torch.no_grad():
                    reference_logits, _ = reference(states)
                kl_loss = masked_reference_kl(
                    logits,
                    reference_logits,
                    legal_masks,
                    temperature=args.kl_temperature,
                )

                supervised_loss = torch.zeros((), device=args.device)
                if supervised_iter is not None:
                    supervised_loss = supervised_loss_from_batch(
                        student,
                        next(supervised_iter),
                        args,
                    )

                loss = (
                    policy_loss
                    + float(args.value_weight) * value_loss
                    - float(args.entropy_weight) * entropy
                    + float(args.kl_weight) * kl_loss
                    + float(args.supervised_weight) * supervised_loss
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            last_metrics = {
                "step": int(step),
                "epoch": int(epoch),
                "policy": float(policy_loss.item()),
                "value": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "kl": float(kl_loss.item()),
                "supervised": float(supervised_loss.item()),
                "loss": float(loss.item()),
            }
            if (
                args.log_every > 0
                and (step == 1 or step % int(args.log_every) == 0)
            ):
                print(
                    "reinforce train step:",
                    f"epoch={epoch}",
                    f"step={step}",
                    f"policy={last_metrics['policy']:.4f}",
                    f"value={last_metrics['value']:.4f}",
                    f"entropy={last_metrics['entropy']:.4f}",
                    f"kl={last_metrics['kl']:.4f}",
                    f"supervised={last_metrics['supervised']:.4f}",
                    f"loss={last_metrics['loss']:.4f}",
                    flush=True,
                )

            if args.train_max_steps and step >= int(args.train_max_steps):
                stop = True
                break
        if stop:
            break

    save_model(
        candidate_path,
        student,
        epoch=max(0, int(args.ppo_epochs) - 1),
        global_step=step,
        extra={
            "type": "teacher_shaped_actor_critic",
            "source_model": source_model,
            "rollout": rollout_path,
            "critic_target": args.critic_target,
            "ppo_epochs": int(args.ppo_epochs),
            "run_id": args.run_id,
        },
    )
    print(
        "reinforce candidate saved:",
        f"path={candidate_path}",
        f"steps={step}",
        flush=True,
    )
    return last_metrics


def evaluate_candidate(args, candidate_path: str, baseline_path: str) -> Dict:
    if int(args.eval_games) <= 0:
        return {"accepted": False, "skipped": True}

    metrics = evaluate_models(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        games=args.eval_games,
        sims=args.eval_sims,
        workers=args.eval_workers or args.parallel,
        device=args.device,
        max_plies=args.eval_max_plies,
        seed=args.seed,
        opening_book=args.eval_opening_book or args.opening_book,
        book_plies=args.eval_book_plies,
        max_book_positions=args.eval_max_book_positions,
        mcts_batch_size=args.eval_mcts_batch_size,
        movetime_ms=args.eval_movetime_ms,
        c_puct=args.eval_c_puct,
        mate_guard_plies=args.eval_mate_guard_plies,
        mate_guard_topk=args.eval_mate_guard_topk,
        mate_guard_nodes=args.eval_mate_guard_nodes,
        mate_guard_time_fraction=args.eval_mate_guard_time_fraction,
        q_tiebreak=args.eval_q_tiebreak,
        q_tiebreak_p_ratio=args.eval_q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio=args.eval_q_tiebreak_visit_ratio,
        q_tiebreak_margin=args.eval_q_tiebreak_margin,
        uci=args.uci,
        uci_depth=args.eval_uci_depth,
        uci_movetime_ms=args.eval_uci_movetime_ms,
        uci_threads=args.uci_threads,
        uci_hash_mb=args.uci_hash_mb,
        uci_multipv=args.eval_uci_multipv,
        teacher_cache=os.path.join(args.data_run_dir, "arena_teacher_cache.sqlite"),
        quality_loss_cap_cp=args.eval_quality_loss_cap_cp,
        log_every=args.log_every,
        progress=True,
    )
    metrics = attach_arena_acceptance(
        metrics,
        min_net_wins=args.eval_min_net_wins,
        min_acpl_improvement=args.eval_min_acpl_improvement,
        min_accuracy_improvement=args.eval_min_accuracy_improvement,
    )
    print("reinforce arena summary:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def run(args):
    if args.run_id is None:
        args.run_id = create_run_id()
    args.data_run_dir = ensure_dir(os.path.join(args.data_runs_dir, args.run_id))
    args.model_run_dir = ensure_dir(os.path.join(args.model_runs_dir, args.run_id))
    if args.teacher_cache is None:
        args.teacher_cache = os.path.join(args.data_run_dir, "teacher_cache.sqlite")

    current_model = os.path.join(args.model_run_dir, "current.pth")
    if not os.path.exists(current_model):
        copy_if_needed(args.model, current_model, make_backup=False)

    print(
        "reinforce start:",
        f"run_id={args.run_id}",
        f"current={current_model}",
        f"data_run_dir={args.data_run_dir}",
        f"model_run_dir={args.model_run_dir}",
        flush=True,
    )

    all_summaries = []
    for iteration in range(1, int(args.iterations) + 1):
        summary_path = os.path.join(args.data_run_dir, f"summary_iter_{iteration:03d}.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            all_summaries.append(summary)
            print(
                "reinforce iteration skip:",
                f"iteration={iteration}",
                f"summary={summary_path}",
                f"accepted={summary.get('accepted')}",
                flush=True,
            )
            continue

        print(f"reinforce iteration {iteration}", flush=True)
        rollout_path = os.path.join(
            args.data_run_dir,
            f"rollout_iter_{iteration:03d}.h5",
        )
        candidate_path = os.path.join(
            args.model_run_dir,
            f"candidate_iter_{iteration:03d}.pth",
        )
        if os.path.exists(rollout_path) and args.reuse_rollout:
            rollout_summary = read_rollout_summary(rollout_path)
            print(
                "reinforce rollout reuse:",
                f"path={rollout_path}",
                f"positions={rollout_summary.get('positions')}",
                flush=True,
            )
        else:
            rollout_summary = generate_rollout(args, current_model, rollout_path)
        train_summary = train_actor_critic(
            args,
            source_model=current_model,
            rollout_path=rollout_path,
            candidate_path=candidate_path,
        )
        arena_summary = evaluate_candidate(args, candidate_path, current_model)
        accepted = bool(arena_summary.get("accepted"))
        promotion_summary = None
        if accepted and args.promote_if_accepted:
            promotion_summary = copy_if_needed(
                candidate_path,
                current_model,
                make_backup=True,
            )
            print(
                "reinforce candidate accepted:",
                current_model,
                f"backup={promotion_summary.get('backup')}",
                flush=True,
            )
        else:
            print("reinforce candidate rejected:", candidate_path, flush=True)

        summary = {
            "iteration": int(iteration),
            "rollout": rollout_summary,
            "train": train_summary,
            "arena": arena_summary,
            "accepted": accepted,
            "current_model": current_model,
            "candidate_model": candidate_path,
            "rollout_path": rollout_path,
            "promotion": promotion_summary,
        }
        all_summaries.append(summary)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(args.data_run_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print("reinforce finished:", args.run_id, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Teacher-shaped actor-critic experiment. The teacher evaluates moves but never plays them.",
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--supervised-data", default=H5_PATH)
    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--rollout-device", default=None)
    parser.add_argument("--data-runs-dir", default=DEFAULT_DATA_RUNS_DIR)
    parser.add_argument("--model-runs-dir", default=DEFAULT_MODEL_RUNS_DIR)
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-iter", type=int, default=200)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--adjudication-threshold", type=float, default=0.25)
    parser.add_argument("--opening-book", default="")
    parser.add_argument("--book-plies", type=int, default=8)
    parser.add_argument("--max-book-positions", type=int, default=50000)
    parser.add_argument("--reuse-rollout", action="store_true", default=True)
    parser.add_argument("--no-reuse-rollout", dest="reuse_rollout", action="store_false")

    parser.add_argument("--sample-temperature", type=float, default=0.5)
    parser.add_argument("--sample-topk", type=int, default=8)
    parser.add_argument("--sharp-check", action="store_true", default=True)
    parser.add_argument("--no-sharp-check", dest="sharp_check", action="store_false")
    parser.add_argument("--sharp-gap-cp", type=int, default=180)
    parser.add_argument("--sharp-temperature", type=float, default=0.15)
    parser.add_argument("--sharp-topk", type=int, default=1)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--delta-weight", type=float, default=0.35)
    parser.add_argument("--regret-weight", type=float, default=0.70)
    parser.add_argument("--regret-scale-cp", type=float, default=250.0)
    parser.add_argument("--blunder-cp", type=float, default=300.0)
    parser.add_argument("--blunder-weight", type=float, default=0.60)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--reward-clip", type=float, default=2.0)
    parser.add_argument("--value-clip", type=float, default=2.0)

    parser.add_argument("--uci-depth", type=int, default=8)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-multipv", type=int, default=6)
    parser.add_argument("--uci-threads", type=int, default=1)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--teacher-cache", default=None)

    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--train-max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--ppo-clip", type=float, default=0.20)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--critic-target", choices=("teacher", "returns"), default="teacher")
    parser.add_argument("--value-weight", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=0.10)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--supervised-weight", type=float, default=0.35)
    parser.add_argument("--supervised-value-weight", type=float, default=0.25)

    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=64)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--eval-max-plies", type=int, default=180)
    parser.add_argument("--eval-opening-book", default="data/openings.gen.bin")
    parser.add_argument("--eval-book-plies", type=int, default=8)
    parser.add_argument("--eval-max-book-positions", type=int, default=50000)
    parser.add_argument("--eval-mcts-batch-size", type=int, default=64)
    parser.add_argument("--eval-movetime-ms", type=int, default=1000)
    parser.add_argument("--eval-c-puct", type=float, default=0.5)
    parser.add_argument("--eval-mate-guard-plies", type=int, default=3)
    parser.add_argument("--eval-mate-guard-topk", type=int, default=8)
    parser.add_argument("--eval-mate-guard-nodes", type=int, default=20000)
    parser.add_argument("--eval-mate-guard-time-fraction", type=float, default=0.10)
    parser.add_argument("--eval-q-tiebreak", action="store_true", default=True)
    parser.add_argument("--no-eval-q-tiebreak", dest="eval_q_tiebreak", action="store_false")
    parser.add_argument("--eval-q-tiebreak-p-ratio", type=float, default=0.70)
    parser.add_argument("--eval-q-tiebreak-visit-ratio", type=float, default=0.70)
    parser.add_argument("--eval-q-tiebreak-margin", type=float, default=0.02)
    parser.add_argument("--eval-uci-depth", type=int, default=10)
    parser.add_argument("--eval-uci-movetime-ms", type=int, default=0)
    parser.add_argument("--eval-uci-multipv", type=int, default=6)
    parser.add_argument("--eval-quality-loss-cap-cp", type=int, default=1000)
    parser.add_argument("--eval-min-net-wins", type=int, default=5)
    parser.add_argument("--eval-min-acpl-improvement", type=float, default=0.0)
    parser.add_argument("--eval-min-accuracy-improvement", type=float, default=0.0)

    parser.add_argument("--promote-if-accepted", action="store_true", default=True)
    parser.add_argument("--no-promote-if-accepted", dest="promote_if_accepted", action="store_false")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
