"""Offline Stockfish-rewarded actor-critic training.

This entry point keeps the reinforce run/gate shape and performs offline FEN
labeling:

1. Read positions sequentially from a PGN or an HDF5 file with a `fens` dataset.
2. Let the current model propose policy top-k legal moves at sim=0.
3. Ask the UCI teacher to score those candidates.
4. Convert candidate scores to continuous rewards and train the actor from
   reward-minus-value advantages while the critic tracks expected reward.
5. Promote only after the same arena gate accepts the candidate.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import shutil
import time
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import chess.pgn
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
    MODEL_DIR,
    MODEL_PATH,
    NUM_ACTIONS,
    PGN_PATH,
    STOCKFISH_PATH,
    WEIGHT_DECAY,
)
from data import H5ChessDataset
from model import load_model, save_model
from move_encoder import move_to_index
from teacher import MATE_SCORE_CP, StockfishTeacher, TeacherConfig


DEFAULT_DATA_RUNS_DIR = os.path.join("data", "runs")
DEFAULT_MODEL_RUNS_DIR = os.path.join(MODEL_DIR, "runs")


def create_run_id() -> str:
    return time.strftime("reinforce_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]


def normalize_run_id(run_id: str) -> str:
    cleaned = str(run_id).strip().replace("\\", "/").split("/")[-1]
    if not cleaned:
        raise ValueError("empty run id")
    return cleaned


def make_run_dirs(data_root: str, model_root: str, run_id: str) -> Tuple[str, str]:
    data_dir = os.path.join(data_root, run_id)
    model_dir = os.path.join(model_root, run_id)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return data_dir, model_dir


def prepare_run_paths(args):
    if args.run_id:
        run_id = normalize_run_id(args.run_id)
    else:
        run_id = create_run_id()
    data_dir, model_dir = make_run_dirs(args.data_runs_dir, args.model_runs_dir, run_id)
    if args.teacher_cache is None:
        args.teacher_cache = os.path.join(data_dir, "teacher_cache.sqlite")
    return {
        "run_id": run_id,
        "data_run_dir": data_dir,
        "model_run_dir": model_dir,
        "current_model": os.path.join(model_dir, "current.pth"),
    }


def atomic_copy(src: str, dst: str, make_backup: bool = True):
    return atomic_copy_with_backup(src, dst, make_backup=make_backup)


def decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def iter_h5_fens(path: str, offset: int, limit: int) -> Iterable[str]:
    with h5py.File(path, "r") as h5:
        if "fens" not in h5:
            raise ValueError(
                f"{path} has no 'fens' dataset; use --fen-source data/games.pgn "
                "or regenerate HDF5 with FEN storage."
            )
        fens = h5["fens"]
        end = len(fens) if limit <= 0 else min(len(fens), offset + limit)
        for index in range(max(0, offset), end):
            fen = decode_h5_string(fens[index]).strip()
            if fen:
                yield fen


def iter_pgn_fens(
    path: str,
    offset: int,
    limit: int,
    min_ply: int,
    max_ply: int,
    max_games: Optional[int],
) -> Iterable[str]:
    produced = 0
    seen_positions = 0
    games = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        while True:
            if max_games is not None and games >= max_games:
                break
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games += 1
            board = game.board()
            for move in game.mainline_moves():
                ply = int(board.ply())
                if ply >= int(min_ply) and (max_ply <= 0 or ply <= int(max_ply)):
                    if seen_positions >= offset:
                        yield board.fen()
                        produced += 1
                        if limit > 0 and produced >= limit:
                            return
                    seen_positions += 1
                board.push(move)


def load_fens(args, iteration: int) -> List[str]:
    source = str(args.fen_source or "").strip()
    if not source:
        source = PGN_PATH
    offset = int(args.source_offset) + max(0, int(iteration) - 1) * int(args.positions_per_iter)
    limit = int(args.positions_per_iter)

    if source.lower() == "startpos":
        return [chess.Board().fen() for _ in range(limit)]
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"fen source not found: {source}. Pass --fen-source <games.pgn|fens.h5>."
        )

    lower = source.lower()
    if lower.endswith((".h5", ".hdf5")):
        fens = list(iter_h5_fens(source, offset=offset, limit=limit))
    else:
        fens = list(
            iter_pgn_fens(
                source,
                offset=offset,
                limit=limit,
                min_ply=args.source_min_ply,
                max_ply=args.source_max_ply,
                max_games=args.source_max_games,
            )
        )

    if len(fens) < limit:
        raise RuntimeError(
            f"fen source exhausted: requested={limit}, got={len(fens)}, "
            f"source={source}, offset={offset}"
        )
    return fens


def legal_policy_candidates(
    model,
    board: chess.Board,
    device: str,
    topk: int,
) -> Tuple[List[chess.Move], np.ndarray, float]:
    legal_pairs = []
    for move in board.legal_moves:
        try:
            legal_pairs.append((move_to_index(move), move))
        except Exception:
            continue
    if not legal_pairs:
        return [], np.zeros(NUM_ACTIONS, dtype=np.float32), 0.0

    state = torch.from_numpy(packed_to_tensor(board_to_packed(board))).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(state)
        logits = logits[0].detach().float().cpu()
        model_value = float(value.squeeze().detach().float().cpu().item())

    indices = np.asarray([index for index, _move in legal_pairs], dtype=np.int64)
    legal_logits = logits[torch.from_numpy(indices)]
    legal_probs = F.softmax(legal_logits, dim=0).numpy().astype(np.float32)

    dense_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for index, probability in zip(indices, legal_probs):
        dense_policy[int(index)] = float(probability)

    order = np.argsort(-legal_probs)
    limit = len(order) if topk <= 0 else min(int(topk), len(order))
    moves = [legal_pairs[int(offset)][1] for offset in order[:limit]]
    return moves, dense_policy, model_value


def reward_from_cp(cp: float, scale_cp: float) -> float:
    cp = float(cp)
    if abs(cp) >= MATE_SCORE_CP * 0.5:
        return 1.0 if cp > 0 else -1.0
    return float(np.tanh(cp / max(1.0, float(scale_cp))))


def rl_targets_from_scores(
    board: chess.Board,
    move_scores: Dict[str, int],
    candidate_moves: Sequence[chess.Move],
    model_policy: np.ndarray,
    reward_scale_cp: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    candidate_mask = np.zeros(NUM_ACTIONS, dtype=np.uint8)
    action_rewards = np.zeros(NUM_ACTIONS, dtype=np.float32)
    rows = []
    legal = set(board.legal_moves)
    usable = []
    for move in candidate_moves:
        if move not in legal:
            continue
        uci = move.uci()
        if uci not in move_scores:
            continue
        usable.append((move, int(move_scores[uci])))

    if not usable:
        return candidate_mask, action_rewards, 0.0, 0.0

    best_score = max(score for _move, score in usable)
    for move, score in usable:
        regret = max(0, int(best_score - score))
        reward = reward_from_cp(score, reward_scale_cp)
        index = move_to_index(move)
        candidate_mask[index] = 1
        action_rewards[index] = float(reward)
        rows.append({
            "move": move.uci(),
            "score_cp": int(score),
            "regret_cp": int(regret),
            "reward": float(reward),
            "model_probability": float(model_policy[index]),
        })

    model_top1 = candidate_moves[0].uci() if candidate_moves else None
    model_top1_regret = next(
        (
            float(row["regret_cp"])
            for row in rows
            if row["move"] == model_top1
        ),
        0.0,
    )
    max_regret = float(max(row["regret_cp"] for row in rows) if rows else 0.0)
    return candidate_mask, action_rewards, max_regret, model_top1_regret


def label_worker(job):
    args_dict, worker_index, worker_count, model_path, specs = job
    device = args_dict.get("label_device") or args_dict.get("device") or "cpu"
    model = load_model(model_path, device=device)
    model.eval()

    teacher_config = TeacherConfig(
        uci=args_dict["uci"],
        depth=int(args_dict["uci_depth"]),
        movetime_ms=int(args_dict["uci_movetime_ms"]),
        multipv=int(args_dict["uci_multipv"]),
        threads=int(args_dict["uci_threads"]),
        hash_mb=int(args_dict["uci_hash_mb"]),
        cache_path=worker_cache_path(
            args_dict.get("teacher_cache"),
            worker_index,
            worker_count,
        ),
    )

    rows = []
    total_regret = 0.0
    total_top1_regret = 0.0
    labeled = 0
    with StockfishTeacher(teacher_config) as teacher:
        for global_index, fen in specs:
            board = chess.Board(fen)
            candidates, model_policy, model_value = legal_policy_candidates(
                model,
                board,
                device=device,
                topk=int(args_dict["sample_topk"]),
            )
            if not candidates:
                continue

            teacher_result = teacher.analyse_candidates(board, candidates)
            if bool(args_dict["include_teacher_best"]):
                try:
                    teacher_move = chess.Move.from_uci(str(teacher_result.get("best_move")))
                    if teacher_move in board.legal_moves and teacher_move not in candidates:
                        candidates.append(teacher_move)
                        teacher_result = teacher.analyse_candidates(board, candidates)
                except Exception:
                    pass

            move_scores = {
                str(move): int(score)
                for move, score in (teacher_result.get("move_scores_cp") or {}).items()
            }
            (
                candidate_mask,
                action_rewards,
                max_regret,
                model_top1_regret,
            ) = rl_targets_from_scores(
                board,
                move_scores,
                candidates,
                model_policy=model_policy,
                reward_scale_cp=float(args_dict["reward_scale_cp"]),
            )
            if int(candidate_mask.sum()) <= 0:
                continue

            best_score = int(teacher_result.get("best_score_cp", 0))
            rows.append({
                "state": board_to_packed(board),
                "candidate_mask": candidate_mask,
                "action_rewards": action_rewards,
                "model_policy": model_policy,
                "teacher_value": float(
                    teacher_result.get(
                        "value",
                        reward_from_cp(best_score, args_dict["reward_scale_cp"]),
                    )
                ),
                "model_value": float(model_value),
                "best_score_cp": int(best_score),
                "max_regret_cp": float(max_regret),
                "model_top1_regret_cp": float(model_top1_regret),
            })
            total_regret += float(max_regret)
            total_top1_regret += float(model_top1_regret)
            labeled += 1

            if bool(args_dict["progress"]):
                log_every = max(0, int(args_dict["log_every"]))
                if log_every > 0 and (labeled == 1 or labeled % log_every == 0):
                    print(
                        "offline label:",
                        f"worker={worker_index}",
                        f"labeled={labeled}",
                        f"global_index={global_index}",
                        f"mean_top1_regret_cp={total_top1_regret / max(1, labeled):.1f}",
                        flush=True,
                    )

    return {
        "rows": rows,
        "summary": {
            "worker": int(worker_index),
            "positions": int(labeled),
            "mean_max_regret_cp": float(total_regret / max(1, labeled)),
            "mean_top1_regret_cp": float(
                total_top1_regret / max(1, labeled)
            ),
        },
    }


def write_offline_h5(path: str, rows: List[Dict], summary: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = len(rows)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "states",
            data=np.asarray([row["state"] for row in rows], dtype=np.uint8).reshape(count, 18, 8),
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )
        for key in ("action_rewards", "model_policy"):
            h5.create_dataset(
                key,
                data=np.asarray([row[key] for row in rows], dtype=np.float16),
                dtype="float16",
                chunks=True,
                compression="lzf",
            )
        h5.create_dataset(
            "candidate_mask",
            data=np.asarray([row["candidate_mask"] for row in rows], dtype=np.uint8),
            dtype="uint8",
            chunks=True,
            compression="lzf",
        )
        for key, dtype in [
            ("teacher_value", np.float32),
            ("model_value", np.float32),
            ("best_score_cp", np.int32),
            ("max_regret_cp", np.float32),
            ("model_top1_regret_cp", np.float32),
        ]:
            h5.create_dataset(key, data=np.asarray([row[key] for row in rows], dtype=dtype))
        h5.attrs["summary_json"] = json.dumps(summary, ensure_ascii=False)
        h5.attrs["generator"] = "offline_actor_critic"


def read_offline_summary(path: str) -> Dict:
    with h5py.File(path, "r") as h5:
        raw = h5.attrs.get("summary_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw:
            return json.loads(str(raw))
        return {"positions": int(h5["states"].shape[0])}


def generate_offline_data(args, model_path: str, output_path: str, iteration: int) -> Dict:
    fens = load_fens(args, iteration=iteration)
    print(
        "offline reinforce labeling start:",
        f"iteration={iteration}",
        f"model={model_path}",
        f"source={args.fen_source or PGN_PATH}",
        f"positions={len(fens)}",
        f"sample_topk={args.sample_topk}",
        f"device={args.label_device or args.device}",
        flush=True,
    )

    workers = max(1, min(int(args.parallel), len(fens)))
    splits = [[] for _ in range(workers)]
    for index, fen in enumerate(fens):
        splits[index % workers].append((index + 1, fen))

    args_dict = vars(args).copy()
    args_dict["progress"] = True
    jobs = [
        (args_dict, worker_index + 1, workers, model_path, chunk)
        for worker_index, chunk in enumerate(splits)
        if chunk
    ]

    if len(jobs) == 1:
        outputs = [label_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
            outputs = pool.map(label_worker, jobs)

    rows = []
    worker_summaries = []
    for output in outputs:
        rows.extend(output["rows"])
        worker_summaries.append(output["summary"])

    if not rows:
        raise RuntimeError("offline labeling produced no rows")

    regret_values = np.asarray([row["max_regret_cp"] for row in rows], dtype=np.float32)
    top1_regret_values = np.asarray(
        [row["model_top1_regret_cp"] for row in rows],
        dtype=np.float32,
    )
    summary = {
        "iteration": int(iteration),
        "path": output_path,
        "positions": int(len(rows)),
        "source": str(args.fen_source or PGN_PATH),
        "source_offset": int(args.source_offset) + max(0, int(iteration) - 1) * int(args.positions_per_iter),
        "sample_topk": int(args.sample_topk),
        "include_teacher_best": bool(args.include_teacher_best),
        "mean_max_regret_cp": float(np.mean(regret_values)),
        "p90_max_regret_cp": float(np.percentile(regret_values, 90)),
        "mean_top1_regret_cp": float(np.mean(top1_regret_values)),
        "p90_top1_regret_cp": float(np.percentile(top1_regret_values, 90)),
        "reward_scale_cp": float(args.reward_scale_cp),
        "workers": worker_summaries,
    }
    write_offline_h5(output_path, rows, summary)
    print("offline reinforce label summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


class OfflineDataset(Dataset):
    def __init__(self, path: str):
        with h5py.File(path, "r") as h5:
            self.states = np.asarray(h5["states"], dtype=np.uint8)
            self.candidate_mask = np.asarray(h5["candidate_mask"], dtype=np.bool_)
            self.action_rewards = np.asarray(h5["action_rewards"], dtype=np.float32)

    def __len__(self):
        return int(self.states.shape[0])

    def __getitem__(self, index):
        return (
            torch.from_numpy(packed_to_tensor(self.states[index])),
            torch.from_numpy(self.candidate_mask[index]),
            torch.from_numpy(self.action_rewards[index]),
        )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def actor_critic_losses(logits, values, candidate_mask, action_rewards, args):
    masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
    raw_policy = F.softmax(masked_logits, dim=1)

    exploration_mix = max(0.0, min(1.0, float(args.actor_exploration_mix)))
    candidate_count = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1)
    uniform = candidate_mask.to(raw_policy.dtype) / candidate_count.to(raw_policy.dtype)
    policy = (
        (1.0 - exploration_mix) * raw_policy
        + exploration_mix * uniform
    )
    log_policy = torch.log(policy.clamp_min(torch.finfo(policy.dtype).tiny))

    critic_target = (policy.detach() * action_rewards).sum(dim=1)
    advantages = action_rewards - values.detach().unsqueeze(1)
    if float(args.advantage_clip) > 0:
        advantages = advantages.clamp(
            -float(args.advantage_clip),
            float(args.advantage_clip),
        )
    actor_loss = -(
        policy.detach() * advantages.detach() * log_policy
    ).sum(dim=1).mean()
    critic_loss = F.smooth_l1_loss(values, critic_target.detach())
    entropy = -(policy * log_policy).sum(dim=1).mean()
    return actor_loss, critic_loss, entropy, critic_target.mean()


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


def train_offline_actor_critic(args, source_model: str, data_path: str, candidate_path: str) -> Dict:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2 ** 32 - 1))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    dataset = OfflineDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError("offline dataset is empty")

    pin_memory = str(args.device).startswith("cuda")
    train_generator = torch.Generator().manual_seed(int(args.seed))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=pin_memory,
        persistent_workers=args.train_workers > 0,
        generator=train_generator,
    )

    supervised_iter = None
    if args.supervised_data and os.path.exists(args.supervised_data) and args.supervised_weight > 0:
        supervised_generator = torch.Generator().manual_seed(int(args.seed) + 1)
        supervised_loader = DataLoader(
            H5ChessDataset(args.supervised_data),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.train_workers,
            pin_memory=pin_memory,
            persistent_workers=args.train_workers > 0,
            generator=supervised_generator,
        )
        supervised_iter = cycle(supervised_loader)

    student = load_model(source_model, device=args.device)
    student.train()
    reference = load_model(source_model, device=args.device)
    reference.train()
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
    for epoch in range(int(args.epochs)):
        for states, candidate_mask, action_rewards in loader:
            states = states.to(args.device, non_blocking=True)
            candidate_mask = candidate_mask.to(
                args.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            action_rewards = action_rewards.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, values = student(states)
                values = values.squeeze(1)
                actor_loss, critic_loss, entropy, mean_reward = actor_critic_losses(
                    logits,
                    values,
                    candidate_mask,
                    action_rewards,
                    args,
                )
                with torch.no_grad():
                    reference_logits, _ = reference(states)
                    reference_logp = F.log_softmax(reference_logits / args.kl_temperature, dim=1)
                    reference_p = reference_logp.exp()
                student_logp = F.log_softmax(logits / args.kl_temperature, dim=1)
                kl_loss = (
                    reference_p * (reference_logp - student_logp)
                ).sum(dim=1).mean() * (max(1e-6, float(args.kl_temperature)) ** 2)

                supervised_loss = torch.zeros((), device=args.device)
                if supervised_iter is not None:
                    supervised_loss = supervised_loss_from_batch(
                        student,
                        next(supervised_iter),
                        args,
                    )

                loss = (
                    float(args.actor_weight) * actor_loss
                    + float(args.critic_weight) * critic_loss
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
                "actor": float(actor_loss.item()),
                "critic": float(critic_loss.item()),
                "reward": float(mean_reward.item()),
                "entropy": float(entropy.item()),
                "kl": float(kl_loss.item()),
                "supervised": float(supervised_loss.item()),
                "loss": float(loss.item()),
            }
            if args.log_every > 0 and (step == 1 or step % int(args.log_every) == 0):
                print(
                    "offline reinforce train step:",
                    f"epoch={epoch}",
                    f"step={step}",
                    f"actor={last_metrics['actor']:.4f}",
                    f"critic={last_metrics['critic']:.4f}",
                    f"reward={last_metrics['reward']:+.4f}",
                    f"entropy={last_metrics['entropy']:.4f}",
                    f"kl={last_metrics['kl']:.4f}",
                    f"supervised={last_metrics['supervised']:.4f}",
                    f"loss={last_metrics['loss']:.4f}",
                    flush=True,
                )
            if args.train_max_steps and step >= args.train_max_steps:
                stop = True
                break
        if stop:
            break

    save_model(
        candidate_path,
        student,
        optimizer=optimizer,
        epoch=max(0, int(args.epochs) - 1),
        global_step=step,
        extra={
            "type": "offline_actor_critic",
            "source_model": source_model,
            "offline_data": data_path,
            "last_metrics": last_metrics,
        },
    )
    print(
        "offline reinforce candidate saved:",
        f"path={candidate_path}",
        f"steps={step}",
        flush=True,
    )
    return {"steps": int(step), "candidate": candidate_path, "last_metrics": last_metrics}


def evaluate_candidate(args, candidate_path: str, baseline_path: str) -> Dict:
    if int(args.eval_games) <= 0:
        return {"accepted": False, "skipped": True, "games": 0}
    metrics = evaluate_models(
        candidate_path=candidate_path,
        baseline_path=baseline_path,
        games=args.eval_games,
        sims=args.eval_sims,
        workers=args.eval_workers or args.parallel,
        max_plies=args.eval_max_plies,
        device=args.device,
        opening_book=args.eval_opening_book,
        book_plies=args.eval_book_plies,
        max_book_positions=args.eval_max_book_positions,
        seed=args.seed,
        mcts_batch_size=args.eval_mcts_batch_size,
        movetime_ms=args.eval_movetime_ms,
        c_puct=args.eval_c_puct,
        c_puct_base=args.eval_c_puct_base,
        c_puct_factor=args.eval_c_puct_factor,
        fpu_reduction=args.eval_fpu_reduction,
        mcts_time_fraction=args.eval_mcts_time_fraction,
        mate_guard_plies=args.eval_mate_guard_plies,
        mate_guard_topk=args.eval_mate_guard_topk,
        mate_guard_nodes=args.eval_mate_guard_nodes,
        uci=args.uci,
        uci_depth=args.eval_uci_depth,
        uci_movetime_ms=args.eval_uci_movetime_ms,
        uci_multipv=args.eval_uci_multipv,
        uci_threads=args.uci_threads,
        uci_hash_mb=args.uci_hash_mb,
        quality_loss_cap_cp=args.eval_quality_loss_cap_cp,
        teacher_cache=args.teacher_cache,
        log_every=args.log_every,
        progress=True,
    )
    metrics = attach_arena_acceptance(
        metrics,
        min_net_wins=args.eval_min_net_wins,
        min_acpl_improvement=args.eval_min_acpl_improvement,
        min_accuracy_improvement=args.eval_min_accuracy_improvement,
    )
    print("offline reinforce arena summary:", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def run(args):
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}")

    paths = prepare_run_paths(args)
    current_model = paths["current_model"]
    if os.path.exists(current_model):
        print("offline reinforce current model reuse:", current_model, flush=True)
    else:
        shutil.copy2(args.model, current_model)
        print("offline reinforce current model initialized:", current_model, flush=True)
    print("offline reinforce run id:", paths["run_id"], flush=True)
    print("offline reinforce data run directory:", paths["data_run_dir"], flush=True)
    print("offline reinforce model run directory:", paths["model_run_dir"], flush=True)
    print(
        "offline reinforce start:",
        f"model={args.model}",
        f"iterations={args.iterations}",
        f"positions_per_iter={args.positions_per_iter}",
        f"device={args.device}",
        f"seed={args.seed}",
        flush=True,
    )

    summaries = []
    for iteration in range(1, int(args.iterations) + 1):
        print(f"offline reinforce iteration {iteration}", flush=True)
        data_path = os.path.join(paths["data_run_dir"], f"offline_iter_{iteration:03d}.h5")
        candidate_path = os.path.join(
            paths["model_run_dir"],
            f"candidate_iter_{iteration:03d}.pth",
        )

        if args.reuse_labels and os.path.exists(data_path):
            label_summary = read_offline_summary(data_path)
            label_summary["path"] = data_path
            print("reusing offline labels:", data_path, flush=True)
            print(json.dumps(label_summary, ensure_ascii=False, indent=2), flush=True)
        else:
            label_summary = generate_offline_data(
                args,
                model_path=current_model,
                output_path=data_path,
                iteration=iteration,
            )

        train_summary = train_offline_actor_critic(
            args,
            source_model=current_model,
            data_path=data_path,
            candidate_path=candidate_path,
        )
        eval_summary = evaluate_candidate(args, candidate_path, current_model)
        accepted = bool(eval_summary.get("accepted"))
        if accepted and args.promote_if_accepted:
            atomic_copy(candidate_path, current_model, make_backup=not args.no_backup)
            print("offline reinforce promoted:", current_model, flush=True)
        elif accepted:
            print(
                "offline reinforce candidate accepted but not promoted:",
                candidate_path,
                flush=True,
            )
        else:
            print("offline reinforce candidate rejected:", candidate_path, flush=True)

        iteration_summary = {
            "iteration": int(iteration),
            "data": label_summary,
            "train": train_summary,
            "eval": eval_summary,
            "accepted": accepted,
            "current_model": current_model,
        }
        summaries.append(iteration_summary)
        summary_path = os.path.join(paths["data_run_dir"], "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": paths["run_id"],
                    "summaries": summaries,
                    "current_model": current_model,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print("offline reinforce iteration summary:", flush=True)
        print(json.dumps(iteration_summary, ensure_ascii=False, indent=2), flush=True)

    print("offline reinforce complete:", current_model, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline Stockfish-rewarded actor-critic training for ChessAI."
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--supervised-data", default=H5_PATH)
    parser.add_argument("--fen-source", default=PGN_PATH)
    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--label-device", default=None)
    parser.add_argument("--data-runs-dir", default=DEFAULT_DATA_RUNS_DIR)
    parser.add_argument("--model-runs-dir", default=DEFAULT_MODEL_RUNS_DIR)
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--positions-per-iter", type=int, default=500)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--source-offset", type=int, default=0)
    parser.add_argument("--source-min-ply", type=int, default=0)
    parser.add_argument("--source-max-ply", type=int, default=160)
    parser.add_argument("--source-max-games", type=int, default=None)
    parser.add_argument("--reuse-labels", action="store_true", default=True)
    parser.add_argument("--no-reuse-labels", dest="reuse_labels", action="store_false")

    parser.add_argument("--sample-topk", type=int, default=8)
    parser.add_argument("--include-teacher-best", action="store_true", default=True)
    parser.add_argument(
        "--no-include-teacher-best",
        dest="include_teacher_best",
        action="store_false",
    )
    parser.add_argument("--reward-scale-cp", type=float, default=600.0)
    parser.add_argument("--actor-exploration-mix", type=float, default=0.05)
    parser.add_argument("--advantage-clip", type=float, default=1.0)

    parser.add_argument("--uci-depth", type=int, default=12)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-multipv", type=int, default=4)
    parser.add_argument("--uci-threads", type=int, default=1)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--teacher-cache", default=None)

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--actor-weight", type=float, default=1.0)
    parser.add_argument("--critic-weight", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=0.10)
    parser.add_argument("--kl-temperature", type=float, default=1.5)
    parser.add_argument("--supervised-weight", type=float, default=0.35)
    parser.add_argument("--supervised-value-weight", type=float, default=0.25)

    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--eval-max-plies", type=int, default=180)
    parser.add_argument("--eval-opening-book", default="data/openings.gen.bin")
    parser.add_argument("--eval-book-plies", type=int, default=8)
    parser.add_argument("--eval-max-book-positions", type=int, default=50000)
    parser.add_argument("--eval-mcts-batch-size", type=int, default=64)
    parser.add_argument("--eval-movetime-ms", type=int, default=1000)
    parser.add_argument("--eval-c-puct", type=float, default=0.5)
    parser.add_argument("--eval-c-puct-base", type=float, default=19652.0)
    parser.add_argument("--eval-c-puct-factor", type=float, default=1.0)
    parser.add_argument("--eval-fpu-reduction", type=float, default=0.15)
    parser.add_argument("--eval-mcts-time-fraction", type=float, default=0.90)
    parser.add_argument("--eval-mate-guard-plies", type=int, default=3)
    parser.add_argument("--eval-mate-guard-topk", type=int, default=8)
    parser.add_argument("--eval-mate-guard-nodes", type=int, default=20000)
    parser.add_argument("--eval-uci-depth", type=int, default=10)
    parser.add_argument("--eval-uci-movetime-ms", type=int, default=0)
    parser.add_argument("--eval-uci-multipv", type=int, default=6)
    parser.add_argument("--eval-quality-loss-cap-cp", type=int, default=1000)
    parser.add_argument("--eval-min-net-wins", type=int, default=5)
    parser.add_argument("--eval-min-acpl-improvement", type=float, default=0.0)
    parser.add_argument("--eval-min-accuracy-improvement", type=float, default=0.0)
    parser.add_argument("--promote-if-accepted", action="store_true", default=True)
    parser.add_argument(
        "--no-promote-if-accepted",
        dest="promote_if_accepted",
        action="store_false",
    )
    parser.add_argument("--no-backup", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
