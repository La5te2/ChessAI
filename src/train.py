import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from acceptance import attach_arena_acceptance
from arena import evaluate_models
from checkpoint_io import atomic_copy_with_backup, tmp_candidate_path_from_resume
from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    H5_PATH,
    LR,
    MODEL_PATH,
    NUM_WORKERS,
    STOCKFISH_PATH,
    VALUE_LOSS_WEIGHT,
    WEIGHT_DECAY,
)
from data import H5ChessDataset
from model import ChessNet, make_model_from_checkpoint, save_model


def _load_resume_weights(path, device):
    checkpoint = torch.load(path, map_location=device)
    source_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    source_step = int(checkpoint.get("global_step", 0)) if isinstance(checkpoint, dict) else 0
    return (
        make_model_from_checkpoint(checkpoint, device=device),
        source_epoch,
        source_step,
    )


def _resolve_output_path(args):
    if not args.resume:
        return args.out or MODEL_PATH
    output = args.out or tmp_candidate_path_from_resume(args.resume)
    if os.path.abspath(output) == os.path.abspath(args.resume):
        raise ValueError("--out must differ from --resume")
    return output


def _run_resume_validation(args, candidate_path):
    if not args.resume:
        return None
    if args.eval_games <= 0:
        print("resume validation skipped: eval_games <= 0", flush=True)
        return {"accepted": None, "reason": "eval_games <= 0"}

    print(
        "resume validation start:",
        f"candidate={candidate_path}",
        f"baseline={args.resume}",
        f"games={args.eval_games}",
        f"sims={args.eval_sims}",
        f"device={args.device}",
        flush=True,
    )
    metrics = evaluate_models(
        candidate_path=candidate_path,
        baseline_path=args.resume,
        games=args.eval_games,
        sims=args.eval_sims,
        workers=args.eval_workers,
        device=args.device,
        max_plies=args.eval_max_plies,
        seed=args.eval_seed,
        opening_book=args.eval_opening_book,
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
        q_tiebreak_min_visits=args.eval_q_tiebreak_min_visits,
        q_tiebreak_p_ratio=args.eval_q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio=args.eval_q_tiebreak_visit_ratio,
        q_tiebreak_margin=args.eval_q_tiebreak_margin,
        uci=args.uci,
        uci_depth=args.eval_uci_depth,
        uci_movetime_ms=args.eval_uci_movetime_ms,
        uci_threads=args.eval_uci_threads,
        uci_hash_mb=args.eval_uci_hash_mb,
        uci_multipv=args.eval_uci_multipv,
        teacher_cache=args.teacher_cache,
        progress=True,
    )
    metrics = attach_arena_acceptance(
        metrics,
        min_net_wins=args.eval_min_net_wins,
        min_acpl_improvement=args.eval_min_acpl_improvement,
        min_accuracy_improvement=args.eval_min_accuracy_improvement,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        "resume validation decision:",
        f"result_ok={metrics.get('result_ok')}",
        f"quality_ok={metrics.get('quality_ok')}",
        f"accepted={metrics.get('accepted')}",
        flush=True,
    )

    if metrics.get("accepted"):
        atomic_copy_with_backup(
            candidate_path,
            args.resume,
            make_backup=not args.no_backup,
        )
        print("resume model updated:", args.resume)
    else:
        print("candidate rejected:", candidate_path)
    return metrics


def train(args):
    args.out = _resolve_output_path(args)
    device = str(args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pin_memory = device.startswith("cuda")
    print(
        "training start:",
        f"data={args.data}",
        f"out={args.out}",
        f"resume={args.resume}",
        f"device={device}",
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"max_steps={args.max_steps}",
        flush=True,
    )

    loader = DataLoader(
        H5ChessDataset(args.data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    source_step = 0
    if args.resume:
        model, source_epoch, source_step = _load_resume_weights(
            args.resume,
            device,
        )
        print(
            f"loaded weights from {args.resume}: "
            f"epoch={source_epoch}, global_step={source_step}"
        )
    else:
        model = ChessNet(
            channels=args.channels,
            blocks=args.blocks,
        ).to(device)
        print(
            "created model:",
            f"channels={args.channels}",
            f"blocks={args.blocks}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()
    amp_enabled = bool(args.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step = int(source_step)
    run_step = 0
    stop = False
    model.train()

    for epoch in range(args.epochs):
        total_policy = 0.0
        total_value = 0.0
        batches = 0

        for state, move, value_target in loader:
            state = state.to(device, non_blocking=pin_memory)
            move = move.to(device, non_blocking=pin_memory)
            value_target = value_target.to(device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                policy_logits, value = model(state)
                policy_loss = policy_loss_fn(policy_logits, move)
                value_loss = value_loss_fn(
                    value.squeeze(1),
                    value_target,
                )
                loss = policy_loss + args.value_weight * value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            run_step += 1
            batches += 1
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            if (
                args.log_every > 0
                and (
                    run_step == 1
                    or run_step % args.log_every == 0
                )
            ):
                print(
                    "train step:",
                    f"epoch={epoch}",
                    f"run_step={run_step}",
                    f"global_step={global_step}",
                    f"policy={policy_loss.item():.4f}",
                    f"value={value_loss.item():.4f}",
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_model(
                    args.out,
                    model,
                    epoch=epoch,
                    global_step=global_step,
                    extra={"type": "supervised"},
                )
                print(
                    "checkpoint saved:",
                    f"path={args.out}",
                    f"global_step={global_step}",
                    flush=True,
                )

            if args.max_steps is not None and run_step >= args.max_steps:
                stop = True
                break

        save_model(
            args.out,
            model,
            epoch=epoch,
            global_step=global_step,
            extra={"type": "supervised"},
        )
        print(
            f"epoch={epoch}, steps={global_step}, "
            f"policy={total_policy / max(1, batches):.4f}, "
            f"value={total_value / max(1, batches):.4f}",
            flush=True,
        )
        if stop:
            break

    print("training finished:", args.out)
    _run_resume_validation(args, args.out)


def parse_args():
    parser = argparse.ArgumentParser(description="ChessAI supervised trainer")
    parser.add_argument("--data", default=H5_PATH)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--value-weight", type=float, default=VALUE_LOSS_WEIGHT)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-sims", type=int, default=80)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--eval-max-plies", type=int, default=240)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--eval-min-net-wins", type=int, default=3)
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
    parser.add_argument("--eval-mcts-batch-size", type=int, default=32)
    parser.add_argument("--eval-movetime-ms", type=int, default=5000)
    parser.add_argument("--eval-c-puct", type=float, default=1.5)
    parser.add_argument("--eval-mate-guard-plies", type=int, default=3)
    parser.add_argument("--eval-mate-guard-topk", type=int, default=8)
    parser.add_argument("--eval-mate-guard-nodes", type=int, default=20000)
    parser.add_argument("--eval-mate-guard-time-fraction", type=float, default=0.10)
    parser.add_argument("--eval-q-tiebreak", action="store_true", default=True)
    parser.add_argument(
        "--no-eval-q-tiebreak",
        dest="eval_q_tiebreak",
        action="store_false",
    )
    parser.add_argument("--eval-q-tiebreak-min-visits", type=int, default=32)
    parser.add_argument("--eval-q-tiebreak-p-ratio", type=float, default=0.90)
    parser.add_argument(
        "--eval-q-tiebreak-visit-ratio",
        type=float,
        default=0.80,
    )
    parser.add_argument("--eval-q-tiebreak-margin", type=float, default=0.25)

    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--eval-uci-depth", type=int, default=8)
    parser.add_argument(
        "--eval-uci-movetime-ms",
        type=int,
        default=0,
    )
    parser.add_argument("--eval-uci-threads", type=int, default=4)
    parser.add_argument("--eval-uci-hash-mb", type=int, default=512)
    parser.add_argument("--eval-uci-multipv", type=int, default=4)
    parser.add_argument(
        "--teacher-cache",
        default="data/selflearn/teacher_cache.sqlite",
    )
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
    parser.add_argument("--no-backup", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
