import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    H5_PATH,
    LR,
    MODEL_PATH,
    NUM_WORKERS,
    VALUE_LOSS_WEIGHT,
    WEIGHT_DECAY,
)
from data import H5ChessDataset
from model import ChessNet, save_model


def train(args):
    output_path = args.out or MODEL_PATH
    device = str(args.device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pin_memory = device.startswith("cuda")
    print(
        "training start:",
        f"data={args.data}",
        f"out={output_path}",
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

    global_step = 0
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
            batches += 1
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            if (
                args.log_every > 0
                and (
                    global_step == 1
                    or global_step % args.log_every == 0
                )
            ):
                print(
                    "train step:",
                    f"epoch={epoch}",
                    f"global_step={global_step}",
                    f"policy={policy_loss.item():.4f}",
                    f"value={value_loss.item():.4f}",
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_model(
                    output_path,
                    model,
                    epoch=epoch,
                    global_step=global_step,
                    extra={"type": "supervised"},
                )
                print(
                    "checkpoint saved:",
                    f"path={output_path}",
                    f"global_step={global_step}",
                    flush=True,
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        save_model(
            output_path,
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

    print("training finished:", output_path)


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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
