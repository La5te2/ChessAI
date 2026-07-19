import argparse
import os

import torch
from torch.utils.data import DataLoader

from architectures import (
    DEFAULT_ARCH_TYPE,
    RESNET_PVA_GAD,
    RESNET_PV_LINEAR,
    SUPPORTED_ARCH_TYPES,
)
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
from model import (
    create_model,
    save_model,
)


def _to_device(batch, device, pin_memory):
    return tuple(item.to(device, non_blocking=pin_memory) for item in batch)


def _resnet_pv_linear_loss(model, batch, losses, args, device, pin_memory):
    state, move, value_target = _to_device(batch, device, pin_memory)
    heads = model.forward_heads(state)
    policy_loss = losses["policy"](heads["policy_logits"], move)
    value_loss = losses["value"](heads["value"].squeeze(1), value_target)
    loss = policy_loss + args.value_weight * value_loss
    return loss, {
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
        "loss": float(loss.item()),
    }


def _resnet_pva_gad_loss(model, batch, losses, args, device, pin_memory):
    state, move, value_target, adv_move, adv_target = _to_device(
        batch,
        device,
        pin_memory,
    )
    heads = model.forward_heads(state)
    policy_loss = losses["policy"](heads["policy_logits"], move)
    predicted_value = heads["value"].squeeze(1)
    value_loss = losses["value"](predicted_value, value_target)
    chosen_advantage = heads["advantages"].gather(1, adv_move.unsqueeze(1)).squeeze(1)
    if int(getattr(args, "has_cmt", 0)):
        predicted_q = torch.clamp(predicted_value + chosen_advantage, -1.0, 1.0)
        q_target = torch.clamp(value_target + adv_target, -1.0, 1.0)
        q_loss = losses["value"](predicted_q, q_target)
    else:
        q_loss = chosen_advantage.sum() * 0.0
    loss = (
        policy_loss
        + args.value_weight * value_loss
        + args.dueling_q_weight * q_loss
    )
    return loss, {
        "policy": float(policy_loss.item()),
        "value": float(value_loss.item()),
        "dueling_q": float(q_loss.item()),
        "loss": float(loss.item()),
    }


TRAIN_HANDLERS = {
    RESNET_PV_LINEAR: {
        "loss": _resnet_pv_linear_loss,
        "metrics": ("policy", "value"),
        "start_fields": (),
    },
    RESNET_PVA_GAD: {
        "loss": _resnet_pva_gad_loss,
        "metrics": ("policy", "value", "dueling_q"),
        "start_fields": ("dueling_q_weight",),
    },
}


def _handler_for_arch(arch_type: str):
    try:
        return TRAIN_HANDLERS[arch_type]
    except KeyError as exc:
        raise ValueError(f"no train handler registered for {arch_type!r}") from exc


def train(args):
    output_path = args.out or MODEL_PATH
    device = str(args.device)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pin_memory = device.startswith("cuda")
    handler = _handler_for_arch(args.arch_type)
    start_parts = [
        "training start:",
        f"data={args.data}",
        f"out={output_path}",
        f"arch_type={args.arch_type}",
        f"device={device}",
        f"epochs={args.epochs}",
        f"batch_size={args.batch_size}",
        f"max_steps={args.max_steps}",
    ]
    for field in handler["start_fields"]:
        start_parts.append(f"{field}={getattr(args, field)}")
    print(*start_parts, flush=True)

    dataset = H5ChessDataset(args.data, arch_type=args.arch_type)
    args.has_cmt = int(dataset.has_cmt)
    print(
        "training data:",
        f"arch_type={dataset.arch_type}",
        f"state_encoding={dataset.state_encoding}",
        f"move_encoding={dataset.move_encoding}",
        f"target_schema={dataset.target_schema}",
        f"has_cmt={dataset.has_cmt}",
        f"datasets={','.join(dataset.datasets)}",
        flush=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    model = create_model(
        arch_type=args.arch_type,
        channels=args.channels,
        blocks=args.blocks,
        device=device,
    )
    model_parts = [
        "created model:",
        f"arch_type={args.arch_type}",
        f"channels={args.channels}",
        f"blocks={args.blocks}",
    ]
    print(*model_parts, flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    losses = {
        "policy": torch.nn.CrossEntropyLoss(),
        "value": torch.nn.MSELoss(),
    }
    amp_enabled = bool(args.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    global_step = 0
    stop = False
    model.train()

    for epoch in range(args.epochs):
        totals = {name: 0.0 for name in handler["metrics"]}
        batches = 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss, metrics = handler["loss"](
                    model,
                    batch,
                    losses,
                    args,
                    device,
                    pin_memory,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            batches += 1
            for name in handler["metrics"]:
                totals[name] += float(metrics[name])
            if (
                args.log_every > 0
                and (
                    global_step == 1
                    or global_step % args.log_every == 0
                )
            ):
                parts = [
                    "train step:",
                    f"epoch={epoch}",
                    f"global_step={global_step}",
                ]
                for name in handler["metrics"]:
                    parts.append(f"{name}={metrics[name]:.4f}")
                parts.append(f"loss={metrics['loss']:.4f}")
                print(*parts, flush=True)

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
        parts = [
            f"epoch={epoch}",
            f"steps={global_step}",
        ]
        for name in handler["metrics"]:
            parts.append(f"{name}={totals[name] / max(1, batches):.4f}")
        print(", ".join(parts), flush=True)
        if stop:
            break

    print("training finished:", output_path)


def parse_args():
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument(
        "--arch-type",
        choices=sorted(SUPPORTED_ARCH_TYPES),
        default=DEFAULT_ARCH_TYPE,
    )
    probe_args, _ = probe.parse_known_args()

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
    parser.add_argument(
        "--arch-type",
        choices=sorted(SUPPORTED_ARCH_TYPES),
        default=DEFAULT_ARCH_TYPE,
    )
    if probe_args.arch_type == RESNET_PVA_GAD:
        parser.add_argument(
            "--dueling-q-weight",
            type=float,
            default=VALUE_LOSS_WEIGHT,
        )
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
