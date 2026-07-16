"""Normalize a model checkpoint to the registered checkpoint format."""

import argparse
import os

import torch

from checkpoint_io import atomic_copy_with_backup
from model import checkpoint_metadata, load_model, save_model


def main():
    parser = argparse.ArgumentParser(description="Standardize a ChessAI checkpoint in place.")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    source_epoch, source_global_step, source_extra = checkpoint_metadata(model_path, device="cpu")
    model = load_model(model_path, device="cpu", infer_unknown_arch=True)
    model.eval()

    extra = dict(source_extra)
    extra["standardized"] = True

    tmp_path = f"{model_path}.standardize_tmp_{os.getpid()}"
    try:
        save_model(
            tmp_path,
            model,
            epoch=source_epoch,
            global_step=source_global_step,
            extra=extra,
        )
        result = atomic_copy_with_backup(tmp_path, model_path, make_backup=True)

        checkpoint = torch.load(model_path, map_location="cpu")
        keys = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else []
        print("standardization finished")
        print("model:", model_path)
        print("backup:", result.get("backup"))
        print("keys:", keys)
        print("arch:", checkpoint.get("arch") if isinstance(checkpoint, dict) else None)
        print("epoch:", checkpoint.get("epoch") if isinstance(checkpoint, dict) else None)
        print("global_step:", checkpoint.get("global_step") if isinstance(checkpoint, dict) else None)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
