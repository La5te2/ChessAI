import argparse
import os
import sys
from typing import Iterable, Tuple

import h5py
import numpy as np

from architectures import (
    RESNET_PVA_GAD,
    RESNET_PV_LINEAR,
    architecture_spec,
    normalize_arch_type,
)
from move_codecs import get_move_codec
from state_codecs import get_state_codec


CHUNK_ROWS = 65536
MAX_ERRORS = 20


class Inspector:
    def __init__(self):
        self.errors = 0

    def error(self, message: str):
        self.errors += 1
        print(f"[ERROR] {message}")

    def check(self, condition: bool, message: str):
        if not condition:
            self.error(message)

    @property
    def ok(self) -> bool:
        return self.errors == 0


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "decode"):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def dataset_names(h5) -> set:
    return set(str(name) for name in h5.keys())


def print_summary(path: str, h5):
    print("file:", path)
    print("size_gb:", f"{os.path.getsize(path) / 1024**3:.4f}")
    print("keys:", list(h5.keys()))
    for key in h5.keys():
        dataset = h5[key]
        print(key, dataset.shape, dataset.dtype)
    print("attrs:", dict(h5.attrs))


def iter_chunks(dataset, rows: int = CHUNK_ROWS) -> Iterable[Tuple[int, np.ndarray]]:
    total = int(dataset.shape[0])
    for start in range(0, total, rows):
        end = min(total, start + rows)
        yield start, np.asarray(dataset[start:end])


def require_datasets(inspector: Inspector, h5, required):
    names = dataset_names(h5)
    missing = [name for name in required if name not in names]
    for name in missing:
        inspector.error(f"missing dataset: {name}")
    return not missing


def require_dataset_set(inspector: Inspector, h5, expected):
    names = dataset_names(h5)
    missing = sorted(set(expected) - names)
    extra = sorted(names - set(expected))
    for name in missing:
        inspector.error(f"missing dataset: {name}")
    for name in extra:
        inspector.error(f"unexpected dataset for schema: {name}")
    return not missing and not extra


def check_same_length(inspector: Inspector, h5, names):
    lengths = {}
    for name in names:
        if name in h5:
            lengths[name] = int(h5[name].shape[0])
    if not lengths:
        return 0
    expected = next(iter(lengths.values()))
    for name, length in lengths.items():
        inspector.check(
            length == expected,
            f"{name} length={length} does not match expected length={expected}",
        )
    inspector.check(expected > 0, "dataset length is zero")
    return expected


def check_states(inspector: Inspector, h5, length: int, state_codec):
    if "states" not in h5:
        return
    states = h5["states"]
    expected_shape = (length, *state_codec.storage_shape)
    inspector.check(
        states.shape == expected_shape,
        f"states shape={states.shape}, expected {expected_shape}",
    )
    inspector.check(
        states.dtype == np.dtype(state_codec.storage_dtype),
        f"states dtype={states.dtype}, expected {state_codec.storage_dtype}",
    )


def check_index_dataset(inspector: Inspector, h5, name: str, action_size: int):
    if name not in h5:
        return
    dataset = h5[name]
    inspector.check(
        np.issubdtype(dataset.dtype, np.integer),
        f"{name} dtype={dataset.dtype}, expected integer",
    )
    for start, chunk in iter_chunks(dataset):
        if chunk.size == 0:
            continue
        bad = np.where((chunk < 0) | (chunk >= action_size))[0]
        if bad.size:
            index = int(start + bad[0])
            inspector.error(
                f"{name}[{index}]={int(chunk[bad[0]])} outside [0, {action_size})"
            )
            if inspector.errors >= MAX_ERRORS:
                return


def check_values(inspector: Inspector, h5, name: str, allowed=None, bounds=None):
    if name not in h5:
        return
    dataset = h5[name]
    for start, chunk in iter_chunks(dataset):
        if chunk.size == 0:
            continue
        if not np.isfinite(chunk).all():
            bad = np.where(~np.isfinite(chunk))[0][0]
            inspector.error(f"{name}[{int(start + bad)}] is not finite")
            return
        if allowed is not None:
            allowed_values = np.asarray(list(allowed))
            ok = np.isin(chunk, allowed_values)
            if not bool(ok.all()):
                bad = np.where(~ok)[0][0]
                inspector.error(
                    f"{name}[{int(start + bad)}]={chunk[bad]} outside {sorted(allowed)}"
                )
                return
        if bounds is not None:
            lo, hi = bounds
            bad = np.where((chunk < lo) | (chunk > hi))[0]
            if bad.size:
                index = int(start + bad[0])
                inspector.error(
                    f"{name}[{index}]={float(chunk[bad[0]])} outside [{lo}, {hi}]"
                )
                return


def check_supervised_attrs(inspector: Inspector, h5):
    for name in ("arch_type", "state_encoding", "move_encoding", "target_schema", "has_cmt"):
        inspector.check(name in h5.attrs, f"missing attr: {name}")
    if not all(name in h5.attrs for name in ("arch_type", "state_encoding", "move_encoding", "target_schema", "has_cmt")):
        return None, None, None, None, None
    try:
        arch_type = normalize_arch_type(decode_attr(h5.attrs["arch_type"]))
    except Exception as exc:
        inspector.error(str(exc))
        return None, None, None, None, None
    state_encoding = decode_attr(h5.attrs["state_encoding"])
    move_encoding = decode_attr(h5.attrs["move_encoding"])
    target_schema = decode_attr(h5.attrs["target_schema"])
    has_cmt = int(h5.attrs["has_cmt"])
    inspector.check(has_cmt in (0, 1), f"has_cmt={has_cmt!r}, expected 0 or 1")
    expected_state_encoding = architecture_spec(arch_type).state_encoding
    expected_move_encoding = architecture_spec(arch_type).move_encoding
    inspector.check(
        state_encoding == expected_state_encoding,
        f"state_encoding={state_encoding!r}, expected {expected_state_encoding!r} for {arch_type}",
    )
    inspector.check(
        move_encoding == expected_move_encoding,
        f"move_encoding={move_encoding!r}, expected {expected_move_encoding!r} for {arch_type}",
    )
    try:
        get_state_codec(state_encoding)
    except Exception as exc:
        inspector.error(str(exc))
    try:
        get_move_codec(move_encoding)
    except Exception as exc:
        inspector.error(str(exc))
    return arch_type, state_encoding, move_encoding, target_schema, has_cmt


def check_resnet_pv_linear(inspector: Inspector, h5, state_encoding: str, move_encoding: str, target_schema: str, has_cmt: int):
    print("schema:", RESNET_PV_LINEAR)
    inspector.check(target_schema == "pv_supervised", f"target_schema={target_schema!r}, expected 'pv_supervised'")
    require_dataset_set(inspector, h5, ("states", "moves", "values"))
    length = check_same_length(inspector, h5, ("states", "moves", "values"))
    state_codec = get_state_codec(state_encoding)
    codec = get_move_codec(move_encoding)
    check_states(inspector, h5, length, state_codec)
    check_index_dataset(inspector, h5, "moves", codec.action_size)
    check_values(inspector, h5, "values", bounds=(-1.0, 1.0))


def check_resnet_pva_gad(inspector: Inspector, h5, state_encoding: str, move_encoding: str, target_schema: str, has_cmt: int):
    print("schema:", RESNET_PVA_GAD)
    inspector.check(
        target_schema == "pva_value_scaled_advantage",
        f"target_schema={target_schema!r}, expected 'pva_value_scaled_advantage'",
    )
    require_dataset_set(
        inspector,
        h5,
        ("states", "moves", "values", "adv_moves", "adv_values"),
    )
    length = check_same_length(
        inspector,
        h5,
        ("states", "moves", "values", "adv_moves", "adv_values"),
    )
    state_codec = get_state_codec(state_encoding)
    codec = get_move_codec(move_encoding)
    check_states(inspector, h5, length, state_codec)
    check_index_dataset(inspector, h5, "moves", codec.action_size)
    check_index_dataset(inspector, h5, "adv_moves", codec.action_size)
    check_values(inspector, h5, "values", bounds=(-1.0, 1.0))
    check_values(inspector, h5, "adv_values", bounds=(-1.0, 0.0))


INSPECTION_HANDLERS = {
    RESNET_PV_LINEAR: check_resnet_pv_linear,
    RESNET_PVA_GAD: check_resnet_pva_gad,
}


def inspect(path: str) -> bool:
    inspector = Inspector()
    with h5py.File(path, "r") as h5:
        print_summary(path, h5)
        arch_type, state_encoding, move_encoding, target_schema, has_cmt = check_supervised_attrs(inspector, h5)
        if arch_type is not None:
            handler = INSPECTION_HANDLERS.get(arch_type)
            if handler is None:
                inspector.error(f"no inspection schema registered for arch_type={arch_type!r}")
            else:
                handler(inspector, h5, state_encoding, move_encoding, target_schema, has_cmt)
    print("INSPECTION OK" if inspector.ok else "INSPECTION FAILED")
    return inspector.ok


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a ChessAI HDF5 file")
    parser.add_argument("--path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not inspect(args.path):
        sys.exit(1)
