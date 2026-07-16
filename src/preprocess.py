import argparse
import json
import os
import re
import chess
import chess.pgn
import h5py
import numpy as np
from architectures import (
    DEFAULT_ARCH_TYPE,
    RESNET_PVA_GAD,
    RESNET_PV_LINEAR,
    SUPPORTED_ARCH_TYPES,
    architecture_spec,
    normalize_arch_type,
)
from config import PGN_PATH, H5_PATH
from move_codecs import get_move_codec
from state_codecs import get_state_codec

CCRL_EVAL_RE = re.compile(r"(?<![\w.])([+-](?:\d+(?:\.\d+)?|\.\d+))(?:/\d+)?")
COMMENT_ADVANTAGE_DELTA_SCALE_PAWNS = 3.0

NON_STATE_DATASET_LAYOUTS = {
    "moves": {
        "shape": (0,),
        "maxshape": (None,),
        "chunks": lambda chunk_size: (min(chunk_size, 8192),),
        "dtype": "uint16",
    },
    "values": {
        "shape": (0,),
        "maxshape": (None,),
        "chunks": lambda chunk_size: (min(chunk_size, 8192),),
        "dtype": "int8",
    },
    "adv_moves": {
        "shape": (0,),
        "maxshape": (None,),
        "chunks": lambda chunk_size: (min(chunk_size, 8192),),
        "dtype": "uint16",
    },
    "adv_values": {
        "shape": (0,),
        "maxshape": (None,),
        "chunks": lambda chunk_size: (min(chunk_size, 8192),),
        "dtype": "float32",
    },
    "adv_weights": {
        "shape": (0,),
        "maxshape": (None,),
        "chunks": lambda chunk_size: (min(chunk_size, 8192),),
        "dtype": "float32",
    },
}

DATASET_DTYPES = {
    "states": np.uint8,
    "moves": np.uint16,
    "values": np.int8,
    "adv_moves": np.uint16,
    "adv_values": np.float32,
    "adv_weights": np.float32,
}


def dataset_layout(name: str, state_codec, chunk_size: int):
    if name == "states":
        return {
            "shape": (0, *state_codec.storage_shape),
            "maxshape": (None, *state_codec.storage_shape),
            "chunks": (min(chunk_size, 8192), *state_codec.storage_shape),
            "dtype": state_codec.storage_dtype,
        }
    return {
        **NON_STATE_DATASET_LAYOUTS[name],
        "chunks": NON_STATE_DATASET_LAYOUTS[name]["chunks"](chunk_size),
    }

def result_to_white_value(result: str) -> int:
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0

def comment_score_white(comment: str):
    match = CCRL_EVAL_RE.search(comment or "")
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def comment_advantage_target(before_comment: str, after_comment: str, turn: chess.Color):
    before_white = comment_score_white(before_comment)
    after_white = comment_score_white(after_comment)
    if before_white is None or after_white is None:
        return None
    white_delta = after_white - before_white
    side_delta = white_delta if turn == chess.WHITE else -white_delta
    side_delta = min(0.0, side_delta)
    return float(np.tanh(side_delta / COMMENT_ADVANTAGE_DELTA_SCALE_PAWNS))

def create_h5(
    path: str,
    compression: str,
    compression_opts: int,
    chunk_size: int,
    arch_type: str,
):
    spec = architecture_spec(arch_type)
    state_codec = get_state_codec(spec.state_encoding)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h5 = h5py.File(path, "w")
    kwargs = {}
    if compression != "none":
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = compression_opts
        kwargs["shuffle"] = True

    datasets = {}
    for name in spec.supervised_datasets:
        layout = dataset_layout(name, state_codec, chunk_size)
        datasets[name] = h5.create_dataset(
            name,
            shape=layout["shape"],
            maxshape=layout["maxshape"],
            chunks=layout["chunks"],
            dtype=layout["dtype"],
            **kwargs,
        )
    h5.attrs["state_encoding"] = spec.state_encoding
    h5.attrs["move_encoding"] = spec.move_encoding
    h5.attrs["value_perspective"] = "side_to_move"
    h5.attrs["arch_type"] = spec.name
    h5.attrs["target_schema"] = spec.target_schema
    if spec.comment_advantage:
        h5.attrs["comment_eval_perspective"] = "white"
        h5.attrs["advantage_perspective"] = "side_to_move"
        h5.attrs["advantage_source"] = "ccrl_after_move_comment_delta"
        h5.attrs["advantage_transform"] = (
            f"tanh(min(side_to_move_delta_pawn_score,0)/{COMMENT_ADVANTAGE_DELTA_SCALE_PAWNS:g})"
        )
    return h5, datasets

def append_chunk(
    datasets,
    buffers,
):
    first_name = next(iter(datasets))
    n = len(buffers[first_name])
    if n == 0:
        return
    old = datasets["states"].shape[0]
    new = old + n
    for name, dataset in datasets.items():
        dataset.resize(new, axis=0)
        dataset[old:new] = np.asarray(buffers[name], dtype=DATASET_DTYPES[name])


def pv_linear_row(board, node, child, action, value, state_codec):
    return {
        "states": state_codec.encode_board(board),
        "moves": action,
        "values": value,
    }, None


def pva_gad_row(board, node, child, action, value, state_codec):
    adv_target = comment_advantage_target(
        node.comment,
        child.comment,
        board.turn,
    )
    if adv_target is None:
        adv_target = 0.0
        adv_weight = 0.0
        skip_reason = "unannotated_advantage"
    else:
        adv_weight = 1.0
        skip_reason = None
    return {
        "states": state_codec.encode_board(board),
        "moves": action,
        "values": value,
        "adv_moves": action,
        "adv_values": adv_target,
        "adv_weights": adv_weight,
    }, skip_reason


PREPROCESS_ROW_BUILDERS = {
    RESNET_PV_LINEAR: pv_linear_row,
    RESNET_PVA_GAD: pva_gad_row,
}

def collect_game_offsets(path: str):
    """
    Collect text-file offsets for PGN games.

    PGN games normally start with an [Event "..."] tag. This function records
    those offsets so --random-select can sample games from the whole input file
    instead of only taking the first max-games games.
    """
    offsets = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith("[Event "):
                offsets.append(pos)
    return offsets

def stratified_random_offsets(offsets, max_games):
    """
    Pick games from different file regions.

    This is intentionally not "take max_games uniformly from the first region".
    It splits the entire PGN game-index range into max_games buckets and picks
    one random game from each bucket. The selected offsets are returned in file
    order, so output HDF5 writing remains sequential.
    """
    total = len(offsets)
    if max_games is None or max_games >= total:
        return list(offsets)
    if max_games <= 0:
        return []

    rng = np.random.default_rng()
    selected_indices = []
    for i in range(max_games):
        lo = (i * total) // max_games
        hi = ((i + 1) * total) // max_games
        if hi <= lo:
            hi = lo + 1
        selected_indices.append(int(rng.integers(lo, hi)))

    # Buckets are increasing, so this is already in file order. sorted() also
    # protects against edge cases and duplicate offsets.
    selected_indices = sorted(set(selected_indices))
    return [offsets[i] for i in selected_indices]

def preprocess(args):
    arch_type = normalize_arch_type(args.arch_type)
    spec = architecture_spec(arch_type)
    move_codec = get_move_codec(spec.move_encoding)
    state_codec = get_state_codec(spec.state_encoding)
    row_builder = PREPROCESS_ROW_BUILDERS[spec.name]
    h5, datasets = create_h5(
        args.output,
        args.compression,
        args.compression_opts,
        args.chunk_size,
        arch_type,
    )
    buffers = {name: [] for name in spec.supervised_datasets}
    games = positions = skipped_moves = unannotated_advantages = annotated_advantages = 0
    print(
        "preprocess start:",
        f"input={args.input}",
        f"output={args.output}",
        f"arch_type={arch_type}",
        flush=True,
    )

    def live_positions():
        first_name = spec.supervised_datasets[0]
        return positions + len(buffers[first_name])

    def print_progress():
        if args.log_every <= 0 or games <= 0:
            return
        if games % args.log_every != 0:
            return
        print(
            "preprocess progress:",
            f"games={games}",
            f"positions={live_positions()}",
            f"skipped_moves={skipped_moves}",
            f"output={args.output}",
            flush=True,
        )

    def flush_chunk():
        nonlocal positions
        first_name = spec.supervised_datasets[0]
        if len(buffers[first_name]) == 0:
            return
        append_chunk(datasets, buffers)
        positions += len(buffers[first_name])
        for values in buffers.values():
            values.clear()

    def process_game(game):
        nonlocal skipped_moves, unannotated_advantages, annotated_advantages
        white_value = result_to_white_value(game.headers.get("Result", "*"))
        board = game.board()

        node = game
        while node.variations:
            child = node.variation(0)
            move = child.move
            try:
                action = move_codec.move_to_index(move)
                value = white_value if board.turn == chess.WHITE else -white_value
                row, skip_reason = row_builder(board, node, child, action, value, state_codec)
                if row is None:
                    skipped_moves += 1
                    board.push(move)
                    node = child
                    continue
                for name, item in row.items():
                    buffers[name].append(item)
                if spec.comment_advantage:
                    if skip_reason == "unannotated_advantage":
                        unannotated_advantages += 1
                    else:
                        annotated_advantages += 1
            except Exception:
                skipped_moves += 1
            board.push(move)
            node = child

            first_name = spec.supervised_datasets[0]
            if len(buffers[first_name]) >= args.chunk_size:
                flush_chunk()

    try:
        if args.random_select:
            print("random_select: indexing PGN game offsets...")
            offsets = collect_game_offsets(args.input)
            print("indexed_games:", len(offsets))

            if offsets:
                selected_offsets = stratified_random_offsets(offsets, args.max_games)
                print("selected_games:", len(selected_offsets))
                print("selection_method: stratified_random_by_game_index")

                with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
                    for offset in selected_offsets:
                        f.seek(offset)
                        game = chess.pgn.read_game(f)
                        if game is None:
                            continue

                        process_game(game)
                        games += 1
                        print_progress()
            else:
                print("warning: no [Event ...] game offsets found; falling back to sequential read.")
                args.random_select = False

        if not args.random_select:
            with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    if args.max_games is not None and games >= args.max_games:
                        break
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    process_game(game)
                    games += 1
                    print_progress()

        flush_chunk()
        h5.attrs["games"] = games
        h5.attrs["positions"] = positions
        h5.attrs["skipped_moves"] = skipped_moves
        h5.attrs["random_select"] = bool(args.random_select)
        h5.flush()
    finally:
        h5.close()

    summary = {
        "games": games,
        "positions": positions,
        "skipped_moves": skipped_moves,
        "random_select": bool(args.random_select),
        "arch_type": arch_type,
        "state_encoding": spec.state_encoding,
        "target_schema": spec.target_schema,
        "output": args.output,
    }
    if spec.comment_advantage:
        summary["unannotated_advantages"] = unannotated_advantages
        summary["annotated_advantages"] = annotated_advantages
    print("preprocess summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PGN_PATH)
    parser.add_argument("--output", default=H5_PATH)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-opts", type=int, default=1)
    parser.add_argument(
        "--arch-type",
        choices=sorted(SUPPORTED_ARCH_TYPES),
        default=DEFAULT_ARCH_TYPE,
    )
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--random-select", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10000)
    return parser.parse_args()

if __name__ == "__main__":
    preprocess(parse_args())
