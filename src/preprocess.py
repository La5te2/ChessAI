import argparse
import json
import os
import chess
import chess.pgn
import h5py
import numpy as np
from chess_env import board_to_packed
from move_encoder import move_to_index
from config import PGN_PATH, H5_PATH, INPUT_CHANNELS

def result_to_white_value(result: str) -> int:
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0

def create_h5(path: str, compression: str, compression_opts: int, chunk_size: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h5 = h5py.File(path, "w")
    kwargs = {}
    if compression != "none":
        kwargs["compression"] = compression
        if compression == "gzip":
            kwargs["compression_opts"] = compression_opts
        kwargs["shuffle"] = True

    states = h5.create_dataset(
        "states", shape=(0, INPUT_CHANNELS, 8), maxshape=(None, INPUT_CHANNELS, 8),
        chunks=(min(chunk_size, 8192), INPUT_CHANNELS, 8), dtype="uint8", **kwargs)
    moves = h5.create_dataset(
        "moves", shape=(0,), maxshape=(None,), chunks=(min(chunk_size, 8192),),
        dtype="uint16", **kwargs)
    values = h5.create_dataset(
        "values", shape=(0,), maxshape=(None,), chunks=(min(chunk_size, 8192),),
        dtype="int8", **kwargs)
    h5.attrs["state_format"] = "packbits_18x64_to_18x8_uint8"
    h5.attrs["move_encoding"] = "alphazero_64x73"
    h5.attrs["value_perspective"] = "side_to_move"
    return h5, states, moves, values

def append_chunk(states_ds, moves_ds, values_ds, states_buf, moves_buf, values_buf):
    n = len(moves_buf)
    if n == 0:
        return
    old = states_ds.shape[0]
    new = old + n
    states_ds.resize(new, axis=0)
    moves_ds.resize(new, axis=0)
    values_ds.resize(new, axis=0)
    states_ds[old:new] = np.asarray(states_buf, dtype=np.uint8)
    moves_ds[old:new] = np.asarray(moves_buf, dtype=np.uint16)
    values_ds[old:new] = np.asarray(values_buf, dtype=np.int8)

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
    h5, states_ds, moves_ds, values_ds = create_h5(
        args.output, args.compression, args.compression_opts, args.chunk_size)
    states_buf, moves_buf, values_buf = [], [], []
    games = positions = skipped_moves = 0

    def live_positions():
        return positions + len(moves_buf)

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
        if len(moves_buf) == 0:
            return
        append_chunk(states_ds, moves_ds, values_ds, states_buf, moves_buf, values_buf)
        positions += len(moves_buf)
        states_buf.clear(); moves_buf.clear(); values_buf.clear()

    def process_game(game):
        nonlocal skipped_moves
        white_value = result_to_white_value(game.headers.get("Result", "*"))
        board = game.board()

        for move in game.mainline_moves():
            try:
                action = move_to_index(move)
                value = white_value if board.turn == chess.WHITE else -white_value
                states_buf.append(board_to_packed(board))
                moves_buf.append(action)
                values_buf.append(value)
            except Exception:
                skipped_moves += 1
            board.push(move)

            if len(moves_buf) >= args.chunk_size:
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
        "output": args.output,
    }
    print("preprocess summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=PGN_PATH)
    parser.add_argument("--output", default=H5_PATH)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-opts", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--random-select", action="store_true", default=False)
    parser.add_argument("--log-every", type=int, default=10000)
    return parser.parse_args()

if __name__ == "__main__":
    preprocess(parse_args())
