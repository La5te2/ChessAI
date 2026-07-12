import argparse
import os
import sys

import chess.pgn
import h5py
import numpy as np

from chess_env import board_to_packed
from config import NUM_ACTIONS
from move_encoder import move_to_index


def result_to_white_value(result: str) -> int:
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def expected_value(board, white_value: int) -> int:
    if white_value == 0:
        return 0
    return white_value if board.turn else -white_value


def print_summary(path):
    with h5py.File(path, "r") as h5:
        print("file:", path)
        print("size_gb:", f"{os.path.getsize(path) / 1024**3:.4f}")
        print("keys:", list(h5.keys()))
        for key in h5.keys():
            print(key, h5[key].shape, h5[key].dtype)
        print("attrs:", dict(h5.attrs))


def check_probability_datasets(path, count):
    errors = 0
    checked = 0
    names = ("policy", "target_policy", "mcts_policy", "teacher_policy")
    with h5py.File(path, "r") as h5:
        present = [name for name in names if name in h5]
        if not present:
            print("no dense policy datasets found")
            return True
        total_rows = min(int(h5[present[0]].shape[0]), int(count))
        for row_index in range(total_rows):
            for name in present:
                row = np.asarray(h5[name][row_index], dtype=np.float32)
                if row.shape != (NUM_ACTIONS,):
                    print(f"[ERROR] {name}[{row_index}] shape={row.shape}")
                    errors += 1
                    continue
                if not np.isfinite(row).all():
                    print(f"[ERROR] {name}[{row_index}] has non-finite values")
                    errors += 1
                    continue
                total = float(row.sum())
                if total > 0 and abs(total - 1.0) > 0.02:
                    print(f"[ERROR] {name}[{row_index}] probability sum={total:.6f}")
                    errors += 1
            checked += 1
            if errors >= 20:
                break
    print(f"probability check rows={checked}, errors={errors}")
    return errors == 0


def verify_against_pgn(h5_path, pgn_path, count):
    errors = 0
    checked = 0
    with h5py.File(h5_path, "r") as h5:
        for key in ("states", "moves", "values"):
            if key not in h5:
                raise KeyError(f"HDF5 missing dataset: {key}")

        states = h5["states"]
        moves = h5["moves"]
        values = h5["values"]
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as handle:
            while checked < count:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                board = game.board()
                white_value = result_to_white_value(game.headers.get("Result", "*"))
                for move in game.mainline_moves():
                    if checked >= count:
                        break
                    expected_move = move_to_index(move)
                    expected_state = board_to_packed(board)
                    expected_result = expected_value(board, white_value)
                    ok = True
                    if int(moves[checked]) != expected_move:
                        print(f"[ERROR] idx={checked} move mismatch")
                        ok = False
                    if not np.array_equal(states[checked], expected_state):
                        print(f"[ERROR] idx={checked} state mismatch")
                        ok = False
                    if int(values[checked]) != expected_result:
                        print(f"[ERROR] idx={checked} value mismatch")
                        ok = False
                    if not ok:
                        errors += 1
                        if errors >= 20:
                            return False
                    board.push(move)
                    checked += 1
    print(f"verify checked={checked}, errors={errors}")
    return errors == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect supervised or self-learning HDF5 files")
    parser.add_argument("path")
    parser.add_argument("--verify-pgn", default=None)
    parser.add_argument("--verify-count", type=int, default=1000)
    parser.add_argument("--check-probabilities", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_summary(args.path)
    ok = True
    if args.verify_pgn:
        ok = verify_against_pgn(args.path, args.verify_pgn, args.verify_count) and ok
    if args.check_probabilities:
        ok = check_probability_datasets(args.path, args.verify_count) and ok
    print("VERIFY OK" if ok else "VERIFY FAILED")
    if not ok:
        sys.exit(1)
