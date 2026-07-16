"""Polyglot opening book sampling for arena and offline-pv evaluation."""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn
import chess.polyglot

from config import UCI_PATH


POLYGLOT_ENTRY_STRUCT = struct.Struct(">QHHI")
MATE_SCORE_CP = 100000
GENERATED_BOOK_PATH = "data/openings.gen.bin"
DEFAULT_MIN_FENS = 50000


@dataclass(frozen=True)
class PolyglotOutputEntry:
    key: int
    raw_move: int
    weight: int
    learn: int = 0


def load_polyglot_positions(
    path: str,
    book_plies: int = 8,
    max_positions: int = 50000,
    seed: int = 2026,
) -> List[str]:
    book_path = Path(path)
    if book_path.suffix.lower() != ".bin":
        raise ValueError(f"opening book must be a Polyglot .bin file: {path}")
    if not book_path.exists():
        raise FileNotFoundError(f"opening book not found: {path}")

    target = max(1, int(max_positions))
    max_plies = max(1, int(book_plies))
    rng = random.Random(int(seed))
    positions = []
    seen_positions = set()
    visited = set()
    queue = deque([(chess.Board(), 0)])

    with chess.polyglot.open_reader(str(book_path)) as reader:
        while queue and len(positions) < target:
            board, plies = queue.popleft()
            fen = board.fen()
            state_key = opening_state_key(fen)
            visit_key = (state_key, int(plies))
            if visit_key in visited:
                continue
            visited.add(visit_key)

            entries = []
            if plies < max_plies:
                entries = [
                    entry
                    for entry in reader.find_all(board)
                    if entry.move in board.legal_moves
                ]
                rng.shuffle(entries)

            if plies > 0 and (plies >= max_plies or not entries):
                if (
                    state_key not in seen_positions
                    and not board.is_game_over(claim_draw=True)
                ):
                    positions.append(fen)
                    seen_positions.add(state_key)
                continue

            for entry in entries:
                child = board.copy(stack=False)
                child.push(entry.move)
                if not child.is_game_over(claim_draw=True):
                    queue.append((child, plies + 1))

    if not positions:
        positions.append(chess.Board().fen())
    return positions


def opening_state_key(fen: str) -> str:
    board = chess.Board(fen)
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep}"


def unique_position_fens(fens: List[str]) -> List[str]:
    seen = set()
    unique = []
    for fen in fens:
        key = opening_state_key(fen)
        if key in seen:
            continue
        seen.add(key)
        unique.append(fen)
    return unique


def _entry_key(entry) -> Tuple[int, int, int, int]:
    return (
        int(entry.key),
        int(entry.raw_move),
        int(entry.weight),
        int(entry.learn),
    )


def _record_book_move(moves_by_key, key: int, raw_move: int, move: chess.Move):
    moves_by_key.setdefault(int(key), {})[int(raw_move)] = move


def expanded_position_count_from_moves(
    moves_by_key,
    book_plies: int = 8,
    min_positions: int = 50000,
) -> int:
    target = max(1, int(min_positions))
    max_plies = max(1, int(book_plies))
    positions = []
    seen_positions = set()
    visited = set()
    queue = deque([(chess.Board(), 0)])

    while queue and len(positions) < target:
        board, plies = queue.popleft()
        fen = board.fen()
        state_key = opening_state_key(fen)
        visit_key = (state_key, int(plies))
        if visit_key in visited:
            continue
        visited.add(visit_key)

        moves = []
        if plies < max_plies:
            key = chess.polyglot.zobrist_hash(board)
            moves = [
                move
                for move in moves_by_key.get(int(key), {}).values()
                if move in board.legal_moves
            ]

        if plies > 0 and (plies >= max_plies or not moves):
            if (
                state_key not in seen_positions
                and not board.is_game_over(claim_draw=True)
            ):
                positions.append(fen)
                seen_positions.add(state_key)
            continue

        for move in moves:
            child = board.copy(stack=False)
            child.push(move)
            if not child.is_game_over(claim_draw=True):
                queue.append((child, plies + 1))

    return len(positions)


def refresh_readable_fens(
    moves_by_key,
    current: int,
    book_plies: int,
    min_fens: int,
) -> int:
    if not moves_by_key:
        return int(current)
    readable = expanded_position_count_from_moves(
        moves_by_key,
        book_plies=book_plies,
        min_positions=min_fens,
    )
    return max(int(current), int(readable))


def _commit_path_entries(path_entries, accepted_entries, accepted_keys, moves_by_key=None):
    for entry in path_entries:
        key = _entry_key(entry)
        if key not in accepted_keys:
            accepted_keys.add(key)
            accepted_entries.append(entry)
        if moves_by_key is not None:
            _record_book_move(
                moves_by_key,
                int(entry.key),
                int(entry.raw_move),
                entry.move,
            )


def _commit_endpoint(
    fen: str,
    path_entries,
    endpoint_fens,
    accepted_entries,
    accepted_keys,
    moves_by_key=None,
):
    if fen in endpoint_fens:
        return False
    endpoint_fens.add(fen)
    _commit_path_entries(path_entries, accepted_entries, accepted_keys, moves_by_key)
    return True


class OpeningBook:
    def __init__(
        self,
        path: str,
        book_plies: int = 8,
        max_positions: int = 50000,
        seed: int = 2026,
    ):
        self.path = str(path)
        self.positions = load_polyglot_positions(
            path=self.path,
            book_plies=book_plies,
            max_positions=max_positions,
            seed=seed,
        )

    def __len__(self):
        return len(self.positions)


def make_arena_specs(
    games: int,
    seed: int = 2026,
    opening_book: str = "data/openings.bin",
    book_plies: int = 8,
    max_positions: int = 50000,
    paired: bool = True,
) -> List[Tuple[str, chess.Color]]:
    games = int(games)
    if games <= 0:
        return []

    if not opening_book:
        start_fen = chess.Board().fen()
        specs: List[Tuple[str, chess.Color]] = []
        for index in range(games):
            specs.append((start_fen, chess.WHITE if index % 2 == 0 else chess.BLACK))
        return specs

    book = OpeningBook(
        path=opening_book,
        book_plies=book_plies,
        max_positions=max(int(max_positions), (games + 1) // 2 if paired else games),
        seed=seed,
    )

    needed = (games + 1) // 2 if paired else games
    selected = unique_position_fens(list(book.positions))
    if len(selected) < needed:
        raise ValueError(
            "arena requires enough unique opening states: "
            f"games={games}, paired={paired}, required={needed}, "
            f"unique_openings={len(selected)}"
        )
    rng = random.Random(int(seed))
    if len(selected) > 1:
        rng.shuffle(selected)
    selected = selected[:needed]

    specs: List[Tuple[str, chess.Color]] = []
    for index, fen in enumerate(selected):
        board = chess.Board(fen)
        if paired:
            specs.append((fen, board.turn))
            if len(specs) < games:
                specs.append((fen, not board.turn))
        else:
            specs.append((fen, chess.WHITE if index % 2 == 0 else chess.BLACK))
        if len(specs) >= games:
            break
    return specs[:games]


def default_balanced_output(path: str) -> str:
    book_path = Path(path)
    return str(book_path.with_name(f"{book_path.stem}.balanced{book_path.suffix}"))


def encode_polyglot_move(board: chess.Board, move: chess.Move) -> int:
    encoded = board._to_chess960(move)
    promotion_part = int(encoded.promotion) - 1 if encoded.promotion else 0
    return (
        int(encoded.to_square)
        | (int(encoded.from_square) << 6)
        | (promotion_part << 12)
    )


def engine_limit(depth: int, movetime_ms: int):
    kwargs = {}
    if int(depth) > 0:
        kwargs["depth"] = int(depth)
    if int(movetime_ms) > 0:
        kwargs["time"] = float(movetime_ms) / 1000.0
    if not kwargs:
        kwargs["depth"] = 12
    return chess.engine.Limit(**kwargs)


def score_cp_white(info) -> Optional[int]:
    score = info.get("score")
    if score is None:
        return None
    try:
        return int(score.pov(chess.WHITE).score(mate_score=MATE_SCORE_CP))
    except Exception:
        return None


def analyse_book_position(engine, board: chess.Board, limit) -> Optional[int]:
    info = engine.analyse(board, limit)
    return score_cp_white(info)


def configure_uci_engine(engine, threads: int, hash_mb: int):
    options = {}
    if int(threads) > 0:
        options["Threads"] = int(threads)
    if int(hash_mb) > 0:
        options["Hash"] = int(hash_mb)
    if options:
        try:
            engine.configure(options)
        except Exception:
            pass


def write_polyglot_book(path: str, entries):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(f"{output.suffix}.tmp")

    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry.key),
            int(entry.raw_move),
            int(entry.weight),
            int(entry.learn),
        ),
    )
    with open(tmp, "wb") as handle:
        for entry in ordered:
            handle.write(
                POLYGLOT_ENTRY_STRUCT.pack(
                    int(entry.key),
                    int(entry.raw_move),
                    int(entry.weight),
                    int(entry.learn),
                )
            )
    os.replace(tmp, output)


def validate_written_book(path: str, book_plies: int, min_fens: int) -> int:
    readable_fens = len(
        load_polyglot_positions(
            path,
            book_plies=book_plies,
            max_positions=min_fens,
        )
    )
    if readable_fens < min_fens:
        raise RuntimeError(
            f"written opening book expanded positions below --min-fens: {readable_fens} < {min_fens}"
        )
    return readable_fens


def verify_opening_book(args):
    source = Path(args.verify)
    if source.suffix.lower() != ".bin":
        raise ValueError(f"opening book must be a Polyglot .bin file: {source}")
    if not source.exists():
        raise FileNotFoundError(f"opening book not found: {source}")
    if not os.path.exists(args.uci):
        raise FileNotFoundError(f"UCI engine not found: {args.uci}")

    output = args.output or default_balanced_output(str(source))
    if Path(output).resolve() == source.resolve():
        raise ValueError("output must differ from source; use --in-place without --output")

    limit = engine_limit(args.uci_depth, args.uci_movetime_ms)
    max_abs_cp = max(0, int(args.max_abs_cp))
    min_fens = max(1, int(args.min_fens))
    max_plies = max(1, int(args.book_plies))
    min_weight = max(0, int(args.min_weight))
    log_every = max(0, int(args.log_every))

    print(
        "opening verify start:",
        f"input={source}",
        f"output={output}",
        f"uci={args.uci}",
        f"book_plies={max_plies}",
        f"min_fens={min_fens}",
        f"max_abs_cp={max_abs_cp}",
        flush=True,
    )

    accepted_entries = []
    accepted_keys = set()
    moves_by_key = {}
    endpoint_fens = set()
    visited_positions = set()
    queue = deque([(chess.Board(), 0, [])])
    readable_fens = 0
    checked = 0
    rejected = 0
    illegal = 0
    terminal = 0
    unknown = 0
    start = time.monotonic()

    with chess.engine.SimpleEngine.popen_uci(args.uci) as engine:
        configure_uci_engine(engine, args.uci_threads, args.uci_hash_mb)

        with chess.polyglot.open_reader(str(source)) as reader:
            while queue and readable_fens < min_fens:
                board, ply, path_entries = queue.popleft()
                fen = board.fen()
                visit_key = (fen, int(ply))
                if visit_key in visited_positions:
                    continue
                visited_positions.add(visit_key)
                if ply >= max_plies:
                    if ply > 0:
                        _commit_endpoint(
                            fen,
                            path_entries,
                            endpoint_fens,
                            accepted_entries,
                            accepted_keys,
                            moves_by_key,
                        )
                        if len(endpoint_fens) >= min_fens:
                            readable_fens = refresh_readable_fens(
                                moves_by_key,
                                readable_fens,
                                book_plies=max_plies,
                                min_fens=min_fens,
                            )
                    continue

                entries = list(reader.find_all(board, minimum_weight=min_weight))
                accepted_child = False
                for entry in entries:
                    checked += 1
                    if entry.move not in board.legal_moves:
                        illegal += 1
                        continue

                    child = board.copy(stack=False)
                    child.push(entry.move)
                    if child.is_game_over(claim_draw=True):
                        terminal += 1
                        rejected += 1
                        continue

                    cp = analyse_book_position(engine, child, limit)
                    if cp is None:
                        unknown += 1
                        rejected += 1
                        continue

                    if abs(cp) <= max_abs_cp:
                        accepted_child = True
                        child_path = path_entries + [entry]
                        child_fen = child.fen()
                        if ply + 1 >= max_plies:
                            _commit_endpoint(
                                child_fen,
                                child_path,
                                endpoint_fens,
                                accepted_entries,
                                accepted_keys,
                                moves_by_key,
                            )
                            if len(endpoint_fens) >= min_fens:
                                readable_fens = refresh_readable_fens(
                                    moves_by_key,
                                    readable_fens,
                                    book_plies=max_plies,
                                    min_fens=min_fens,
                                )
                            if readable_fens >= min_fens:
                                queue.clear()
                                break
                        else:
                            queue.append((child, ply + 1, child_path))
                    else:
                        rejected += 1

                    if log_every and checked % log_every == 0:
                        readable_fens = refresh_readable_fens(
                            moves_by_key,
                            readable_fens,
                            book_plies=max_plies,
                            min_fens=min_fens,
                        )
                        print(
                            "opening verify:",
                            f"checked={checked}",
                            f"accepted_entries={len(accepted_entries)}",
                            f"accepted_fens={len(endpoint_fens)}",
                            f"readable_fens={readable_fens}/{min_fens}",
                            f"rejected={rejected}",
                            f"positions={len(visited_positions)}",
                            f"queue={len(queue)}",
                            flush=True,
                        )
                if not accepted_child and ply > 0:
                    _commit_endpoint(
                        fen,
                        path_entries,
                        endpoint_fens,
                        accepted_entries,
                        accepted_keys,
                        moves_by_key,
                    )
                    if len(endpoint_fens) >= min_fens:
                        readable_fens = refresh_readable_fens(
                            moves_by_key,
                            readable_fens,
                            book_plies=max_plies,
                            min_fens=min_fens,
                        )

    if not accepted_entries:
        raise RuntimeError("no balanced opening entries found")
    if readable_fens < min_fens:
        readable_fens = refresh_readable_fens(
            moves_by_key,
            readable_fens,
            book_plies=max_plies,
            min_fens=min_fens,
        )
    if readable_fens < min_fens:
        raise RuntimeError(
            f"opening book expanded positions below --min-fens: {readable_fens} < {min_fens}"
        )

    write_polyglot_book(output, accepted_entries)
    readable_fens = validate_written_book(output, max_plies, min_fens)

    if args.in_place:
        backup = source.with_suffix(f"{source.suffix}.bak_verified")
        os.replace(source, backup)
        os.replace(output, source)
        output = str(source)
    summary = {
        "input": str(source),
        "output": str(output),
        "checked_entries": checked,
        "accepted_entries": len(accepted_entries),
        "rejected_entries": rejected,
        "visited_positions": len(visited_positions),
        "accepted_fens": len(endpoint_fens),
        "readable_fens": int(readable_fens),
        "min_fens": int(min_fens),
        "illegal_entries": illegal,
        "terminal_entries": terminal,
        "unknown_score_entries": unknown,
        "book_plies": max_plies,
        "max_abs_cp": max_abs_cp,
        "uci_depth": int(args.uci_depth),
        "uci_movetime_ms": int(args.uci_movetime_ms),
        "elapsed_sec": round(time.monotonic() - start, 3),
    }
    print("opening verify summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def generate_opening_book_from_pgn(args):
    source = Path(args.pgn)
    if not source.exists():
        raise FileNotFoundError(f"PGN not found: {source}")
    if not os.path.exists(args.uci):
        raise FileNotFoundError(f"UCI engine not found: {args.uci}")

    output = args.output or GENERATED_BOOK_PATH
    limit = engine_limit(args.uci_depth, args.uci_movetime_ms)
    max_abs_cp = max(0, int(args.max_abs_cp))
    min_fens = max(1, int(args.min_fens))
    max_plies = max(1, int(args.book_plies))
    log_every = max(0, int(args.log_every))

    print(
        "opening pgn start:",
        f"input={source}",
        f"output={output}",
        f"uci={args.uci}",
        f"book_plies={max_plies}",
        f"min_fens={min_fens}",
        f"max_abs_cp={max_abs_cp}",
        flush=True,
    )

    counts: Dict[Tuple[int, int], int] = {}
    moves_by_key = {}
    games = 0
    checked = 0
    accepted_moves = 0
    accepted_lines = 0
    rejected = 0
    illegal = 0
    terminal = 0
    unknown = 0
    endpoint_fens = set()
    readable_fens = 0
    stop = False
    start = time.monotonic()

    with chess.engine.SimpleEngine.popen_uci(args.uci) as engine:
        configure_uci_engine(engine, args.uci_threads, args.uci_hash_mb)
        with open(source, "r", encoding="utf-8", errors="replace") as handle:
            while not stop:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games += 1
                board = game.board()
                path_edges = []
                line_ok = True

                for ply, move in enumerate(game.mainline_moves(), 1):
                    if ply > max_plies:
                        break
                    if move not in board.legal_moves:
                        illegal += 1
                        line_ok = False
                        break

                    checked += 1
                    child = board.copy(stack=False)
                    child.push(move)
                    if child.is_game_over(claim_draw=True):
                        terminal += 1
                        rejected += 1
                        line_ok = False
                        break

                    cp = analyse_book_position(engine, child, limit)
                    if cp is None:
                        unknown += 1
                        rejected += 1
                        line_ok = False
                        break

                    if abs(cp) <= max_abs_cp:
                        key = chess.polyglot.zobrist_hash(board)
                        raw_move = encode_polyglot_move(board, move)
                        path_edges.append((int(key), int(raw_move), move))
                        board = child
                    else:
                        rejected += 1
                        line_ok = False
                        break

                    if log_every and checked % log_every == 0:
                        readable_fens = refresh_readable_fens(
                            moves_by_key,
                            readable_fens,
                            book_plies=max_plies,
                            min_fens=min_fens,
                        )
                        print(
                            "opening pgn:",
                            f"games={games}",
                            f"checked={checked}",
                            f"accepted_lines={accepted_lines}",
                            f"accepted_fens={len(endpoint_fens)}",
                            f"readable_fens={readable_fens}/{min_fens}",
                            f"unique_entries={len(counts)}",
                            f"rejected={rejected}",
                            flush=True,
                        )
                if line_ok and path_edges and not board.is_game_over(claim_draw=True):
                    fen = board.fen()
                    if fen not in endpoint_fens:
                        endpoint_fens.add(fen)
                        accepted_lines += 1
                        accepted_moves += len(path_edges)
                        for key, raw_move, move in path_edges:
                            edge = (key, raw_move)
                            counts[edge] = counts.get(edge, 0) + 1
                            _record_book_move(moves_by_key, key, raw_move, move)
                        refresh_interval = max(1, min(100, min_fens // 50))
                        if (
                            len(endpoint_fens) >= min_fens
                            or accepted_lines % refresh_interval == 0
                        ):
                            readable_fens = refresh_readable_fens(
                                moves_by_key,
                                readable_fens,
                                book_plies=max_plies,
                                min_fens=min_fens,
                            )
                        if readable_fens >= min_fens:
                            stop = True

    if not counts:
        raise RuntimeError("no balanced PGN opening entries found")
    if readable_fens < min_fens:
        readable_fens = refresh_readable_fens(
            moves_by_key,
            readable_fens,
            book_plies=max_plies,
            min_fens=min_fens,
        )
    if readable_fens < min_fens:
        raise RuntimeError(
            f"opening book expanded positions below --min-fens: {readable_fens} < {min_fens}"
        )

    entries = [
        PolyglotOutputEntry(
            key=key,
            raw_move=raw_move,
            weight=min(65535, max(1, int(weight))),
        )
        for (key, raw_move), weight in counts.items()
    ]
    write_polyglot_book(output, entries)
    readable_fens = validate_written_book(output, max_plies, min_fens)

    summary = {
        "input": str(source),
        "output": str(output),
        "games": int(games),
        "checked_positions": int(checked),
        "accepted_lines": int(accepted_lines),
        "accepted_moves": int(accepted_moves),
        "accepted_fens": len(endpoint_fens),
        "readable_fens": int(readable_fens),
        "min_fens": int(min_fens),
        "unique_entries": len(entries),
        "rejected_positions": int(rejected),
        "illegal_moves": int(illegal),
        "terminal_positions": int(terminal),
        "unknown_score_positions": int(unknown),
        "book_plies": int(max_plies),
        "max_abs_cp": int(max_abs_cp),
        "uci_depth": int(args.uci_depth),
        "uci_movetime_ms": int(args.uci_movetime_ms),
        "elapsed_sec": round(time.monotonic() - start, 3),
    }
    print("opening pgn summary:", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Polyglot opening book sampling and UCI verification"
    )
    parser.add_argument("--verify", default=None)
    parser.add_argument("--pgn", default=None)
    parser.add_argument("--uci", default=UCI_PATH)
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true", default=False)
    parser.add_argument("--max-abs-cp", type=int, default=80)
    parser.add_argument("--min-fens", type=int, default=DEFAULT_MIN_FENS)
    parser.add_argument("--book-plies", type=int, default=8)
    parser.add_argument("--min-weight", type=int, default=1)
    parser.add_argument("--uci-depth", type=int, default=12)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.verify:
        verify_opening_book(args)
        return
    if args.pgn:
        generate_opening_book_from_pgn(args)
        return
    raise SystemExit(
        "usage: python src/opening_book.py --verify data/openings.bin --uci <engine>"
    )


if __name__ == "__main__":
    main()
