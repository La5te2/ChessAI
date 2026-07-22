from __future__ import annotations

import argparse
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

import chess
import chess.pgn

from teacher import UciTeacher, TeacherConfig, move_accuracy_from_regret


MATE_CP_THRESHOLD = 90000
UCI_BINARY = "stockfish.exe" if os.name == "nt" else "stockfish"
UCI_PATH = str(Path(__file__).resolve().parent.parent / "models" / "stockfish" / UCI_BINARY)


@dataclass
class MoveRow:
    ply: int
    move_label: str
    played_san: str
    played_uci: str
    best_label: str
    best_san: str
    best_uci: str
    best_cp: int
    played_cp: int
    regret_cp: int
    accuracy: float
    mark: str
    top_moves: List[str]


def cp_text(value: int) -> str:
    value = int(value)
    if abs(value) >= MATE_CP_THRESHOLD:
        return "+mate" if value > 0 else "-mate"
    return f"{value:+d}"


def white_cp_from_mover_cp(board: chess.Board, cp: int) -> int:
    return int(cp) if board.turn == chess.WHITE else -int(cp)


def pgn_eval_comment_from_white_cp(cp: int) -> str:
    cp = int(cp)
    if abs(cp) >= MATE_CP_THRESHOLD:
        pawns = 1000.0 if cp > 0 else -1000.0
    else:
        pawns = float(cp) / 100.0
    return f"{pawns:+.2f}"


def regret_mark(regret_cp: int) -> str:
    regret = int(regret_cp)
    if regret <= 30:
        return "ok"
    if regret <= 80:
        return "?!"
    if regret <= 200:
        return "?"
    return "??"


def side_prefix(board: chess.Board) -> str:
    return (
        f"{board.fullmove_number}."
        if board.turn == chess.WHITE
        else f"{board.fullmove_number}..."
    )


def san_or_uci(board: chess.Board, uci: Optional[str]) -> str:
    if not uci:
        return "-"
    try:
        move = chess.Move.from_uci(str(uci))
        if move in board.legal_moves:
            return board.san(move)
    except Exception:
        pass
    return str(uci)


def move_with_prefix(prefix: str, san: str) -> str:
    return f"{prefix}{san}"


def sorted_score_rows(result: Dict) -> List[tuple[str, int]]:
    scores = result.get("move_scores_cp") or {}
    return sorted(
        ((str(move), int(score)) for move, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def top_move_texts(board: chess.Board, result: Dict, count: int) -> List[str]:
    out = []
    for uci, cp in sorted_score_rows(result)[: max(0, int(count))]:
        out.append(f"{san_or_uci(board, uci)}({uci},{cp_text(cp)})")
    return out


def final_result_text(board: chess.Board, last_move_label: str) -> str:
    result = board.result(claim_draw=True)
    if result == "*":
        return "*"
    outcome = board.outcome(claim_draw=True)
    if outcome and outcome.termination == chess.Termination.CHECKMATE:
        return f"{result} by {last_move_label}"
    return result


def iter_games(handle: TextIO) -> Iterable[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return
        yield game


def render_pgn(game: chess.pgn.Game, columns: int = 88) -> str:
    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=False,
        comments=True,
        columns=None if int(columns) <= 0 else int(columns),
    )
    text = game.accept(exporter)
    if int(columns) <= 0:
        return text
    return wrap_pgn_comments(text, columns=int(columns))


def wrap_pgn_comments(text: str, columns: int) -> str:
    wrapped_lines = []
    width = max(40, int(columns) - 4)
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            wrapped_lines.append(line)
            continue
        content = stripped[1:-1].strip()
        content = re.sub(r",(?=\S)", ", ", content)
        parts = textwrap.wrap(
            content,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if len(parts) <= 1:
            wrapped_lines.append("{ " + content + " }")
            continue
        wrapped_lines.append("{ " + parts[0])
        for part in parts[1:-1]:
            wrapped_lines.append("  " + part)
        wrapped_lines.append("  " + parts[-1] + " }")
    return "\n".join(wrapped_lines)


def row_comment(row: MoveRow) -> str:
    if row.regret_cp >= MATE_CP_THRESHOLD:
        return "Decisive blunder: enters a forced mate line."
    if row.regret_cp > 300:
        return f"Large blunder; UCI engine prefers {row.best_label}."
    if row.regret_cp > 200:
        return f"Major defensive failure; UCI engine prefers {row.best_label}."
    if row.regret_cp > 80:
        return f"Clear inaccuracy; UCI engine prefers {row.best_label}."
    if row.regret_cp > 50:
        return f"Small but visible inaccuracy; UCI engine prefers {row.best_label}."
    return f"Minor difference; UCI engine slightly prefers {row.best_label}."


def analyse_game(
    game: chess.pgn.Game,
    game_number: int,
    teacher: UciTeacher,
    top_count: int,
    critical_threshold_cp: int,
    annotate_pgn: bool = False,
) -> tuple[str, Dict, chess.pgn.Game]:
    board = game.board()
    rows: List[MoveRow] = []
    last_move_label = "-"
    node = game

    if annotate_pgn:
        root_result = teacher.analyse(board, played_move=None)
        root_cp = white_cp_from_mover_cp(
            board,
            int(root_result.get("best_score_cp", 0)),
        )
        game.comment = pgn_eval_comment_from_white_cp(root_cp)

    for ply, move in enumerate(game.mainline_moves(), 1):
        child = node.variation(0)
        prefix = side_prefix(board)
        played_san = board.san(move)
        played_label = move_with_prefix(prefix, played_san)
        result = teacher.analyse(board, played_move=move)

        best_uci = str(result.get("best_move") or "")
        best_san = san_or_uci(board, best_uci)
        best_label = move_with_prefix(prefix, best_san)
        regret_cp = int(result.get("regret_cp", 0))
        row = MoveRow(
            ply=ply,
            move_label=played_label,
            played_san=played_san,
            played_uci=move.uci(),
            best_label=best_label,
            best_san=best_san,
            best_uci=best_uci,
            best_cp=int(result.get("best_score_cp", 0)),
            played_cp=int(result.get("played_score_cp", 0)),
            regret_cp=regret_cp,
            accuracy=float(move_accuracy_from_regret(regret_cp)),
            mark=regret_mark(regret_cp),
            top_moves=top_move_texts(board, result, top_count),
        )
        rows.append(row)
        last_move_label = played_label
        if annotate_pgn:
            child.comment = pgn_eval_comment_from_white_cp(
                white_cp_from_mover_cp(board, row.played_cp)
            )
        board.push(move)
        node = child

    text = render_game(
        game=game,
        game_number=game_number,
        rows=rows,
        final_board=board,
        last_move_label=last_move_label,
        critical_threshold_cp=critical_threshold_cp,
    )
    return text, summarize_rows(rows), game


def summarize_rows(rows: List[MoveRow]) -> Dict:
    moves = len(rows)
    total_regret = sum(row.regret_cp for row in rows)
    non_mate = [row.regret_cp for row in rows if row.regret_cp < MATE_CP_THRESHOLD]
    mean_regret = total_regret / max(1, moves)
    mean_non_mate = sum(non_mate) / max(1, len(non_mate))
    return {
        "moves": moves,
        "total_regret": total_regret,
        "mean_regret": mean_regret,
        "mean_non_mate_regret": mean_non_mate,
        "mean_accuracy": sum(row.accuracy for row in rows) / max(1, moves),
        "inaccuracies": sum(1 for row in rows if row.regret_cp > 50),
        "mistakes": sum(1 for row in rows if row.regret_cp > 150),
        "blunders": sum(1 for row in rows if row.regret_cp > 300),
        "mate_coded": [row for row in rows if row.regret_cp >= MATE_CP_THRESHOLD],
    }


def render_game(
    game: chess.pgn.Game,
    game_number: int,
    rows: List[MoveRow],
    final_board: chess.Board,
    last_move_label: str,
    critical_threshold_cp: int,
) -> str:
    summary = summarize_rows(rows)
    critical_rows = [
        row for row in rows if row.regret_cp >= max(0, int(critical_threshold_cp))
    ]
    biggest = max(rows, key=lambda row: row.regret_cp, default=None)

    lines: List[str] = []
    lines.append(f"Game {game_number}")
    lines.append("----")
    lines.append(f"White: {game.headers.get('White', '?')}")
    lines.append(f"Black: {game.headers.get('Black', '?')}")
    lines.append(f"Event: {game.headers.get('Event', '?')}")
    lines.append(f"PGN result tag: {game.headers.get('Result', '*')}")
    lines.append(f"Board result: {final_result_text(final_board, last_move_label)}")
    lines.append("")

    lines.append("Summary")
    lines.append("-------")
    lines.append(f"Plies analysed: {summary['moves']}")
    lines.append(f"Mean accuracy: {summary['mean_accuracy']:.1f}")
    lines.append(f"Raw mean regret: {summary['mean_regret']:.1f} cp")
    if summary["mate_coded"]:
        lines.append(
            "Mean regret excluding mate-coded rows: "
            f"about {summary['mean_non_mate_regret']:.1f} cp"
        )
    lines.append(f"Inaccuracies: {summary['inaccuracies']}")
    lines.append(f"Mistakes: {summary['mistakes']}")
    lines.append(f"Blunders: {summary['blunders']}")
    if summary["mate_coded"]:
        row = summary["mate_coded"][0]
        lines.append("")
        lines.append(
            f"Note: {row.move_label} is scored as {cp_text(row.played_cp)} by the UCI engine, "
            f"so its regret is encoded as {row.regret_cp} cp."
        )
        if len(summary["mate_coded"]) > 1:
            lines.append(
                f"There are {len(summary['mate_coded'])} mate-coded rows in this game."
            )
    lines.append("")

    lines.append("Main Reading")
    lines.append("------------")
    if biggest is None:
        lines.append("No moves were found in the PGN mainline.")
    elif biggest.regret_cp <= 50:
        lines.append("The mainline is very close to the UCI engine's preferred play.")
    else:
        lines.append(
            f"The largest swing is {biggest.move_label}, where the UCI engine prefers "
            f"{biggest.best_label}."
        )
        lines.append(
            f"That move scores {cp_text(biggest.played_cp)} instead of "
            f"{cp_text(biggest.best_cp)}, for {biggest.regret_cp} cp regret."
        )
        if biggest.regret_cp >= MATE_CP_THRESHOLD:
            lines.append("This is the decisive tactical failure of the game.")
    lines.append(
        "Rows marked ?!/?/?? are the main candidates for manual review."
    )
    lines.append("")

    lines.append("Critical Moves")
    lines.append("--------------")
    lines.append(
        "Ply  Move       Played  Best       Best CP  Played CP  Regret  Mark  Comment"
    )
    if critical_rows:
        for row in critical_rows:
            lines.append(
                f"{row.ply:03d}  {row.move_label:<10} {row.played_uci:<7} "
                f"{row.best_label:<10} {cp_text(row.best_cp):>7} "
                f"{cp_text(row.played_cp):>10} {row.regret_cp:>7} "
                f"  {row.mark:<2}    {row_comment(row)}"
            )
    else:
        lines.append("No move reached the critical threshold.")
    lines.append("")

    lines.append("Full Move Table")
    lines.append("---------------")
    lines.append(
        "Ply  Move       UCI    Best       Best UCI  Best CP  Played CP  Regret  Acc    Mark"
    )
    for row in rows:
        lines.append(
            f"{row.ply:03d}  {row.move_label:<10} {row.played_uci:<6} "
            f"{row.best_label:<10} {row.best_uci:<8} "
            f"{cp_text(row.best_cp):>7} {cp_text(row.played_cp):>10} "
            f"{row.regret_cp:>7} {row.accuracy:7.1f}  {row.mark}"
        )
    lines.append("")

    lines.append("Best Alternatives On Key Rows")
    lines.append("-----------------------------")
    if critical_rows:
        for row in critical_rows:
            top = ", ".join(row.top_moves) if row.top_moves else row.best_label
            lines.append(f"{row.move_label:<10} UCI: {top}")
    else:
        lines.append("No critical rows to list.")
    return "\n".join(lines)


def output_path_for(input_path: str, output_path: Optional[str]) -> Path:
    if output_path:
        return Path(output_path)
    return Path(input_path).with_suffix(".cmt")


def pgn_output_path_for(input_path: str, output_path: Optional[str]) -> Path:
    if output_path:
        return Path(output_path)
    path = Path(input_path)
    return path.with_name(f"{path.stem}_cmt.pgn")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse PGN mainlines with a UCI engine and write a .cmt report.",
    )
    parser.add_argument("--input", required=True, help="Path to a PGN file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .cmt path. Defaults to the input path with .cmt suffix.",
    )
    parser.add_argument(
        "--all-games",
        dest="all_games",
        action="store_true",
        default=True,
        help="Analyse every game in the PGN. This is the default.",
    )
    parser.add_argument(
        "--single-game",
        dest="all_games",
        action="store_false",
        help="Analyse one game selected by --game-index.",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=1,
        help="1-based game index used with --single-game.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Maximum number of games to analyse when analysing all games.",
    )
    parser.add_argument("--critical-threshold-cp", type=int, default=50)
    parser.add_argument("--top-moves", type=int, default=3)
    parser.add_argument("--pgn-comments", action="store_true", default=False)
    parser.add_argument("--pgn-output", default=None)
    parser.add_argument("--pgn-columns", type=int, default=88)

    parser.add_argument("--uci", default=UCI_PATH)
    parser.add_argument("--uci-depth", type=int, default=14)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-multipv", type=int, default=5)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--teacher-cache", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"PGN not found: {input_path}")

    out_path = output_path_for(str(input_path), args.output)
    os.makedirs(out_path.parent or ".", exist_ok=True)

    config = TeacherConfig(
        uci=args.uci,
        depth=args.uci_depth,
        movetime_ms=args.uci_movetime_ms,
        multipv=args.uci_multipv,
        threads=args.uci_threads,
        hash_mb=args.uci_hash_mb,
        cache_path=args.teacher_cache,
    )

    print(
        "pgn analysis start:",
        f"input={input_path}",
        f"output={out_path}",
        f"uci={args.uci}",
        f"uci_depth={args.uci_depth}",
        f"uci_movetime_ms={args.uci_movetime_ms}",
        f"uci_multipv={args.uci_multipv}",
        f"uci_threads={args.uci_threads}",
        flush=True,
    )

    reports: List[str] = []
    annotated_games: List[chess.pgn.Game] = []
    selected = 0
    totals = {
        "games": 0,
        "moves": 0,
        "total_regret": 0.0,
        "accuracy_sum": 0.0,
        "inaccuracies": 0,
        "mistakes": 0,
        "blunders": 0,
    }

    with open(input_path, "r", encoding="utf-8", errors="ignore") as handle:
        with UciTeacher(config) as teacher:
            for game_number, game in enumerate(iter_games(handle), 1):
                if not args.all_games and game_number != int(args.game_index):
                    continue
                report, summary, annotated_game = analyse_game(
                    game=game,
                    game_number=game_number,
                    teacher=teacher,
                    top_count=args.top_moves,
                    critical_threshold_cp=args.critical_threshold_cp,
                    annotate_pgn=args.pgn_comments,
                )
                reports.append(report)
                if args.pgn_comments:
                    annotated_games.append(annotated_game)
                selected += 1
                totals["games"] += 1
                totals["moves"] += summary["moves"]
                totals["total_regret"] += summary["total_regret"]
                totals["accuracy_sum"] += summary["mean_accuracy"]
                totals["inaccuracies"] += summary["inaccuracies"]
                totals["mistakes"] += summary["mistakes"]
                totals["blunders"] += summary["blunders"]
                print(
                    "pgn analysis game:",
                    f"game={game_number}",
                    f"moves={summary['moves']}",
                    f"mean_regret_cp={summary['mean_regret']:.1f}",
                    f"mean_accuracy={summary['mean_accuracy']:.1f}",
                    flush=True,
                )
                if args.max_games > 0 and selected >= int(args.max_games):
                    break
                if not args.all_games:
                    break

    if selected <= 0:
        raise RuntimeError("no PGN game selected")

    header = [
        f"PGN analysis: {input_path}",
        f"Engine: {args.uci}",
        (
            "Settings: "
            f"depth={args.uci_depth}, "
            f"movetime={args.uci_movetime_ms}, "
            f"multipv={args.uci_multipv}, "
            f"threads={args.uci_threads}"
        ),
        "",
    ]
    footer: List[str] = []
    if selected > 1:
        footer = [
            "",
            "Total",
            "-----",
            f"Games analysed: {int(totals['games'])}",
            f"Plies analysed: {int(totals['moves'])}",
            (
                "Raw mean regret: "
                f"{totals['total_regret'] / max(1, totals['moves']):.1f} cp"
            ),
            (
                "Mean accuracy: "
                f"{totals['accuracy_sum'] / max(1, totals['games']):.1f}"
            ),
            f"Inaccuracies: {int(totals['inaccuracies'])}",
            f"Mistakes: {int(totals['mistakes'])}",
            f"Blunders: {int(totals['blunders'])}",
        ]

    text = "\n".join(header + ["\n\n".join(reports)] + footer) + "\n"
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    print(f"pgn analysis saved: {out_path}", flush=True)

    if args.pgn_comments:
        pgn_out_path = pgn_output_path_for(str(input_path), args.pgn_output)
        os.makedirs(pgn_out_path.parent or ".", exist_ok=True)
        with open(pgn_out_path, "w", encoding="utf-8") as handle:
            for game in annotated_games:
                handle.write(render_pgn(game, columns=args.pgn_columns))
                handle.write("\n\n")
        print(f"annotated pgn saved: {pgn_out_path}", flush=True)


if __name__ == "__main__":
    main()
