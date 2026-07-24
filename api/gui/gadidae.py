"""Gadidae desktop GUI.

This single module contains:
1. shared chess termination rules;
2. shared Tk board presentation;
3. UCI protocol helpers;
4. the Stadium two-engine match mode;
5. the Simulator position-analysis mode;
6. the unified application shell and command-line entry point.

Neural-network inference remains in the external Gadus and Melano UCI engines.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import io
import json
import multiprocessing as mp
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chess
import chess.engine
import chess.pgn
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


# =============================================================================
# Shared chess rules
# =============================================================================

def completed_claim_termination(
    board: chess.Board,
) -> Optional[chess.Termination]:
    """Returns a claim-based draw only after its board condition is reached."""
    if board.is_repetition(3):
        return chess.Termination.THREEFOLD_REPETITION
    if board.is_fifty_moves():
        return chess.Termination.FIFTY_MOVES
    return None


def game_is_over(board: chess.Board) -> bool:
    return bool(
        board.is_game_over(claim_draw=False)
        or completed_claim_termination(board) is not None
    )


def game_result(board: chess.Board) -> str:
    result = board.result(claim_draw=False)
    if result != "*":
        return result
    if completed_claim_termination(board) is not None:
        return "1/2-1/2"
    return "*"


def game_termination(board: chess.Board) -> Optional[chess.Termination]:
    outcome = board.outcome(claim_draw=False)
    if outcome is not None:
        return outcome.termination
    return completed_claim_termination(board)


def game_termination_text(board: chess.Board) -> Optional[str]:
    termination = game_termination(board)
    if termination is None:
        return None
    return termination.name.lower().replace("_", " ")


# =============================================================================
# Shared Tk chessboard presentation
# =============================================================================

UNICODE_PIECES = {
    chess.Piece(chess.PAWN, chess.WHITE): "♙",
    chess.Piece(chess.KNIGHT, chess.WHITE): "♘",
    chess.Piece(chess.BISHOP, chess.WHITE): "♗",
    chess.Piece(chess.ROOK, chess.WHITE): "♖",
    chess.Piece(chess.QUEEN, chess.WHITE): "♕",
    chess.Piece(chess.KING, chess.WHITE): "♔",
    chess.Piece(chess.PAWN, chess.BLACK): "♟",
    chess.Piece(chess.KNIGHT, chess.BLACK): "♞",
    chess.Piece(chess.BISHOP, chess.BLACK): "♝",
    chess.Piece(chess.ROOK, chess.BLACK): "♜",
    chess.Piece(chess.QUEEN, chess.BLACK): "♛",
    chess.Piece(chess.KING, chess.BLACK): "♚",
}


class ChessGUIBase:
    root: tk.Tk
    engine: object
    canvas: tk.Canvas
    board_state_text: tk.Text

    def initialize_board_gui(self):
        self.square_size = 72
        self.board_size = self.square_size * 8
        self.flipped = False
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move = self.engine.last_move()
        self.buttons: List[ttk.Button] = []

        self.light = "#EEEED2"
        self.dark = "#769656"
        self.selected_color = "#F6F669"
        self.target_color = "#BACA44"
        self.lastmove_color = "#CDD26A"

    def _add_button(self, parent, text, command, side=tk.LEFT):
        button = ttk.Button(parent, text=text, command=command)
        button.pack(side=side, padx=2)
        self.buttons.append(button)
        return button

    def square_to_screen(self, square: chess.Square):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.flipped:
            return 7 - file_index, rank_index
        return file_index, 7 - rank_index

    def screen_to_square(self, x: int, y: int):
        column = x // self.square_size
        row = y // self.square_size
        if not (0 <= column < 8 and 0 <= row < 8):
            return None
        if self.flipped:
            file_index = 7 - column
            rank_index = row
        else:
            file_index = column
            rank_index = 7 - row
        return chess.square(file_index, rank_index)

    def clear_selection(self):
        self.selected_square = None
        self.legal_targets = []

    def sync_last_move(self):
        self.last_move = self.engine.last_move()

    def draw_board(self):
        self.canvas.delete("all")
        highlighted = set()
        if self.last_move is not None:
            highlighted = {
                self.last_move.from_square,
                self.last_move.to_square,
            }

        for square in chess.SQUARES:
            column, row = self.square_to_screen(square)
            x1 = column * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size

            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            color = (
                self.light
                if (file_index + rank_index) % 2 == 0
                else self.dark
            )
            if square in highlighted:
                color = self.lastmove_color
            if square == self.selected_square:
                color = self.selected_color
            elif square in self.legal_targets:
                color = self.target_color

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color,
                outline=color,
            )

            piece = self.engine.board.piece_at(square)
            if piece:
                self.canvas.create_text(
                    x1 + self.square_size / 2,
                    y1 + self.square_size / 2 + 2,
                    text=UNICODE_PIECES.get(piece, piece.symbol()),
                    font=("Segoe UI Symbol", int(self.square_size * 0.62)),
                    fill="#111111",
                )

        self.update_panels()

    def update_board_state(self):
        text = (
            f"FEN:\n{self.engine.board.fen()}\n\n"
            f"PGN:\n{self.engine.pgn_movetext()}"
        )
        self.board_state_text.configure(state=tk.NORMAL)
        self.board_state_text.delete("1.0", tk.END)
        self.board_state_text.insert(tk.END, text)
        self.board_state_text.configure(state=tk.DISABLED)

    def flip_board(self):
        self.flipped = not self.flipped
        self.draw_board()

    def save_pgn(self):
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save PGN",
            defaultextension=".pgn",
            filetypes=[
                ("PGN files", "*.pgn"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.engine.save_pgn(path)
        except Exception as exc:
            messagebox.showerror("Save PGN", str(exc), parent=self.root)


# =============================================================================
# UCI protocol helpers
# =============================================================================

def strip_wrapping_quotes(text: str) -> str:
    """Removes straight or typographic quotes around user-entered paths and FENs."""
    value = str(text).strip()
    quote_pairs = {
        '"': '"',
        "'": "'",
        "“": "”",
        "‘": "’",
    }
    if len(value) >= 2 and value[0] in quote_pairs:
        if value[-1] == quote_pairs[value[0]]:
            return value[1:-1].strip()
    return value


def uci_options_from_text(text: str) -> Dict[str, object]:
    value = str(text or "").strip()
    if not value:
        return {}
    options = json.loads(value)
    if not isinstance(options, dict):
        raise ValueError("UCI options must be a JSON object")
    for name, option_value in options.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("UCI option names must be non-empty strings")
        if option_value is None or isinstance(option_value, (dict, list)):
            raise ValueError(f"unsupported UCI option value for {name!r}")
    return options



def safe_san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


def command_from_text(text: str) -> List[str]:
    value = str(text).strip()
    if not value:
        raise ValueError("UCI command is empty")
    if os.name != "nt":
        return shlex.split(value)

    argc = ctypes.c_int()
    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(ctypes.c_wchar_p)
    argv = command_line_to_argv(value, ctypes.byref(argc))
    if not argv:
        raise ValueError(f"could not parse UCI command: {value}")
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


def popen_uci_engine(command: Sequence[str]):
    popen_args = {}
    if os.name == "nt":
        popen_args["creationflags"] = subprocess.CREATE_NO_WINDOW
    return chess.engine.SimpleEngine.popen_uci(list(command), **popen_args)


def score_text(score, turn: chess.Color) -> str:
    if score is None:
        return "?"
    relative = score.pov(turn)
    mate = relative.mate()
    if mate is not None:
        return f"#{mate:+d}"
    cp = relative.score()
    return "?" if cp is None else f"{cp / 100.0:+.2f}"


def pv_text(board: chess.Board, pv: Sequence[chess.Move], limit: int = 8) -> str:
    line = board.copy(stack=False)
    sans = []
    for move in list(pv)[: max(0, int(limit))]:
        if move not in line.legal_moves:
            break
        sans.append(line.san(move))
        line.push(move)
    return " ".join(sans)


def engine_multipv_count(
    engine: chess.engine.SimpleEngine,
    requested: int,
) -> int:
    option = engine.options.get("MultiPV")
    if option is None:
        return 1
    count = int(requested)
    if option.min is not None:
        count = max(int(option.min), count)
    if option.max is not None:
        count = min(int(option.max), count)
    return max(1, count)


def configurable_uci_options(options: Dict[str, object]) -> Dict[str, object]:
    managed = {name.lower() for name in chess.engine.MANAGED_OPTIONS}
    return {
        name: value
        for name, value in options.items()
        if str(name).lower() not in managed
    }


def primary_info_for_move(infos: Sequence[Dict], move: chess.Move) -> Dict:
    for info in infos:
        pv = list(info.get("pv") or [])
        if pv and pv[0] == move:
            return dict(info)
    for info in infos:
        if int(info.get("multipv", 1) or 1) == 1:
            return dict(info)
    return dict(infos[0]) if infos else {}


def multipv_move_rows(
    board: chess.Board,
    infos: Sequence[Dict],
    selected_move: chess.Move,
) -> List[str]:
    rows = []
    ordered = sorted(
        infos,
        key=lambda info: int(info.get("multipv", 1) or 1),
    )
    for fallback_rank, info in enumerate(ordered, 1):
        pv = list(info.get("pv") or [])
        if not pv or pv[0] not in board.legal_moves:
            continue
        move = pv[0]
        rank = int(info.get("multipv", fallback_rank) or fallback_rank)
        marker = "*" if move == selected_move else " "
        score = score_text(info.get("score"), board.turn)
        line = pv_text(board, pv)
        text = f"{marker}{rank}. {safe_san(board, move)} ({move.uci()})  score={score}"
        if line:
            text += f"  pv={line}"
        rows.append(text)
    return rows


def analyse_uci_turn(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    movetime_ms: int,
    multipv: int,
    nodes: Optional[int] = None,
    progress_callback=None,
    stop_event=None,
    update_interval_ms: int = 100,
):
    last_update = 0.0
    limit = chess.engine.Limit(
        time=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        nodes=int(nodes) if nodes is not None and int(nodes) > 0 else None,
    )
    analysis_options = {"info": chess.engine.INFO_ALL}
    if engine.options.get("MultiPV") is not None:
        analysis_options["multipv"] = max(1, int(multipv))
    with engine.analysis(
        board,
        limit,
        **analysis_options,
    ) as analysis:
        for _ in analysis:
            if stop_event is not None and stop_event.is_set():
                analysis.stop()
                break
            now = time.monotonic()
            if (
                progress_callback is not None
                and (now - last_update) * 1000.0 >= max(1, update_interval_ms)
            ):
                infos = [dict(info) for info in analysis.multipv]
                if infos:
                    progress_callback(infos)
                    last_update = now
        best = analysis.wait()
        infos = [dict(info) for info in analysis.multipv]
        if progress_callback is not None and infos:
            progress_callback(infos)
        return best.move, infos


def engine_info_text(
    name: str,
    board: chess.Board,
    move: chess.Move,
    info: Dict,
    elapsed_ms: float,
) -> str:
    parts = [
        f"Engine: {name}",
        f"Move: {safe_san(board, move)} ({move.uci()})",
        f"Score: {score_text(info.get('score'), board.turn)}",
    ]
    for key, label in (
        ("depth", "Depth"),
        ("seldepth", "Selective depth"),
        ("nodes", "Nodes"),
        ("nps", "NPS"),
    ):
        if info.get(key) is not None:
            parts.append(f"{label}: {info[key]}")
    parts.append(f"Elapsed: {elapsed_ms:.1f} ms")
    pv = pv_text(board, info.get("pv") or [])
    if pv:
        parts.append(f"PV: {pv}")
    return "\n".join(parts)


# =============================================================================
# Stadium mode
# =============================================================================


class StadiumBoardState:
    def __init__(self):
        self.board = chess.Board()
        self.white_name = "White UCI"
        self.black_name = "Black UCI"
        self.result_override = "*"
        self.termination = "unfinished"

    def reset_arena(self, fen: str, white_name: str, black_name: str):
        self.board = chess.Board(fen)
        self.white_name = white_name
        self.black_name = black_name
        self.result_override = "*"
        self.termination = "unfinished"

    def finish(self, result: str, termination: str):
        self.result_override = str(result)
        self.termination = str(termination)

    def last_move(self) -> Optional[chess.Move]:
        return self.board.move_stack[-1] if self.board.move_stack else None

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Gadidae Stadium"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["White"] = self.white_name
        game.headers["Black"] = self.black_name
        game.headers["Result"] = self.result_override
        game.headers["Termination"] = self.termination
        exporter = chess.pgn.StringExporter(
            headers=True,
            variations=False,
            comments=False,
            columns=80,
        )
        return game.accept(exporter).strip()

    def pgn_movetext(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(
            headers=False,
            variations=False,
            comments=False,
        )
        text = game.accept(exporter).strip()
        for result in ("1-0", "0-1", "1/2-1/2", "*"):
            if text == result:
                return ""
            suffix = " " + result
            if text.endswith(suffix):
                return text[: -len(suffix)].rstrip()
        return text

    def save_pgn(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.pgn())
            handle.write("\n")


class StadiumSettingsDialog(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Settings")
        self.resizable(True, False)
        self.transient(parent)
        self.grab_set()

        self.values = {}
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        white_tab = ttk.Frame(notebook, padding=10)
        black_tab = ttk.Frame(notebook, padding=10)
        match_tab = ttk.Frame(notebook, padding=10)
        notebook.add(white_tab, text="White")
        notebook.add(black_tab, text="Black")
        notebook.add(match_tab, text="Match")

        self._add_engine_tab(white_tab, "white", app)
        self._add_engine_tab(black_tab, "black", app)
        self._add_entry(match_tab, 0, "delay_ms", "Display delay (ms)", app)
        self._add_entry(match_tab, 1, "max_plies", "Max plies", app)
        match_tab.columnconfigure(1, weight=1)

        actions = ttk.Frame(self)
        actions.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(actions, text="Apply", command=self.apply).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        self.bind("<Escape>", lambda _event: self.destroy())
        self.minsize(680, 280)

    def _add_engine_tab(self, parent, color: str, app):
        command_name = f"{color}_uci"
        ttk.Label(parent, text="UCI command").grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=6
        )
        command_var = tk.StringVar(value=str(app.settings[command_name]))
        self.values[command_name] = command_var
        command_entry = ttk.Entry(parent, textvariable=command_var, width=66)
        command_entry.grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(
            parent,
            text="Browse",
            command=lambda: app.browse_uci(command_entry),
        ).grid(row=0, column=2, padx=(8, 0), pady=6)

        self._add_entry(parent, 1, f"{color}_movetime_ms", "Move time (ms)", app)
        self._add_entry(parent, 2, f"{color}_multipv", "Analysis lines", app)
        self._add_entry(parent, 3, f"{color}_options", "Additional UCI options (JSON)", app)
        parent.columnconfigure(1, weight=1)

    def _add_entry(self, parent, row, name, label, app):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=6
        )
        variable = tk.StringVar(value=str(app.settings[name]))
        self.values[name] = variable
        ttk.Entry(parent, textvariable=variable, width=28).grid(
            row=row, column=1, sticky="ew", pady=6
        )

    def apply(self):
        try:
            settings = {
                "white_uci": self.values["white_uci"].get().strip(),
                "white_options": self.values["white_options"].get().strip(),
                "white_movetime_ms": int(self.values["white_movetime_ms"].get()),
                "white_multipv": int(self.values["white_multipv"].get()),
                "black_uci": self.values["black_uci"].get().strip(),
                "black_options": self.values["black_options"].get().strip(),
                "black_movetime_ms": int(self.values["black_movetime_ms"].get()),
                "black_multipv": int(self.values["black_multipv"].get()),
                "delay_ms": int(self.values["delay_ms"].get()),
                "max_plies": int(self.values["max_plies"].get()),
            }
            command_from_text(settings["white_uci"])
            command_from_text(settings["black_uci"])
            uci_options_from_text(settings["white_options"])
            uci_options_from_text(settings["black_options"])
            if settings["white_movetime_ms"] <= 0:
                raise ValueError("white move time must be positive")
            if settings["black_movetime_ms"] <= 0:
                raise ValueError("black move time must be positive")
            if settings["white_multipv"] <= 0 or settings["black_multipv"] <= 0:
                raise ValueError("analysis lines must be positive")
            if settings["delay_ms"] < 0:
                raise ValueError("display delay must be non-negative")
            if settings["max_plies"] <= 0:
                raise ValueError("max plies must be positive")
        except Exception as exc:
            messagebox.showerror("Stadium Settings", str(exc), parent=self)
            return
        self.app.settings = settings
        self.destroy()


class StadiumApp(ChessGUIBase):
    def __init__(
        self,
        root: tk.Tk,
        engine: StadiumBoardState,
        defaults: Dict[str, object],
        parent=None,
        manage_window: bool = True,
    ):
        self.root = root
        self.parent = parent or root
        self.engine = engine
        if manage_window:
            self.root.title("Gadidae Stadium")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.initialize_board_gui()
        self.settings = {
            "white_uci": str(defaults.get("white_uci") or ""),
            "white_options": str(defaults.get("white_options") or "{}"),
            "white_movetime_ms": int(defaults.get("white_movetime_ms", 1000)),
            "white_multipv": int(defaults.get("white_multipv", 5)),
            "black_uci": str(defaults.get("black_uci") or ""),
            "black_options": str(defaults.get("black_options") or "{}"),
            "black_movetime_ms": int(defaults.get("black_movetime_ms", 1000)),
            "black_multipv": int(defaults.get("black_multipv", 5)),
            "delay_ms": int(defaults.get("delay_ms", 250)),
            "max_plies": int(defaults.get("max_plies", 240)),
        }
        self.start_fen = str(defaults.get("fen") or "startpos")
        self.running = False
        self.paused = False
        self.game_status = "Ready"
        self.game_generation = 0
        self.game_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.uci_lock = threading.Lock()
        self.uci_engines: List[chess.engine.SimpleEngine] = []
        self.ui_queue = queue.Queue()
        self.ui_poll_interval_ms = 100
        self.ui_after_id = None
        self.closed = False
        self._build_ui()
        self.draw_board()
        self.refresh_controls()
        self.ui_after_id = self.root.after(
            self.ui_poll_interval_ms,
            self.process_ui_events,
        )

    @property
    def arena_engine(self) -> StadiumBoardState:
        return self.engine

    def _build_ui(self):
        container = ttk.Frame(self.parent, padding=8)
        container.pack(fill=tk.BOTH, expand=True)
        body = ttk.Frame(container)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH)
        right = ttk.Frame(body, padding=(10, 0, 0, 0))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            left,
            width=self.board_size,
            height=self.board_size,
            highlightthickness=0,
        )
        self.canvas.pack()

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=(8, 0))
        self.start_button = self._add_button(controls, "Start", self.start_game)
        self.pause_button = self._add_button(controls, "Pause", self.toggle_pause)
        self.stop_button = self._add_button(controls, "Stop", self.stop_game)
        self.settings_button = self._add_button(
            controls,
            "Settings",
            self.open_settings,
        )
        self._add_button(controls, "Flip", self.flip_board)
        self._add_button(controls, "Save PGN", self.save_pgn)

        fen_bar = ttk.Frame(left)
        fen_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(fen_bar, text="Start FEN:").pack(side=tk.LEFT)
        self.fen_entry = ttk.Entry(fen_bar)
        self.fen_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.fen_entry.insert(0, self.start_fen)

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            right,
            textvariable=self.status_var,
            font=("Arial", 11, "bold"),
            wraplength=500,
        ).pack(anchor="w")

        ttk.Label(right, text="Moves", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(8, 0)
        )
        moves_frame = ttk.Frame(right)
        moves_frame.pack(fill=tk.X, pady=(4, 8))
        self.moves_list = tk.Listbox(moves_frame, width=64, height=8)
        self.moves_list.pack(fill=tk.X)
        moves_scroll = ttk.Scrollbar(
            moves_frame,
            orient=tk.HORIZONTAL,
            command=self.moves_list.xview,
        )
        moves_scroll.pack(fill=tk.X)
        self.moves_list.configure(xscrollcommand=moves_scroll.set)

        ttk.Label(
            right,
            text="UCI analysis",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        self.info_text = tk.Text(right, width=64, height=10, wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(
            right,
            text="Board state",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        self.board_state_text = tk.Text(right, width=64, height=10, wrap=tk.WORD)
        self.board_state_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.config_entries = [self.fen_entry]

    def process_ui_events(self):
        if self.closed:
            return
        while True:
            try:
                kind, generation, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if int(generation) != self.game_generation:
                continue
            if kind == "engines":
                self.arena_engine.white_name = payload["white"]
                self.arena_engine.black_name = payload["black"]
                self.game_status = "Game running"
                self.draw_board()
            elif kind == "thinking":
                self.begin_uci_thinking(payload)
            elif kind == "analysis":
                self.apply_uci_analysis(payload)
            elif kind == "move":
                self.apply_uci_move(payload)
            elif kind == "finished":
                self.finish_game(payload["result"], payload["termination"])
            elif kind == "error":
                self.fail_game(payload)
        self.ui_after_id = self.root.after(
            self.ui_poll_interval_ms,
            self.process_ui_events,
        )

    def refresh_controls(self):
        active = bool(self.running)
        self.start_button.configure(state=tk.DISABLED if active else tk.NORMAL)
        self.pause_button.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.stop_button.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.settings_button.configure(state=tk.DISABLED if active else tk.NORMAL)
        for entry in self.config_entries:
            entry.configure(state=tk.DISABLED if active else tk.NORMAL)

    def update_panels(self):
        turn = "White" if self.engine.board.turn == chess.WHITE else "Black"
        self.status_var.set(f"Turn: {turn} | {self.game_status}")
        self.update_board_state()

    def browse_uci(self, entry: ttk.Entry):
        path = filedialog.askopenfilename(parent=self.root, title="Select UCI engine")
        if not path:
            return
        entry.delete(0, tk.END)
        entry.insert(0, f'"{path}"' if " " in path else path)

    def open_settings(self):
        if self.running:
            return
        StadiumSettingsDialog(self.root, self)

    def read_settings(self):
        white_text = str(self.settings["white_uci"]).strip()
        black_text = str(self.settings["black_uci"]).strip()
        white = command_from_text(white_text)
        black = command_from_text(black_text)
        white_options = uci_options_from_text(self.settings["white_options"])
        black_options = uci_options_from_text(self.settings["black_options"])
        fen_text = strip_wrapping_quotes(self.fen_entry.get())
        board = chess.Board() if not fen_text or fen_text == "startpos" else chess.Board(fen_text)
        white_movetime_ms = int(self.settings["white_movetime_ms"])
        white_multipv = int(self.settings["white_multipv"])
        black_movetime_ms = int(self.settings["black_movetime_ms"])
        black_multipv = int(self.settings["black_multipv"])
        delay_ms = int(self.settings["delay_ms"])
        max_plies = int(self.settings["max_plies"])
        if white_movetime_ms <= 0:
            raise ValueError("white move time must be positive")
        if black_movetime_ms <= 0:
            raise ValueError("black move time must be positive")
        if white_multipv <= 0 or black_multipv <= 0:
            raise ValueError("analysis lines must be positive")
        if delay_ms < 0:
            raise ValueError("display delay must be non-negative")
        if max_plies <= 0:
            raise ValueError("max plies must be positive")
        return (
            white,
            white_options,
            white_movetime_ms,
            white_multipv,
            black,
            black_options,
            black_movetime_ms,
            black_multipv,
            board.fen(),
            delay_ms,
            max_plies,
        )

    def start_game(self):
        if self.running:
            return
        try:
            (
                white,
                white_options,
                white_movetime_ms,
                white_multipv,
                black,
                black_options,
                black_movetime_ms,
                black_multipv,
                fen,
                delay_ms,
                max_plies,
            ) = self.read_settings()
        except Exception as exc:
            messagebox.showerror("Stadium", str(exc), parent=self.root)
            return

        self.game_generation += 1
        generation = self.game_generation
        self.stop_event = threading.Event()
        self.resume_event = threading.Event()
        self.resume_event.set()
        self.running = True
        self.paused = False
        self.game_status = "Starting UCI engines"
        self.arena_engine.reset_arena(fen, "White UCI", "Black UCI")
        self.last_move = None
        self.moves_list.delete(0, tk.END)
        self.info_text.delete("1.0", tk.END)
        self.draw_board()
        self.refresh_controls()

        self.game_thread = threading.Thread(
            target=self.run_uci_game,
            args=(
                generation,
                white,
                white_options,
                white_movetime_ms,
                white_multipv,
                black,
                black_options,
                black_movetime_ms,
                black_multipv,
                fen,
                delay_ms,
                max_plies,
            ),
            daemon=True,
            name=f"uci-arena-{generation}",
        )
        self.game_thread.start()

    def run_uci_game(
        self,
        generation: int,
        white_command: Sequence[str],
        white_options: Dict[str, object],
        white_movetime_ms: int,
        white_multipv: int,
        black_command: Sequence[str],
        black_options: Dict[str, object],
        black_movetime_ms: int,
        black_multipv: int,
        fen: str,
        delay_ms: int,
        max_plies: int,
    ):
        engines: List[chess.engine.SimpleEngine] = []
        terminal_event = (
            "finished",
            {"result": "*", "termination": "unfinished"},
        )
        try:
            white = popen_uci_engine(white_command)
            engines.append(white)
            black = popen_uci_engine(black_command)
            engines.append(black)
            white_multipv = engine_multipv_count(white, white_multipv)
            black_multipv = engine_multipv_count(black, black_multipv)
            white_configurable = configurable_uci_options(white_options)
            black_configurable = configurable_uci_options(black_options)
            if white_configurable:
                white.configure(white_configurable)
            if black_configurable:
                black.configure(black_configurable)
            with self.uci_lock:
                self.uci_engines = list(engines)

            white_name = str(white.id.get("name") or Path(white_command[0]).name)
            black_name = str(black.id.get("name") or Path(black_command[0]).name)
            self.ui_queue.put((
                "engines",
                generation,
                {"white": white_name, "black": black_name},
            ))

            board = chess.Board(fen)
            ply = 0
            while not game_is_over(board) and ply < max_plies:
                if self.stop_event.is_set():
                    break
                while not self.resume_event.wait(timeout=0.1):
                    if self.stop_event.is_set():
                        break
                if self.stop_event.is_set():
                    break

                uci = white if board.turn == chess.WHITE else black
                name = white_name if board.turn == chess.WHITE else black_name
                movetime_ms = (
                    white_movetime_ms if board.turn == chess.WHITE else black_movetime_ms
                )
                multipv = white_multipv if board.turn == chess.WHITE else black_multipv
                before_fen = board.fen()
                started = time.monotonic()
                self.ui_queue.put((
                    "thinking",
                    generation,
                    {
                        "before_fen": before_fen,
                        "engine": name,
                        "turn": "White" if board.turn == chess.WHITE else "Black",
                    },
                ))

                def publish_analysis(infos, selected_move=None):
                    ordered = sorted(
                        infos,
                        key=lambda info: int(info.get("multipv", 1) or 1),
                    )
                    displayed_move = selected_move
                    if displayed_move is None:
                        for info in ordered:
                            pv = list(info.get("pv") or [])
                            if pv and pv[0] in board.legal_moves:
                                displayed_move = pv[0]
                                break
                    if displayed_move is None:
                        return
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    primary_info = primary_info_for_move(ordered, displayed_move)
                    self.ui_queue.put((
                        "analysis",
                        generation,
                        {
                            "before_fen": before_fen,
                            "engine": name,
                            "detail": engine_info_text(
                                name,
                                board,
                                displayed_move,
                                primary_info,
                                elapsed_ms,
                            ),
                            "multipv": multipv_move_rows(
                                board,
                                ordered,
                                displayed_move,
                            ),
                        },
                    ))

                move, multipv_infos = analyse_uci_turn(
                    uci,
                    board,
                    movetime_ms=movetime_ms,
                    multipv=multipv,
                    progress_callback=publish_analysis,
                    stop_event=self.stop_event,
                )
                if move is None or move not in board.legal_moves:
                    raise RuntimeError(
                        f"{name} returned an illegal move: "
                        f"{move.uci() if move is not None else 'none'}"
                )
                san = safe_san(board, move)
                publish_analysis(multipv_infos, selected_move=move)
                board.push(move)
                ply += 1
                self.ui_queue.put((
                    "move",
                    generation,
                    {
                        "before_fen": before_fen,
                        "move": move.uci(),
                        "san": san,
                        "ply": ply,
                        "engine": name,
                    },
                ))
                if delay_ms > 0 and self.stop_event.wait(delay_ms / 1000.0):
                    break

            if self.stop_event.is_set():
                result_text, termination = "*", "stopped"
            else:
                termination_text = game_termination_text(board)
                if termination_text is not None:
                    result_text = game_result(board)
                    termination = termination_text
                elif ply >= max_plies:
                    result_text, termination = "1/2-1/2", "max plies"
                else:
                    result_text, termination = "*", "unfinished"
            terminal_event = (
                "finished",
                {"result": result_text, "termination": termination},
            )
        except Exception as exc:
            if self.stop_event.is_set():
                terminal_event = (
                    "finished",
                    {"result": "*", "termination": "stopped"},
                )
            else:
                terminal_event = ("error", str(exc))
        finally:
            for uci in engines:
                try:
                    uci.quit()
                except Exception:
                    try:
                        uci.close()
                    except Exception:
                        pass
            with self.uci_lock:
                if self.uci_engines == engines:
                    self.uci_engines = []
        self.ui_queue.put((terminal_event[0], generation, terminal_event[1]))

    def begin_uci_thinking(self, payload: Dict):
        if self.engine.board.fen() != payload["before_fen"]:
            return
        self.game_status = f"{payload['turn']} thinking: {payload['engine']}"
        self.moves_list.delete(0, tk.END)
        self.info_text.delete("1.0", tk.END)
        self.update_panels()

    def apply_uci_analysis(self, payload: Dict):
        if self.engine.board.fen() != payload["before_fen"]:
            return
        self.moves_list.delete(0, tk.END)
        for row in payload.get("multipv") or []:
            self.moves_list.insert(tk.END, row)
        if self.moves_list.size() > 0:
            self.moves_list.see(0)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, payload["detail"])

    def apply_uci_move(self, payload: Dict):
        if self.engine.board.fen() != payload["before_fen"]:
            self.fail_game("GUI board and UCI game became unsynchronized")
            return
        move = chess.Move.from_uci(payload["move"])
        if move not in self.engine.board.legal_moves:
            self.fail_game(f"illegal UCI move received: {move.uci()}")
            return
        self.engine.board.push(move)
        self.last_move = move
        self.moves_list.delete(0, tk.END)
        self.info_text.delete("1.0", tk.END)
        self.game_status = "Waiting for next engine"
        self.draw_board()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.resume_event.clear()
            self.game_status = "Paused after current move"
        else:
            self.resume_event.set()
            self.game_status = "Game running"
        self.refresh_controls()
        self.update_panels()

    def stop_game(self):
        if not self.running:
            return
        self.game_status = "Stopping"
        self.stop_event.set()
        self.resume_event.set()
        with self.uci_lock:
            engines = list(self.uci_engines)
        for uci in engines:
            try:
                uci.close()
            except Exception:
                pass
        self.update_panels()

    def finish_game(self, result: str, termination: str):
        self.running = False
        self.paused = False
        self.arena_engine.finish(result, termination)
        self.game_status = f"Finished: {result} ({termination})"
        self.refresh_controls()
        self.draw_board()

    def fail_game(self, error):
        self.stop_event.set()
        self.resume_event.set()
        with self.uci_lock:
            engines = list(self.uci_engines)
        for uci in engines:
            try:
                uci.close()
            except Exception:
                pass
        self.running = False
        self.paused = False
        self.arena_engine.finish("*", "engine error")
        self.game_status = "Engine error"
        self.refresh_controls()
        self.draw_board()
        messagebox.showerror("Stadium", str(error), parent=self.root)

    def on_close(self):
        self.shutdown()
        self.root.destroy()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.game_generation += 1
        self.stop_event.set()
        self.resume_event.set()
        with self.uci_lock:
            engines = list(self.uci_engines)
        for uci in engines:
            try:
                uci.close()
            except Exception:
                pass
        if self.game_thread is not None and self.game_thread.is_alive():
            self.game_thread.join(timeout=1.0)
        if self.ui_after_id is not None:
            try:
                self.root.after_cancel(self.ui_after_id)
            except Exception:
                pass
            self.ui_after_id = None


# =============================================================================
# Simulator mode
# =============================================================================

VALID_SEARCH_TYPES = ("closed", "only-mcts")


def count_pieces(board: chess.Board) -> int:
    return len(board.piece_map())



@dataclasses.dataclass
class EngineConfig:
    uci_command: Optional[str] = None
    device: str = "auto"

    search_type: str = "only-mcts"
    mcts_sims: int = 100
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    movetime_ms: int = 3000
    c_puct: float = 1.5
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_reduction: float = 0.15
    repetition_policy_penalty: float = 0.0
    instant_mate_first: bool = False
    progress_interval_ms: int = 750
    root_topn: int = 8


def bool_from_text(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


SEARCH_PARAMETER_TYPES = {
    "device": str,
    "search_type": str,
    "mcts_sims": int,
    "mcts_min_sims": int,
    "mcts_batch_size": int,
    "movetime_ms": int,
    "c_puct": float,
    "c_puct_base": float,
    "c_puct_factor": float,
    "fpu_reduction": float,
    "repetition_policy_penalty": float,
    "instant_mate_first": bool_from_text,
    "progress_interval_ms": int,
    "root_topn": int,
}


class SimulatorSearchCancellation:
    def __init__(self, active_generation, generation: int, stop_event):
        self.active_generation = active_generation
        self.generation = int(generation)
        self.stop_event = stop_event

    def is_set(self) -> bool:
        return bool(
            self.stop_event.is_set()
            or int(self.active_generation.value) != self.generation
        )


def score_value(score, turn: chess.Color) -> float:
    if score is None:
        return 0.0
    relative = score.pov(turn)
    mate = relative.mate()
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    cp = relative.score()
    return max(-1.0, min(1.0, float(cp or 0) / 1000.0))


def simulator_info(
    board: chess.Board,
    move: chess.Move,
    infos: List[Dict],
    parameters: Dict,
    elapsed_ms: float,
) -> Dict:
    ordered = sorted(infos, key=lambda info: int(info.get("multipv", 1) or 1))
    root = []
    for info in ordered:
        pv = list(info.get("pv") or [])
        if not pv or pv[0] not in board.legal_moves:
            continue
        candidate = pv[0]
        value = score_value(info.get("score"), board.turn)
        root.append({
            "move": candidate.uci(),
            "san": safe_san(board, candidate),
            "score": value,
            "nodes": int(info.get("nodes", 0) or 0),
            "selected": candidate == move,
        })
    primary = root[0] if root else {
        "move": move.uci(),
        "san": safe_san(board, move),
        "score": 0.0,
        "nodes": 0,
        "selected": True,
    }
    return {
        "best_move": move.uci(),
        "best_san": safe_san(board, move),
        "search_type": str(parameters.get("search_type", "only-mcts")),
        "search_backend": str(parameters.get("_engine_name", "UCI")),
        "sims_completed": int(primary.get("nodes", 0)),
        "mcts_soft_cap": int(parameters.get("mcts_sims", 0) or 0),
        "value": float(primary.get("score", 0.0)),
        "elapsed_ms": round(float(elapsed_ms), 2),
        "root": root,
    }


def run_simulator_search_job(job: Dict, output_queue, cancel_event, engine):
    try:
        board = chess.Board(str(job["fen"]))
        parameters = dict(job["parameters"])
        topn = int(job.get("root_topn") or parameters.get("root_topn", 8) or 8)

        def progress(infos):
            if cancel_event is not None and cancel_event.is_set():
                return
            move = next(
                (list(info.get("pv") or [None])[0] for info in infos if info.get("pv")),
                None,
            )
            if move is None:
                return
            info = simulator_info(board, move, infos, parameters, 0.0)
            output_queue.put((
                "progress",
                (int(job["generation"]), str(job["before_fen"]), info),
            ))

        started = time.monotonic()
        move, infos = analyse_uci_turn(
            engine,
            board,
            movetime_ms=max(0, int(parameters.get("movetime_ms", 0) or 0)),
            multipv=max(1, topn),
            nodes=max(0, int(parameters.get("mcts_sims", 0) or 0)),
            progress_callback=progress,
            stop_event=cancel_event,
            update_interval_ms=max(1, int(parameters.get("progress_interval_ms", 750) or 750)),
        )
        if move is None:
            raise RuntimeError("UCI engine returned no move")
        info = simulator_info(
            board,
            move,
            infos,
            parameters,
            (time.monotonic() - started) * 1000.0,
        )
        output_queue.put((
            "finish_suggestions",
            ({
                "ok": True,
                "move": move.uci(),
                "info": info,
                "error": None,
                "generation": int(job["generation"]),
                "before_fen": str(job["before_fen"]),
            },),
        ))
    except Exception as exc:
        output_queue.put((
            "finish_suggestions",
            ({
                "ok": False,
                "move": None,
                "info": None,
                "error": str(exc),
                "generation": int(job.get("generation", -1)),
                "before_fen": str(job.get("before_fen", "")),
            },),
        ))


def simulator_search_worker(
    job_queue,
    output_queue,
    active_generation,
    stop_event,
):
    engine = None
    engine_key = None

    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            return

        # A burst of board edits only needs analysis for the newest position.
        while True:
            try:
                newer_job = job_queue.get_nowait()
            except queue.Empty:
                break
            if newer_job is None:
                return
            job = newer_job

        generation = int(job["generation"])
        cancel_event = SimulatorSearchCancellation(
            active_generation,
            generation,
            stop_event,
        )
        if cancel_event.is_set():
            continue

        parameters = dict(job["parameters"])
        uci_command = str(job.get("uci_command") or "").strip()
        engine_revision = int(job.get("engine_revision", 0))
        next_engine_key = (uci_command, engine_revision)

        try:
            if next_engine_key != engine_key:
                if engine is not None:
                    engine.quit()
                engine = popen_uci_engine(command_from_text(uci_command))
                requested = {
                    "Device": parameters.get("device", "auto"),
                    "SearchType": parameters.get("search_type", "only-mcts"),
                    "MCTSSims": parameters.get("mcts_sims", 0),
                    "MCTSMinSims": parameters.get("mcts_min_sims", 0),
                    "MCTSBatchSize": parameters.get("mcts_batch_size", 32),
                    "MoveTimeMS": parameters.get("movetime_ms", 0),
                    "CPuct": parameters.get("c_puct", 1.5),
                    "CPuctBase": parameters.get("c_puct_base", 19652.0),
                    "CPuctFactor": parameters.get("c_puct_factor", 1.0),
                    "FPUReduction": parameters.get("fpu_reduction", 0.15),
                    "RepetitionPolicyPenalty": parameters.get(
                        "repetition_policy_penalty", 0.0
                    ),
                    "InstantMateFirst": parameters.get("instant_mate_first", False),
                    "ProgressIntervalMS": parameters.get("progress_interval_ms", 750),
                }
                available = {name.lower(): name for name in engine.options}
                configured = {
                    available[name.lower()]: value
                    for name, value in requested.items()
                    if name.lower() in available
                }
                if configured:
                    engine.configure(configured)
                engine_key = next_engine_key
            parameters["_engine_name"] = str(engine.id.get("name", "UCI"))
            if cancel_event.is_set():
                continue
            run_simulator_search_job(job, output_queue, cancel_event, engine)
        except Exception as exc:
            output_queue.put((
                "finish_suggestions",
                ({
                    "ok": False,
                    "move": None,
                    "info": None,
                    "error": str(exc),
                    "generation": generation,
                    "before_fen": str(job.get("before_fen", "")),
                },),
            ))
    if engine is not None:
        engine.quit()


def side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"



def resolve_uci_command(uci_command: Optional[str]) -> Optional[str]:
    if uci_command is None:
        return None
    value = str(uci_command).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return value


class SimulatorState:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        connect_engine: bool = False,
    ):
        self.config = config or EngineConfig()
        self.board = chess.Board()
        self.uci_command: Optional[str] = None
        self.engine_revision = 0
        self.last_ai_info: Optional[Dict] = None
        self.last_suggestions: List[Dict] = []
        self.ai_suggest_open = True

        if connect_engine and self.config.uci_command:
            self.load_engine(self.config.uci_command)

    @property
    def engine_loaded(self) -> bool:
        return bool(self.uci_command)

    def load_engine(self, uci_command: str):
        resolved = resolve_uci_command(uci_command)
        if not resolved:
            raise ValueError("UCI command is empty")
        command_from_text(resolved)
        self.uci_command = resolved
        self.config.uci_command = resolved
        self.engine_revision += 1
        self.clear_analysis()
        return resolved

    def unload_engine(self):
        self.uci_command = None
        self.config.uci_command = None
        self.engine_revision += 1
        self.clear_analysis()

    def configure_engine(self, uci_command: str, parameters: Dict):
        previous_parameters = self.parameter_dict()
        previous_config_command = self.config.uci_command
        previous_uci_command = self.uci_command

        try:
            for name, converter in SEARCH_PARAMETER_TYPES.items():
                if name not in parameters:
                    continue
                value = converter(parameters[name])
                if name in {
                    "mcts_sims",
                    "mcts_min_sims",
                    "mcts_batch_size",
                    "movetime_ms",
                    "progress_interval_ms",
                    "root_topn",
                } and value < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name == "search_type":
                    value = str(value).strip().lower()
                    if value not in VALID_SEARCH_TYPES:
                        raise ValueError(
                            f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}"
                        )
                if name in {
                    "c_puct",
                    "c_puct_base",
                    "c_puct_factor",
                    "fpu_reduction",
                    "repetition_policy_penalty",
                } and value < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name == "c_puct_base" and value < 1:
                    raise ValueError("c_puct_base must be at least 1")
                if name == "repetition_policy_penalty" and value > 1:
                    raise ValueError("repetition_policy_penalty must not exceed 1")
                if name == "mcts_batch_size" and value < 1:
                    raise ValueError("mcts_batch_size must be at least 1")
                if name == "root_topn" and value < 1:
                    raise ValueError("root_topn must be at least 1")
                setattr(self.config, name, value)

            return self.load_engine(uci_command)
        except Exception:
            for name, value in previous_parameters.items():
                setattr(self.config, name, value)
            self.config.uci_command = previous_config_command
            self.uci_command = previous_uci_command
            raise

    def reload_engine(self):
        if not self.uci_command:
            raise RuntimeError("set a UCI command before applying parameters")
        return self.load_engine(self.uci_command)

    def clear_analysis(self):
        self.last_ai_info = None
        self.last_suggestions = []

    def open_ai_suggest(self):
        self.ai_suggest_open = True

    def close_ai_suggest(self):
        self.ai_suggest_open = False
        self.clear_analysis()

    def reset(self, fen: Optional[str] = None):
        value = strip_wrapping_quotes(fen or "")
        self.board = chess.Board() if not value or value == "startpos" else chess.Board(value)
        self.clear_analysis()
        return self.state()

    def load_pgn_string(self, pgn_text: str):
        if not pgn_text.strip():
            raise ValueError("empty PGN")
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            raise ValueError("could not parse PGN")

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        self.board = board
        self.clear_analysis()
        return self.state()

    def load_pgn_file(self, path: str):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return self.load_pgn_string(handle.read())

    def undo(self) -> int:
        if not self.board.move_stack:
            return 0
        self.board.pop()
        self.clear_analysis()
        return 1

    def last_move(self) -> Optional[chess.Move]:
        return self.board.move_stack[-1] if self.board.move_stack else None

    def game_over(self) -> bool:
        return game_is_over(self.board)

    def result(self) -> str:
        return game_result(self.board)

    def outcome_text(self) -> str:
        if not self.game_over():
            return "Game in progress"
        termination = game_termination(self.board)
        return (
            self.result()
            if termination is None
            else f"{self.result()} - {termination.name}"
        )

    def legal_moves_from(self, square: chess.Square) -> List[chess.Move]:
        return [
            move
            for move in self.board.legal_moves
            if move.from_square == square
        ]

    def parse_move(self, text: str) -> chess.Move:
        value = str(text).strip()
        if not value:
            raise ValueError("empty move")

        try:
            move = chess.Move.from_uci(value)
            if move in self.board.legal_moves:
                return move
        except Exception:
            pass

        try:
            move = self.board.parse_san(value)
            if move in self.board.legal_moves:
                return move
        except Exception:
            pass

        raise ValueError(f"illegal move: {text}")

    def make_move(self, move: chess.Move) -> Dict:
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move: {move.uci()}")
        san = safe_san(self.board, move)
        self.board.push(move)
        self.clear_analysis()
        return {
            "uci": move.uci(),
            "san": san,
            "fen": self.board.fen(),
            "turn": side_name(self.board.turn),
            "game_over": self.game_over(),
            "result": self.result() if self.game_over() else "*",
        }

    def make_text_move(self, move_text: str) -> Dict:
        if self.game_over():
            raise RuntimeError("game is already over")
        return self.make_move(self.parse_move(move_text))

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Gadidae Simulator"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["White"] = "Player"
        game.headers["Black"] = "Player"
        game.headers["Result"] = self.result() if self.game_over() else "*"
        exporter = chess.pgn.StringExporter(
            headers=True,
            variations=False,
            comments=False,
            columns=80,
        )
        return game.accept(exporter).strip()

    def pgn_movetext(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(
            headers=False,
            variations=False,
            comments=False,
        )
        text = game.accept(exporter).strip()
        for result in ("1-0", "0-1", "1/2-1/2", "*"):
            if text == result:
                return ""
            suffix = " " + result
            if text.endswith(suffix):
                return text[: -len(suffix)].rstrip()
        return text

    def save_pgn(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.pgn())
            handle.write("\n")

    def parameter_dict(self) -> Dict:
        return {
            name: getattr(self.config, name)
            for name in SEARCH_PARAMETER_TYPES
        }

    def state(self) -> Dict:
        return {
            "fen": self.board.fen(),
            "turn": side_name(self.board.turn),
            "uci_command": self.uci_command,
            "engine_loaded": self.engine_loaded,
            "ai_suggest": "open" if self.ai_suggest_open else "closed",
            "piece_count": count_pieces(self.board),
            "legal_moves": len(list(self.board.legal_moves)),
            "game_over": self.game_over(),
            "result": self.result() if self.game_over() else "*",
            "outcome": self.outcome_text(),
        }


class EngineSettingsDialog(tk.Toplevel):
    def __init__(self, parent, engine: SimulatorState, on_applied):
        super().__init__(parent)
        self.engine = engine
        self.on_applied = on_applied
        self.title("Settings")
        self.resizable(True, False)
        self.transient(parent)
        self.grab_set()

        self.variables = {}
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 6))

        engine_tab = ttk.Frame(notebook, padding=10)
        budget_tab = ttk.Frame(notebook, padding=10)
        mcts_tab = ttk.Frame(notebook, padding=10)
        decisions_tab = ttk.Frame(notebook, padding=10)
        notebook.add(engine_tab, text="Engine")
        notebook.add(budget_tab, text="Budget")
        notebook.add(mcts_tab, text="MCTS")
        notebook.add(decisions_tab, text="Decisions")

        self.path_var = tk.StringVar(value=engine.uci_command or "")
        ttk.Label(engine_tab, text="UCI command").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=6
        )
        path_entry = ttk.Entry(engine_tab, textvariable=self.path_var, width=64)
        path_entry.grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(engine_tab, text="Browse", command=self.browse).grid(
            row=0, column=2, padx=(8, 0), pady=6
        )

        current = engine.parameter_dict()
        self._add_choice(engine_tab, 1, "device", "Device", current, ("auto", "cpu", "cuda"))
        self._add_choice(
            engine_tab, 2, "search_type", "Search type", current, VALID_SEARCH_TYPES
        )
        engine_tab.columnconfigure(1, weight=1)

        budget_fields = (
            ("movetime_ms", "Movetime (ms)"),
            ("mcts_sims", "MCTS sims soft cap"),
            ("mcts_min_sims", "MCTS minimum sims"),
            ("mcts_batch_size", "MCTS batch size"),
            ("root_topn", "Analysis lines"),
            ("progress_interval_ms", "Display update (ms)"),
        )
        for row, (name, label) in enumerate(budget_fields):
            self._add_entry(budget_tab, row, name, label, current)
        budget_tab.columnconfigure(1, weight=1)

        mcts_fields = (
            ("c_puct", "C-PUCT initial"),
            ("c_puct_base", "C-PUCT schedule base"),
            ("c_puct_factor", "C-PUCT schedule factor"),
            ("fpu_reduction", "FPU reduction"),
        )
        for row, (name, label) in enumerate(mcts_fields):
            self._add_entry(mcts_tab, row, name, label, current)
        mcts_tab.columnconfigure(1, weight=1)

        self._add_entry(
            decisions_tab,
            0,
            "repetition_policy_penalty",
            "Repetition policy penalty",
            current,
        )
        variable = tk.BooleanVar(value=bool(current["instant_mate_first"]))
        self.variables["instant_mate_first"] = variable
        ttk.Checkbutton(
            decisions_tab,
            text="Instant Mate First",
            variable=variable,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=6)
        decisions_tab.columnconfigure(1, weight=1)

        buttons = ttk.Frame(self)
        buttons.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(
            buttons,
            text="Disconnect",
            command=self.unload,
        ).pack(side=tk.LEFT)
        ttk.Button(
            buttons,
            text="Apply and Reload",
            command=self.apply,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            buttons,
            text="Cancel",
            command=self.destroy,
        ).pack(side=tk.RIGHT)

        self.bind("<Escape>", lambda _event: self.destroy())
        self.minsize(620, 300)
        path_entry.focus_set()

    def _add_entry(self, parent, row, name, label, current):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=6
        )
        variable = tk.StringVar(value=str(current[name]))
        self.variables[name] = variable
        ttk.Entry(parent, textvariable=variable, width=24).grid(
            row=row, column=1, sticky="ew", pady=6
        )

    def _add_choice(self, parent, row, name, label, current, choices):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=6
        )
        variable = tk.StringVar(value=str(current[name]))
        self.variables[name] = variable
        ttk.Combobox(
            parent,
            textvariable=variable,
            values=tuple(choices),
            state="readonly",
            width=22,
        ).grid(row=row, column=1, sticky="w", pady=6)

    def browse(self):
        path = filedialog.askopenfilename(parent=self, title="Select UCI engine")
        if path:
            self.path_var.set(f'"{path}"' if " " in path else path)

    def apply(self):
        command = self.path_var.get().strip()
        if not command:
            messagebox.showerror(
                "UCI settings",
                "Enter a UCI command.",
                parent=self,
            )
            return
        parameters = {}
        for name, variable in self.variables.items():
            value = variable.get()
            parameters[name] = value.strip() if isinstance(value, str) else value
        try:
            loaded = self.engine.configure_engine(command, parameters)
        except Exception as exc:
            messagebox.showerror(
                "UCI settings",
                str(exc),
                parent=self,
            )
            return
        self.on_applied(loaded)
        self.destroy()

    def unload(self):
        self.engine.unload_engine()
        self.on_applied(None)
        self.destroy()


class SimulatorApp(ChessGUIBase):
    def __init__(
        self,
        root: tk.Tk,
        engine: SimulatorState,
        parent=None,
        manage_window: bool = True,
    ):
        self.root = root
        self.parent = parent or root
        self.engine = engine

        if manage_window:
            self.root.title("Gadidae Simulator")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.initialize_board_gui()

        self.closed = False
        self.ai_thinking = False
        self.ai_worker_process = None
        self.ai_context: Optional[Dict] = None
        self.ai_generation = 0
        self.pending_ai_after_id = None
        self.mp_context = mp.get_context("spawn" if os.name == "nt" else "fork")
        self.ui_queue = self.mp_context.Queue()
        self.ai_job_queue = self.mp_context.Queue()
        self.ai_active_generation = self.mp_context.Value("q", 0)
        self.ai_stop_event = self.mp_context.Event()
        self.ui_poll_interval_ms = 100
        self.ui_after_id = None
        self._build_ui()
        self.draw_board()
        self.refresh_controls()
        self.ui_after_id = self.root.after(
            self.ui_poll_interval_ms,
            self.process_ui_events,
        )
        self.schedule_ai_reply()

    def _build_ui(self):
        container = ttk.Frame(self.parent, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        right = ttk.Frame(container, padding=(10, 0, 0, 0))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            left,
            width=self.board_size,
            height=self.board_size,
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        main_bar = ttk.Frame(left)
        main_bar.pack(fill=tk.X, pady=(8, 0))
        self._add_button(main_bar, "Undo", self.undo_move)
        self.settings_button = self._add_button(
            main_bar,
            "Settings",
            self.open_engine_settings,
        )
        self._add_button(main_bar, "Flip", self.flip_board)
        self._add_button(main_bar, "Import PGN", self.import_pgn)
        self._add_button(main_bar, "Save PGN", self.save_pgn)
        self.ai_suggest_button = self._add_button(
            main_bar,
            "Close",
            self.toggle_ai_suggest,
        )

        reset_bar = ttk.Frame(left)
        reset_bar.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(reset_bar, text="Reset FEN:").pack(side=tk.LEFT)
        self.reset_entry = ttk.Entry(reset_bar)
        self.reset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.reset_entry.bind("<Return>", lambda _event: self.reset_board())
        self._add_button(reset_bar, "Reset", self.reset_board)

        self.status_var = tk.StringVar(value="")
        ttk.Label(
            right,
            textvariable=self.status_var,
            font=("Arial", 11, "bold"),
            wraplength=470,
        ).pack(anchor="w")

        ttk.Label(
            right,
            text="Suggested moves",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(8, 0))
        self.moves_list = tk.Listbox(right, width=60, height=9)
        self.moves_list.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(
            right,
            text="Engine analysis",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        self.info_text = tk.Text(
            right,
            width=60,
            height=11,
            wrap=tk.WORD,
        )
        self.info_text.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(
            right,
            text="Board state",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w")
        self.board_state_text = tk.Text(
            right,
            width=60,
            height=11,
            wrap=tk.WORD,
        )
        self.board_state_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        move_bar = ttk.Frame(right)
        move_bar.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(move_bar, text="Move:").pack(side=tk.LEFT)
        self.move_entry = ttk.Entry(move_bar, width=16)
        self.move_entry.pack(side=tk.LEFT, padx=4)
        self.move_entry.bind(
            "<Return>",
            lambda _event: self.submit_entry_move(),
        )
        self._add_button(move_bar, "Move", self.submit_entry_move)

    def refresh_controls(self):
        for button in self.buttons:
            try:
                button.configure(state=tk.NORMAL)
            except Exception:
                pass

        label = "Close" if self.engine.ai_suggest_open else "Open"
        self.ai_suggest_button.configure(text=label, state=tk.NORMAL)

    def cancel_pending_ai(self, cancel_running: bool = True):
        if self.pending_ai_after_id is not None:
            try:
                self.root.after_cancel(self.pending_ai_after_id)
            except Exception:
                pass
            self.pending_ai_after_id = None
        if cancel_running and self.ai_thinking:
            self.ai_generation += 1
            self.ai_active_generation.value = self.ai_generation
            self.ai_thinking = False
            self.ai_context = None
            self.refresh_controls()

    def begin_ai_task(self):
        self.cancel_pending_ai()
        self.ai_generation += 1
        self.ai_active_generation.value = self.ai_generation
        self.ai_thinking = True
        self.refresh_controls()
        return self.ai_generation

    def ensure_ai_worker(self):
        process = self.ai_worker_process
        if process is not None and process.is_alive():
            return
        if process is not None:
            try:
                process.join(timeout=0.1)
                process.close()
            except Exception:
                pass
        process = self.mp_context.Process(
            target=simulator_search_worker,
            args=(
                self.ai_job_queue,
                self.ui_queue,
                self.ai_active_generation,
                self.ai_stop_event,
            ),
            daemon=True,
        )
        self.ai_worker_process = process
        process.start()

    def stop_ai_worker(self):
        process = self.ai_worker_process
        self.ai_worker_process = None
        if process is None:
            return
        self.ai_stop_event.set()
        try:
            self.ai_job_queue.put_nowait(None)
        except Exception:
            pass
        try:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass

    def on_close(self):
        self.shutdown()
        self.root.destroy()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.cancel_pending_ai()
        self.stop_ai_worker()
        if self.ui_after_id is not None:
            try:
                self.root.after_cancel(self.ui_after_id)
            except Exception:
                pass
            self.ui_after_id = None

    def merge_ai_context(self, payload: Dict) -> Dict:
        generation = int(payload.get("generation", -1))
        context = (
            self.ai_context
            if self.ai_context is not None
            and int(self.ai_context.get("generation", -2)) == generation
            else {}
        )
        merged = dict(context)
        merged.update(payload)
        return merged

    def start_search_process(
        self,
        board: chess.Board,
        generation: int,
        context: Dict,
        root_topn: Optional[int] = None,
    ):
        self.ai_context = dict(context)
        job = {
            "generation": int(generation),
            "before_fen": str(context.get("before_fen") or board.fen()),
            "fen": board.fen(),
            "uci_command": self.engine.uci_command,
            "engine_revision": self.engine.engine_revision,
            "parameters": self.engine.parameter_dict(),
            "root_topn": int(root_topn or self.engine.config.root_topn),
        }
        self.ensure_ai_worker()
        self.ai_job_queue.put(job)

    def is_current_ai_task(self, generation: int, before_fen: Optional[str] = None) -> bool:
        if generation != self.ai_generation:
            return False
        if before_fen is not None and self.engine.board.fen() != before_fen:
            return False
        return True

    def post_ui_event(self, kind: str, *payload):
        self.ui_queue.put((kind, payload))

    def process_ui_events(self):
        if self.closed:
            return
        latest_progress = None
        events = []
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                latest_progress = payload
            else:
                events.append((kind, payload))

        if latest_progress is not None:
            self.apply_search_progress(*latest_progress)

        for kind, payload in events:
            if kind == "finish_suggestions":
                self.finish_suggestions(self.merge_ai_context(payload[0]))

        self.ui_after_id = self.root.after(
            self.ui_poll_interval_ms,
            self.process_ui_events,
        )

    def schedule_ai_reply(self):
        self.cancel_pending_ai(cancel_running=False)
        if (
            self.ai_thinking
            or self.engine.game_over()
            or not self.engine.engine_loaded
        ):
            return
        if self.engine.ai_suggest_open:
            self.pending_ai_after_id = self.root.after(
                200,
                self._run_scheduled_simulator_suggestion,
            )

    def _run_scheduled_simulator_suggestion(self):
        self.pending_ai_after_id = None
        self.start_simulator_suggestion()

    def update_panels(self):
        turn = side_name(self.engine.board.turn)
        status = f"Turn: {turn}"
        if self.ai_thinking:
            status = f"{status} | Analysis running"

        self.status_var.set(status)
        self.update_board_state()
        self.refresh_controls()

    def update_analysis(self, info: Optional[Dict]):
        self.info_text.delete("1.0", tk.END)
        self.moves_list.delete(0, tk.END)
        if not info:
            return

        text = (
            f"Best: {info.get('best_san')} ({info.get('best_move')})\n"
            f"Search: {info.get('search_type')}\n"
            f"Backend: {info.get('search_backend')}\n"
            f"MCTS nodes: {info.get('sims_completed')}/"
            f"{info.get('mcts_soft_cap')}\n"
            f"Value: {info.get('value')}\n"
            f"Elapsed: {info.get('elapsed_ms')} ms\n"
        )
        self.info_text.insert(tk.END, text)

        for index, row in enumerate(info.get("root", []), 1):
            marker = "*" if row.get("selected") else " "
            self.moves_list.insert(
                tk.END,
                f"{marker}{index}. {row.get('san')} ({row.get('move')})  "
                f"score={row.get('score', 0.0):+.3f}  "
                f"nodes={row.get('nodes', 0)}",
            )

    def selectable_piece(self, square: chess.Square) -> bool:
        piece = self.engine.board.piece_at(square)
        return bool(piece and piece.color == self.engine.board.turn)

    def on_click(self, event):
        if self.engine.game_over():
            return

        square = self.screen_to_square(event.x, event.y)
        if square is None:
            return

        if self.selected_square is None:
            if self.selectable_piece(square):
                self.selected_square = square
                self.legal_targets = [
                    move.to_square
                    for move in self.engine.legal_moves_from(square)
                ]
                self.draw_board()
            return

        if square == self.selected_square:
            self.clear_selection()
            self.draw_board()
            return

        move = self.make_move_from_squares(
            self.selected_square,
            square,
        )
        if move is None:
            if self.selectable_piece(square):
                self.selected_square = square
                self.legal_targets = [
                    candidate.to_square
                    for candidate in self.engine.legal_moves_from(square)
                ]
            else:
                self.clear_selection()
            self.draw_board()
            return

        self.clear_selection()
        self.play_move(move)

    def make_move_from_squares(self, from_square, to_square):
        candidates = [
            move
            for move in self.engine.board.legal_moves
            if move.from_square == from_square
            and move.to_square == to_square
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        promotion_text = simpledialog.askstring(
            "Promotion",
            "Promote to q/r/b/n:",
            initialvalue="q",
            parent=self.root,
        )
        promotion_map = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }
        promotion = promotion_map.get(
            (promotion_text or "q").strip().lower(),
            chess.QUEEN,
        )
        for move in candidates:
            if move.promotion == promotion:
                return move
        return candidates[0]

    def play_move(self, move: chess.Move):
        try:
            self.cancel_pending_ai()
            self.engine.make_text_move(move.uci())
            self.last_move = move
            self.update_analysis(None)
            self.draw_board()
            self.schedule_ai_reply()
        except Exception as exc:
            messagebox.showerror("Move", str(exc), parent=self.root)
            self.draw_board()

    def submit_entry_move(self):
        text = self.move_entry.get().strip()
        if not text:
            return
        try:
            move = self.engine.parse_move(text)
            self.move_entry.delete(0, tk.END)
            self.play_move(move)
        except Exception as exc:
            messagebox.showerror("Move", str(exc), parent=self.root)

    def apply_search_progress(self, generation: int, before_fen: str, info: Dict):
        if not self.is_current_ai_task(generation, before_fen):
            return
        self.update_analysis(info)

    def start_simulator_suggestion(self):
        if self.ai_thinking:
            self.cancel_pending_ai()
        if (
            self.engine.game_over()
            or not self.engine.ai_suggest_open
        ):
            return

        before_fen = self.engine.board.fen()
        board_snapshot = self.engine.board.copy(stack=True)
        generation = self.begin_ai_task()
        self.draw_board()

        self.start_search_process(
            board=board_snapshot,
            generation=generation,
            context={
                "generation": generation,
                "before_fen": before_fen,
            },
            root_topn=self.engine.config.root_topn,
        )

    def finish_suggestions(self, payload):
        error_message = None
        try:
            generation = int(payload.get("generation", -1))
            if not self.is_current_ai_task(generation, payload.get("before_fen")):
                return
            if self.engine.board.fen() != payload.get("before_fen"):
                return
            if not self.engine.ai_suggest_open:
                self.engine.clear_analysis()
                return
            if not payload.get("ok"):
                raise RuntimeError(
                    payload.get("error") or "suggestion failed"
                )
            self.update_analysis(payload["info"])
        except Exception as exc:
            generation = int(payload.get("generation", -1))
            if generation != self.ai_generation:
                return
            error_message = f"Engine analysis error: {exc}"
        finally:
            generation = int(payload.get("generation", -1))
            if generation == self.ai_generation:
                self.ai_thinking = False
                self.ai_context = None
                self.draw_board()
                if error_message:
                    self.update_analysis(None)
                    self.info_text.insert(tk.END, error_message)

    def toggle_ai_suggest(self):
        if self.engine.ai_suggest_open:
            self.engine.close_ai_suggest()
            self.cancel_pending_ai()
            self.update_analysis(None)
            if not self.ai_thinking:
                self.draw_board()
            self.refresh_controls()
            return

        self.engine.open_ai_suggest()
        self.update_analysis(None)
        self.draw_board()
        self.schedule_ai_reply()

    def open_engine_settings(self):
        self.cancel_pending_ai()
        EngineSettingsDialog(
            self.root,
            self.engine,
            self.engine_applied,
        )

    def engine_applied(self, command):
        self.update_analysis(None)
        self.draw_board()
        if command:
            self.schedule_ai_reply()
            messagebox.showinfo(
                "UCI engine",
                f"UCI engine configured:\n{command}",
                parent=self.root,
            )
        else:
            self.schedule_ai_reply()
            messagebox.showinfo(
                "UCI engine",
                "UCI engine disconnected.",
                parent=self.root,
            )

    def reset_board(self):
        self.cancel_pending_ai()
        fen = strip_wrapping_quotes(self.reset_entry.get())
        try:
            self.engine.reset(fen or "startpos")
            self.reset_entry.delete(0, tk.END)
            self.clear_selection()
            self.sync_last_move()
            self.update_analysis(None)
            self.draw_board()
            self.schedule_ai_reply()
        except Exception as exc:
            messagebox.showerror(
                "Reset",
                str(exc),
                parent=self.root,
            )

    def undo_move(self):
        self.cancel_pending_ai()
        undone = self.engine.undo()
        self.clear_selection()
        self.sync_last_move()
        self.update_analysis(None)
        self.draw_board()
        self.schedule_ai_reply()

    def import_pgn(self):
        self.cancel_pending_ai()
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Import PGN",
            filetypes=[
                ("PGN files", "*.pgn"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.engine.load_pgn_file(path)
            self.clear_selection()
            self.sync_last_move()
            self.update_analysis(None)
            self.draw_board()
            self.schedule_ai_reply()
        except Exception as exc:
            messagebox.showerror(
                "Import PGN",
                str(exc),
                parent=self.root,
            )


# =============================================================================
# Unified application shell and command-line entry point
# =============================================================================

class GadidaeApp:
    """Owns the shared window and switches between the two GUI workflows."""

    def __init__(
        self,
        root: tk.Tk,
        initial_mode: str,
        simulator_state: SimulatorState,
        stadium_defaults,
    ):
        self.root = root
        self.simulator_state = simulator_state
        self.stadium_state = StadiumBoardState()
        self.stadium_defaults = dict(stadium_defaults)
        self.active_mode = ""
        self.active_app = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.minsize(930, 650)

        mode_bar = ttk.Frame(root, padding=(8, 8, 8, 0))
        mode_bar.pack(fill=tk.X)
        ttk.Label(
            mode_bar,
            text="Gadidae",
            font=("Arial", 12, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 12))

        self.mode_var = tk.StringVar(value=initial_mode)
        for value, label in (("simulator", "Simulator"), ("stadium", "Stadium")):
            ttk.Radiobutton(
                mode_bar,
                text=label,
                value=value,
                variable=self.mode_var,
                command=self.switch_from_control,
            ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 0))
        self.content = ttk.Frame(root)
        self.content.pack(fill=tk.BOTH, expand=True)
        self.switch_mode(initial_mode)

    def switch_from_control(self):
        self.switch_mode(self.mode_var.get())

    def remember_active_settings(self):
        if self.active_mode != "stadium" or self.active_app is None:
            return
        self.stadium_defaults.update(self.active_app.settings)
        self.stadium_defaults["fen"] = self.active_app.fen_entry.get()

    def switch_mode(self, mode: str):
        if mode == self.active_mode:
            return
        if mode not in {"simulator", "stadium"}:
            raise ValueError(f"unsupported GUI mode: {mode}")

        self.remember_active_settings()
        if self.active_app is not None:
            self.active_app.shutdown()
        for child in self.content.winfo_children():
            child.destroy()

        self.active_mode = mode
        self.mode_var.set(mode)
        if mode == "simulator":
            self.root.title("Gadidae - Simulator")
            self.active_app = SimulatorApp(
                self.root,
                self.simulator_state,
                parent=self.content,
                manage_window=False,
            )
        else:
            self.root.title("Gadidae - Stadium")
            self.active_app = StadiumApp(
                self.root,
                self.stadium_state,
                self.stadium_defaults,
                parent=self.content,
                manage_window=False,
            )

    def on_close(self):
        if self.active_app is not None:
            self.active_app.shutdown()
        self.root.destroy()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gadidae position analysis and visible UCI matches"
    )
    parser.add_argument(
        "--mode",
        choices=("simulator", "stadium"),
        default="simulator",
    )

    parser.add_argument("--uci", default="none")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--search-type",
        choices=sorted(VALID_SEARCH_TYPES),
        default="only-mcts",
    )
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--mcts-min-sims", type=int, default=0)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=3000)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--c-puct-base", type=float, default=19652.0)
    parser.add_argument("--c-puct-factor", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.15)
    parser.add_argument("--repetition-policy-penalty", type=float, default=0.0)
    parser.add_argument(
        "--instant-mate-first",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress-interval-ms", type=int, default=750)
    parser.add_argument("--root-topn", type=int, default=8)

    parser.add_argument("--white-uci", default="")
    parser.add_argument("--white-options", default="{}")
    parser.add_argument("--white-movetime-ms", type=int, default=1000)
    parser.add_argument("--white-multipv", type=int, default=5)
    parser.add_argument("--black-uci", default="")
    parser.add_argument("--black-options", default="{}")
    parser.add_argument("--black-movetime-ms", type=int, default=1000)
    parser.add_argument("--black-multipv", type=int, default=5)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--delay-ms", type=int, default=250)
    parser.add_argument("--max-plies", type=int, default=240)
    return parser.parse_args(argv)


def simulator_state_from_args(args) -> SimulatorState:
    uci_command = resolve_uci_command(args.uci)
    config = EngineConfig(
        uci_command=uci_command,
        device=args.device,
        search_type=args.search_type,
        mcts_sims=args.mcts_sims,
        mcts_min_sims=args.mcts_min_sims,
        mcts_batch_size=args.mcts_batch_size,
        movetime_ms=args.movetime_ms,
        c_puct=args.c_puct,
        c_puct_base=args.c_puct_base,
        c_puct_factor=args.c_puct_factor,
        fpu_reduction=args.fpu_reduction,
        repetition_policy_penalty=args.repetition_policy_penalty,
        instant_mate_first=args.instant_mate_first,
        progress_interval_ms=args.progress_interval_ms,
        root_topn=args.root_topn,
    )
    return SimulatorState(config, connect_engine=bool(uci_command))


def stadium_defaults_from_args(args):
    return {
        "white_uci": args.white_uci,
        "white_options": args.white_options,
        "white_movetime_ms": args.white_movetime_ms,
        "white_multipv": args.white_multipv,
        "black_uci": args.black_uci,
        "black_options": args.black_options,
        "black_movetime_ms": args.black_movetime_ms,
        "black_multipv": args.black_multipv,
        "fen": args.fen,
        "delay_ms": args.delay_ms,
        "max_plies": args.max_plies,
    }


def use_repository_working_directory():
    """Keeps relative engine and model paths stable for source and packaged runs."""
    entry = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve()
    project_root = entry.parents[2]
    if (project_root / "scripts").is_dir() and (project_root / "models").is_dir():
        os.chdir(project_root)


def main(argv=None):
    mp.freeze_support()
    use_repository_working_directory()
    args = parse_args(argv)
    root = tk.Tk()
    GadidaeApp(
        root,
        args.mode,
        simulator_state_from_args(args),
        stadium_defaults_from_args(args),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
