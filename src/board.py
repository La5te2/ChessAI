"""GUI chessboard simulator with optional model-assisted play."""

from __future__ import annotations

import argparse
import dataclasses
import io
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import chess
import chess.pgn
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from config import DEVICE
except Exception:
    DEVICE = "cuda"

from model import load_model
from search import (
    SearchOptions,
    VALID_SEARCH_TYPES,
    count_pieces,
    safe_san,
    select_move,
)


@dataclasses.dataclass
class EngineConfig:
    model_path: Optional[str] = None
    device: str = DEVICE

    search_type: str = "only-mcts"
    mcts_sims: int = 100
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    movetime_ms: int = 3000
    c_puct: float = 1.5
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_reduction: float = 0.15
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
    "progress_interval_ms": int,
    "root_topn": int,
}


def search_options_from_parameters(parameters: Dict, root_topn: Optional[int] = None) -> SearchOptions:
    movetime_ms = int(parameters.get("movetime_ms", 0) or 0)
    return SearchOptions(
        search_type=str(parameters.get("search_type", "only-mcts")),
        mcts_sims=int(parameters.get("mcts_sims", 0) or 0),
        mcts_min_sims=int(parameters.get("mcts_min_sims", 0) or 0),
        mcts_batch_size=max(1, int(parameters.get("mcts_batch_size", 32) or 32)),
        time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
        c_puct=float(parameters.get("c_puct", 1.5) or 1.5),
        c_puct_base=float(parameters.get("c_puct_base", 19652.0) or 19652.0),
        c_puct_factor=float(parameters.get("c_puct_factor", 1.0) or 1.0),
        fpu_reduction=float(parameters.get("fpu_reduction", 0.15) or 0.15),
        progress_interval_sec=max(
            0.0,
            float(parameters.get("progress_interval_ms", 750) or 0) / 1000.0,
        ),
        root_topn=max(1, int(root_topn or parameters.get("root_topn", 8) or 8)),
    )


def board_search_process(job: Dict, output_queue, cancel_event):
    try:
        board = chess.Board(str(job["fen"]))
        parameters = dict(job["parameters"])
        model_path = str(job.get("model_path") or "").strip()
        if model_path.lower() == "none":
            model_path = ""
        model = (
            load_model(model_path, device=str(parameters.get("device", DEVICE)))
            if model_path
            else None
        )
        options = search_options_from_parameters(
            parameters,
            root_topn=int(job.get("root_topn") or parameters.get("root_topn", 8) or 8),
        )

        def progress(info):
            if cancel_event is not None and cancel_event.is_set():
                return
            output_queue.put((
                "progress",
                (int(job["generation"]), str(job["before_fen"]), info),
            ))

        move, info = select_move(
            board,
            model,
            options,
            device=str(parameters.get("device", DEVICE)),
            cancel_event=cancel_event,
            progress_callback=progress,
        )
        output_queue.put((
            str(job["finish_kind"]),
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
            str(job.get("finish_kind") or "finish_suggestions"),
            ({
                "ok": False,
                "move": None,
                "info": None,
                "error": str(exc),
                "generation": int(job.get("generation", -1)),
                "before_fen": str(job.get("before_fen", "")),
            },),
        ))


def color_from_side(side: str) -> chess.Color:
    value = str(side).strip().lower()
    if value in {"white", "w", "白", "白棋"}:
        return chess.WHITE
    if value in {"black", "b", "黑", "黑棋"}:
        return chess.BLACK
    raise ValueError(f"unknown side: {side}")


def side_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def strip_wrapping_quotes(text: str) -> str:
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


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path.cwd()


def resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    if model_path is None:
        return None

    value = str(model_path).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    path = Path(value)
    if path.is_absolute() and path.exists():
        return str(path)

    candidates = [
        Path.cwd() / path,
        app_base_dir() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


class ChessEngine:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        load_weights: bool = False,
    ):
        self.config = config or EngineConfig()
        self.board = chess.Board()
        self.mode = "simulator"
        self.user_color = chess.WHITE
        self.ai_color = chess.BLACK
        self.model = None
        self.model_path: Optional[str] = None
        self.last_ai_info: Optional[Dict] = None
        self.last_suggestions: List[Dict] = []
        self.ai_suggest_open = True

        if load_weights and self.config.model_path:
            self.load_model(self.config.model_path)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and bool(self.model_path)

    @property
    def playing_with_ai(self) -> bool:
        return self.mode == "ai"

    def search_options(self) -> SearchOptions:
        return SearchOptions(
            search_type=self.config.search_type,
            mcts_sims=self.config.mcts_sims,
            mcts_min_sims=self.config.mcts_min_sims,
            mcts_batch_size=self.config.mcts_batch_size,
            time_limit=(
                self.config.movetime_ms / 1000.0
                if self.config.movetime_ms > 0
                else None
            ),
            c_puct=self.config.c_puct,
            c_puct_base=self.config.c_puct_base,
            c_puct_factor=self.config.c_puct_factor,
            fpu_reduction=self.config.fpu_reduction,
            progress_interval_sec=max(0.0, self.config.progress_interval_ms / 1000.0),
            root_topn=self.config.root_topn,
        )

    def load_model(self, model_path: str):
        resolved = resolve_model_path(model_path)
        if not resolved:
            raise ValueError("model path is empty")
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"model not found: {resolved}")

        model = load_model(resolved, device=self.config.device)
        model.eval()
        self.model = model
        self.model_path = resolved
        self.config.model_path = resolved
        self.clear_analysis()
        return resolved

    def unload_model(self):
        self.model = None
        self.model_path = None
        self.config.model_path = None
        self.mode = "simulator"
        self.clear_analysis()

    def configure_model(self, model_path: str, parameters: Dict):
        previous_parameters = self.parameter_dict()
        previous_config_path = self.config.model_path
        previous_model = self.model
        previous_model_path = self.model_path

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
                } and value < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name == "c_puct_base" and value < 1:
                    raise ValueError("c_puct_base must be at least 1")
                if name == "mcts_batch_size" and value < 1:
                    raise ValueError("mcts_batch_size must be at least 1")
                if name == "root_topn" and value < 1:
                    raise ValueError("root_topn must be at least 1")
                setattr(self.config, name, value)

            return self.load_model(model_path)
        except Exception:
            for name, value in previous_parameters.items():
                setattr(self.config, name, value)
            self.config.model_path = previous_config_path
            self.model = previous_model
            self.model_path = previous_model_path
            raise

    def reload_model(self):
        if not self.model_path:
            raise RuntimeError("load a model before applying model parameters")
        return self.load_model(self.model_path)

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

    def start_ai_game(
        self,
        user_side: str,
        from_current_position: bool,
    ):
        if not from_current_position:
            self.reset("startpos")
        self.user_color = color_from_side(user_side)
        self.ai_color = not self.user_color
        self.mode = "ai"
        self.clear_analysis()
        return self.state()

    def enter_simulator(self):
        self.mode = "simulator"
        self.clear_analysis()
        return self.state()

    def undo(self) -> int:
        if not self.board.move_stack:
            return 0

        if self.playing_with_ai:
            undone = 0
            if self.board.turn == self.user_color:
                self.board.pop()
                undone += 1
                if self.board.move_stack and self.board.turn == self.ai_color:
                    self.board.pop()
                    undone += 1
            else:
                self.board.pop()
                undone += 1
        else:
            self.board.pop()
            undone = 1

        self.clear_analysis()
        return undone

    def last_move(self) -> Optional[chess.Move]:
        return self.board.move_stack[-1] if self.board.move_stack else None

    def is_human_turn(self) -> bool:
        return (
            self.mode == "simulator"
            or self.board.turn == self.user_color
        )

    def is_ai_turn(self) -> bool:
        return (
            self.playing_with_ai
            and self.board.turn == self.ai_color
        )

    def game_over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def result(self) -> str:
        return self.board.result(claim_draw=True)

    def outcome_text(self) -> str:
        if not self.game_over():
            return "Game in progress"
        outcome = self.board.outcome(claim_draw=True)
        return (
            self.result()
            if outcome is None
            else f"{self.result()} - {outcome.termination.name}"
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

    def make_human_move(self, move_text: str) -> Dict:
        if self.game_over():
            raise RuntimeError("game is already over")
        if not self.is_human_turn():
            raise RuntimeError("the AI side is to move")
        return self.make_move(self.parse_move(move_text))

    def choose_ai_move(
        self,
        cancel_event=None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        board_snapshot: Optional[chess.Board] = None,
    ) -> Tuple[chess.Move, Dict]:
        if not self.playing_with_ai:
            raise RuntimeError("start Play first")
        if not self.is_ai_turn():
            raise RuntimeError("the AI side is not to move")
        if self.game_over():
            raise RuntimeError("game is already over")

        search_board = (
            board_snapshot.copy(stack=True)
            if board_snapshot is not None
            else self.board.copy(stack=True)
        )
        live_fen = search_board.fen()
        move, info = select_move(
            search_board,
            self.model,
            self.search_options(),
            device=self.config.device,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("AI search cancelled")
        if self.board.fen() != live_fen:
            raise RuntimeError("board changed during AI search")

        legal = {candidate.uci(): candidate for candidate in self.board.legal_moves}
        if move.uci() not in legal:
            for row in info.get("root", []):
                candidate = row.get("move")
                if candidate in legal:
                    return legal[candidate], info
            raise RuntimeError(f"search returned illegal move: {move.uci()}")
        return legal[move.uci()], info

    def make_ai_move(self) -> Dict:
        if not self.is_ai_turn():
            raise RuntimeError("the AI side is not to move")

        before_board = self.board.copy(stack=True)
        before_fen = self.board.fen()
        before_turn = self.board.turn
        before_length = len(self.board.move_stack)

        try:
            move, info = self.choose_ai_move()

            if self.board.fen() != before_fen:
                raise RuntimeError("board changed before the AI move was applied")
            if self.board.turn != before_turn:
                raise RuntimeError("side to move changed during AI search")
            if move not in self.board.legal_moves:
                raise RuntimeError(f"AI selected illegal move: {move.uci()}")

            result = self.make_move(move)

            after_length = len(self.board.move_stack)
            if after_length != before_length + 1:
                raise RuntimeError(
                    f"AI must play exactly one ply; got "
                    f"{after_length - before_length}"
                )
            if not self.game_over() and self.board.turn != self.user_color:
                raise RuntimeError("AI move did not return the turn to the user")

            self.last_ai_info = info
            self.last_suggestions = info.get("root", [])
            return {"move": result, "info": info}
        except Exception:
            self.board = before_board
            self.clear_analysis()
            raise

    def suggestions(
        self,
        topn: int = 8,
        cancel_event=None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        board_snapshot: Optional[chess.Board] = None,
    ) -> Tuple[List[Dict], Dict]:
        options = dataclasses.replace(
            self.search_options(),
            root_topn=max(1, int(topn)),
        )
        board_copy = (
            board_snapshot.copy(stack=True)
            if board_snapshot is not None
            else self.board.copy(stack=True)
        )
        live_fen = board_copy.fen()
        move, info = select_move(
            board_copy,
            self.model,
            options,
            device=self.config.device,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("AI search cancelled")
        if self.board.fen() != live_fen:
            raise RuntimeError("board changed during AI search")
        suggestions = info.get("root", [])[: max(1, int(topn))]
        self.last_ai_info = info
        self.last_suggestions = suggestions
        return suggestions, info

    def pgn(self) -> str:
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = (
            "Human vs AI"
            if self.playing_with_ai
            else "Chessboard Simulation"
        )
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        if self.playing_with_ai:
            game.headers["White"] = (
                "Human" if self.user_color == chess.WHITE else "ChessAI"
            )
            game.headers["Black"] = (
                "Human" if self.user_color == chess.BLACK else "ChessAI"
            )
        else:
            game.headers["White"] = "Player"
            game.headers["Black"] = "Player"
        game.headers["Result"] = self.result() if self.game_over() else "*"
        return str(game)

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
            "mode": self.mode,
            "user_side": side_name(self.user_color),
            "ai_side": side_name(self.ai_color),
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "ai_suggest": "open" if self.ai_suggest_open else "closed",
            "piece_count": count_pieces(self.board),
            "legal_moves": len(list(self.board.legal_moves)),
            "game_over": self.game_over(),
            "result": self.result() if self.game_over() else "*",
            "outcome": self.outcome_text(),
        }


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


class ModelSettingsDialog(tk.Toplevel):
    def __init__(self, parent, engine: ChessEngine, on_applied):
        super().__init__(parent)
        self.engine = engine
        self.on_applied = on_applied
        self.title("Model and Search Parameters")
        self.resizable(False, True)
        self.transient(parent)
        self.grab_set()

        self.variables = {}
        row = 0

        ttk.Label(self, text="Model path").grid(
            row=row,
            column=0,
            sticky="w",
            padx=8,
            pady=5,
        )
        self.path_var = tk.StringVar(value=engine.model_path or "")
        ttk.Entry(self, textvariable=self.path_var, width=54).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=8,
            pady=5,
        )
        ttk.Button(self, text="Browse", command=self.browse).grid(
            row=row,
            column=2,
            padx=8,
            pady=5,
        )
        row += 1

        labels = {
            "device": "Device",
            "search_type": "Search type",
            "mcts_sims": "MCTS sims soft cap",
            "mcts_min_sims": "MCTS minimum sims",
            "mcts_batch_size": "MCTS batch size",
            "movetime_ms": "Movetime (ms)",
            "c_puct": "C-PUCT initial",
            "c_puct_base": "C-PUCT schedule base",
            "c_puct_factor": "C-PUCT schedule factor",
            "fpu_reduction": "FPU reduction",
            "progress_interval_ms": "Progress interval (ms)",
            "root_topn": "Suggestion count",
        }
        current = engine.parameter_dict()
        for name in SEARCH_PARAMETER_TYPES:
            ttk.Label(self, text=labels[name]).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=3,
            )
            variable = tk.StringVar(value=str(current[name]))
            self.variables[name] = variable
            ttk.Entry(self, textvariable=variable, width=24).grid(
                row=row,
                column=1,
                sticky="ew",
                padx=8,
                pady=3,
            )
            row += 1

        buttons = ttk.Frame(self)
        buttons.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="e",
            padx=8,
            pady=10,
        )
        ttk.Button(
            buttons,
            text="Unload Model",
            command=self.unload,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            buttons,
            text="Apply and Reload",
            command=self.apply,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            buttons,
            text="Cancel",
            command=self.destroy,
        ).pack(side=tk.LEFT, padx=4)

        self.columnconfigure(1, weight=1)

    def browse(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select model",
            filetypes=[
                ("PyTorch checkpoint", "*.pth *.pt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.path_var.set(path)

    def apply(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showerror(
                "Model settings",
                "Select a model file.",
                parent=self,
            )
            return
        parameters = {
            name: variable.get().strip()
            for name, variable in self.variables.items()
        }
        try:
            loaded = self.engine.configure_model(path, parameters)
        except Exception as exc:
            messagebox.showerror(
                "Model settings",
                str(exc),
                parent=self,
            )
            return
        self.on_applied(loaded)
        self.destroy()

    def unload(self):
        self.engine.unload_model()
        self.on_applied(None)
        self.destroy()


class ChessBoardApp:
    def __init__(self, root: tk.Tk, engine: ChessEngine):
        self.root = root
        self.engine = engine

        self.root.title("Chessboard Simulator")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.square_size = 72
        self.board_size = self.square_size * 8
        self.flipped = False
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move = self.engine.last_move()

        self.ai_thinking = False
        self.ai_process = None
        self.ai_task: Optional[str] = None
        self.ai_context: Optional[Dict] = None
        self.ai_generation = 0
        self.ai_cancel_event = None
        self.pending_ai_after_id = None
        self.mp_context = mp.get_context("spawn" if os.name == "nt" else "fork")
        self.ui_queue = self.mp_context.Queue()
        self.ui_poll_interval_ms = 100
        self.buttons: List[ttk.Button] = []

        self.light = "#EEEED2"
        self.dark = "#769656"
        self.selected_color = "#F6F669"
        self.target_color = "#BACA44"
        self.lastmove_color = "#CDD26A"

        self._build_ui()
        self.draw_board()
        self.refresh_controls()
        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)
        self.schedule_ai_reply()

    def _add_button(self, parent, text, command, side=tk.LEFT):
        button = ttk.Button(parent, text=text, command=command)
        button.pack(side=side, padx=2)
        self.buttons.append(button)
        return button

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=8)
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
        self.model_button = self._add_button(
            main_bar,
            "Settings",
            self.open_model_settings,
        )
        self.play_ai_button = self._add_button(
            main_bar,
            "Play",
            self.start_ai_game,
        )
        self.simulator_button = self._add_button(
            main_bar,
            "Simulator",
            self.enter_simulator,
        )
        self._add_button(main_bar, "Flip", self.flip_board)
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
            text="Model analysis",
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
        self._add_button(move_bar, "Import PGN", self.import_pgn)
        self._add_button(move_bar, "Save PGN", self.save_pgn)

    def refresh_controls(self):
        for button in self.buttons:
            try:
                button.configure(state=tk.NORMAL)
            except Exception:
                pass

        self.play_ai_button.configure(state=tk.NORMAL)
        self.simulator_button.configure(
            state=(
                tk.NORMAL
                if self.engine.playing_with_ai
                else tk.DISABLED
            )
        )

        label = "Close" if self.engine.ai_suggest_open else "Open"
        state = (
            tk.NORMAL
            if self.engine.mode == "simulator"
            else tk.DISABLED
        )
        self.ai_suggest_button.configure(text=label, state=state)

    def cancel_pending_ai(self, cancel_running: bool = True):
        if self.pending_ai_after_id is not None:
            try:
                self.root.after_cancel(self.pending_ai_after_id)
            except Exception:
                pass
            self.pending_ai_after_id = None
        if cancel_running and self.ai_thinking:
            self.ai_generation += 1
            if self.ai_cancel_event is not None:
                self.ai_cancel_event.set()
            self.stop_ai_process(terminate=True)
            self.ai_cancel_event = None
            self.ai_thinking = False
            self.ai_task = None
            self.ai_context = None
            self.refresh_controls()

    def begin_ai_task(self, task: str):
        self.cancel_pending_ai()
        self.ai_generation += 1
        cancel_event = self.mp_context.Event()
        self.ai_cancel_event = cancel_event
        self.ai_thinking = True
        self.ai_task = task
        self.refresh_controls()
        return self.ai_generation, cancel_event

    def stop_ai_process(self, terminate: bool = False):
        process = self.ai_process
        self.ai_process = None
        if process is None:
            return
        try:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
        except Exception:
            pass
        try:
            process.close()
        except Exception:
            pass

    def on_close(self):
        self.cancel_pending_ai()
        self.root.destroy()

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
        task: str,
        board: chess.Board,
        generation: int,
        cancel_event,
        finish_kind: str,
        context: Dict,
        root_topn: Optional[int] = None,
    ):
        self.ai_context = dict(context)
        job = {
            "task": task,
            "generation": int(generation),
            "before_fen": str(context.get("before_fen") or board.fen()),
            "fen": board.fen(),
            "model_path": self.engine.model_path,
            "parameters": self.engine.parameter_dict(),
            "root_topn": int(root_topn or self.engine.config.root_topn),
            "finish_kind": finish_kind,
        }
        process = self.mp_context.Process(
            target=board_search_process,
            args=(job, self.ui_queue, cancel_event),
            daemon=True,
        )
        self.ai_process = process
        process.start()

    def is_current_ai_task(self, generation: int, before_fen: Optional[str] = None) -> bool:
        if generation != self.ai_generation:
            return False
        if before_fen is not None and self.engine.board.fen() != before_fen:
            return False
        return True

    def post_ui_event(self, kind: str, *payload):
        self.ui_queue.put((kind, payload))

    def process_ui_events(self):
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
            if kind == "finish_ai_move":
                self.finish_ai_move(self.merge_ai_context(payload[0]))
            elif kind == "finish_suggestions":
                self.finish_suggestions(self.merge_ai_context(payload[0]))

        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)

    def schedule_ai_reply(self):
        self.cancel_pending_ai(cancel_running=False)
        if (
            self.ai_thinking
            or self.engine.game_over()
        ):
            return
        if self.engine.is_ai_turn():
            self.pending_ai_after_id = self.root.after(
                200,
                self._run_scheduled_ai_move,
            )
        elif (
            self.engine.mode == "simulator"
            and self.engine.ai_suggest_open
        ):
            self.pending_ai_after_id = self.root.after(
                200,
                self._run_scheduled_simulator_suggestion,
            )

    def _run_scheduled_ai_move(self):
        self.pending_ai_after_id = None
        self.ai_move_once()

    def _run_scheduled_simulator_suggestion(self):
        self.pending_ai_after_id = None
        self.start_simulator_suggestion()

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

    def update_panels(self):
        turn = side_name(self.engine.board.turn)
        mode = "Play" if self.engine.playing_with_ai else "Simulator"
        status = f"Mode: {mode} | Turn: {turn}"
        if self.ai_thinking:
            label = "AI move" if self.ai_task == "move" else "Analysis"
            status = f"{status} | {label} running"

        self.status_var.set(status)
        self.update_board_state()
        self.refresh_controls()

    def update_board_state(self):
        text = (
            f"FEN:\n{self.engine.board.fen()}\n\n"
            f"PGN:\n{self.engine.pgn_movetext()}"
        )
        self.board_state_text.configure(state=tk.NORMAL)
        self.board_state_text.delete("1.0", tk.END)
        self.board_state_text.insert(tk.END, text)
        self.board_state_text.configure(state=tk.DISABLED)

    def update_analysis(self, info: Optional[Dict]):
        self.info_text.delete("1.0", tk.END)
        self.moves_list.delete(0, tk.END)
        if not info:
            return

        text = (
            f"Best: {info.get('best_san')} ({info.get('best_move')})\n"
            f"Search: {info.get('search_type')}\n"
            f"Partial: {info.get('partial')}\n"
            f"Cancelled: {info.get('cancelled')}\n"
            f"MCTS sims: {info.get('sims_completed')}/"
            f"{info.get('mcts_dynamic_target')}/"
            f"{info.get('mcts_soft_cap')}\n"
            f"Uncertainty: {info.get('uncertainty')}\n"
            f"Value: {info.get('value')}\n"
            f"C-PUCT root: {info.get('c_puct_root')}\n"
            f"FPU root: {info.get('fpu_root')}\n"
            f"Expanded nodes: {info.get('nodes')}\n"
            f"NN batches: {info.get('nn_batches')}\n"
            f"Leaf depth avg/max: "
            f"{info.get('avg_leaf_depth')}/{info.get('max_leaf_depth')}\n"
            f"Elapsed: {info.get('elapsed_ms')} ms\n"
        )
        self.info_text.insert(tk.END, text)

        for index, row in enumerate(info.get("root", []), 1):
            marker = "*" if row.get("selected") else " "
            self.moves_list.insert(
                tk.END,
                f"{marker}{index}. {row.get('san')} ({row.get('move')})  "
                f"p={row.get('p', 0.0):.4f}  "
                f"visits={row.get('visits', 0)}  "
                f"q={row.get('q', 0.0):+.3f}",
            )

    def selectable_piece(self, square: chess.Square) -> bool:
        piece = self.engine.board.piece_at(square)
        if not piece or piece.color != self.engine.board.turn:
            return False
        if self.engine.playing_with_ai:
            return (
                self.engine.is_human_turn()
                and piece.color == self.engine.user_color
            )
        return True

    def on_click(self, event):
        if self.engine.game_over():
            return
        if not self.engine.is_human_turn():
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
        self.play_human_move(move)

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

    def play_human_move(self, move: chess.Move):
        try:
            self.cancel_pending_ai()
            self.engine.make_human_move(move.uci())
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
            self.play_human_move(move)
        except Exception as exc:
            messagebox.showerror("Move", str(exc), parent=self.root)

    def ai_move_once(self):
        if self.ai_thinking:
            self.cancel_pending_ai()
        if not self.engine.is_ai_turn():
            return

        before_board = self.engine.board.copy(stack=True)
        before_fen = self.engine.board.fen()
        before_turn = self.engine.board.turn
        before_length = len(self.engine.board.move_stack)
        generation, cancel_event = self.begin_ai_task("move")
        self.clear_selection()
        self.draw_board()

        self.start_search_process(
            task="move",
            board=before_board,
            generation=generation,
            cancel_event=cancel_event,
            finish_kind="finish_ai_move",
            context={
                "generation": generation,
                "cancel_event": cancel_event,
                "before_board": before_board,
                "before_fen": before_fen,
                "before_turn": before_turn,
                "before_length": before_length,
            },
        )

    def apply_search_progress(self, generation: int, before_fen: str, info: Dict):
        if not self.is_current_ai_task(generation, before_fen):
            return
        self.update_analysis(info)

    def finish_ai_move(self, payload):
        try:
            generation = int(payload.get("generation", -1))
            if not self.is_current_ai_task(generation, payload.get("before_fen")):
                return
            cancel_event = payload.get("cancel_event")
            if cancel_event is not None and cancel_event.is_set():
                return
            if self.engine.board.fen() != payload.get("before_fen"):
                return
            if not payload.get("ok"):
                raise RuntimeError(payload.get("error") or "AI search failed")

            if self.engine.board.turn != payload.get("before_turn"):
                raise RuntimeError("side to move changed during AI search")
            if not self.engine.is_ai_turn():
                raise RuntimeError("the AI side is no longer to move")

            move = chess.Move.from_uci(payload["move"])
            if move not in self.engine.board.legal_moves:
                raise RuntimeError(f"AI selected illegal move: {move.uci()}")

            self.engine.make_move(move)

            before_length = int(payload.get("before_length", -1))
            after_length = len(self.engine.board.move_stack)
            if after_length != before_length + 1:
                raise RuntimeError(
                    f"AI must play exactly one ply; got {after_length - before_length}"
                )
            if (
                not self.engine.game_over()
                and self.engine.board.turn != self.engine.user_color
            ):
                raise RuntimeError("AI move did not return the turn to the user")

            self.engine.last_ai_info = payload["info"]
            self.last_move = move
            self.update_analysis(payload["info"])
        except Exception as exc:
            generation = int(payload.get("generation", -1))
            if generation != self.ai_generation:
                return
            before_board = payload.get("before_board")
            if before_board is not None:
                self.engine.board = before_board
                self.engine.clear_analysis()
                self.sync_last_move()
            messagebox.showerror(
                "AI move",
                str(exc),
                parent=self.root,
            )
        finally:
            generation = int(payload.get("generation", -1))
            if generation == self.ai_generation:
                self.ai_thinking = False
                self.ai_task = None
                self.ai_cancel_event = None
                self.ai_context = None
                self.stop_ai_process(terminate=False)
                self.draw_board()
                self.schedule_ai_reply()

    def start_simulator_suggestion(self):
        if self.ai_thinking:
            self.cancel_pending_ai()
        if (
            self.engine.mode != "simulator"
            or self.engine.game_over()
            or not self.engine.ai_suggest_open
        ):
            return

        before_fen = self.engine.board.fen()
        board_snapshot = self.engine.board.copy(stack=True)
        generation, cancel_event = self.begin_ai_task("suggestion")
        self.draw_board()

        self.start_search_process(
            task="suggestion",
            board=board_snapshot,
            generation=generation,
            cancel_event=cancel_event,
            finish_kind="finish_suggestions",
            context={
                "generation": generation,
                "cancel_event": cancel_event,
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
            cancel_event = payload.get("cancel_event")
            if cancel_event is not None and cancel_event.is_set():
                return
            if self.engine.mode != "simulator":
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
            error_message = f"Model analysis error: {exc}"
        finally:
            generation = int(payload.get("generation", -1))
            if generation == self.ai_generation:
                self.ai_thinking = False
                self.ai_task = None
                self.ai_cancel_event = None
                self.ai_context = None
                self.stop_ai_process(terminate=False)
                self.draw_board()
                if error_message:
                    self.update_analysis(None)
                    self.info_text.insert(tk.END, error_message)

    def toggle_ai_suggest(self):
        if self.engine.mode != "simulator" or self.ai_task == "move":
            return
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

    def open_model_settings(self):
        self.cancel_pending_ai()
        ModelSettingsDialog(
            self.root,
            self.engine,
            self.model_applied,
        )

    def model_applied(self, path):
        self.update_analysis(None)
        self.draw_board()
        if path:
            self.schedule_ai_reply()
            messagebox.showinfo(
                "Model",
                f"Model loaded with the current parameters:\n{path}",
                parent=self.root,
            )
        else:
            self.enter_simulator()
            messagebox.showinfo(
                "Model",
                "Model unloaded.",
                parent=self.root,
            )

    def start_ai_game(self):
        side_answer = messagebox.askyesnocancel(
            "Play",
            "Choose your side.\n\nYes = White\nNo = Black",
            parent=self.root,
        )
        if side_answer is None:
            return
        user_side = "white" if side_answer else "black"

        position_answer = messagebox.askyesnocancel(
            "Play",
            "Choose the starting position.\n\n"
            "Yes = Current position\n"
            "No = Start position",
            parent=self.root,
        )
        if position_answer is None:
            return

        self.cancel_pending_ai()
        self.engine.start_ai_game(
            user_side=user_side,
            from_current_position=bool(position_answer),
        )
        self.flipped = self.engine.user_color == chess.BLACK
        self.clear_selection()
        self.sync_last_move()
        self.update_analysis(None)
        self.draw_board()
        self.schedule_ai_reply()

    def enter_simulator(self):
        self.cancel_pending_ai()
        self.engine.enter_simulator()
        self.clear_selection()
        self.update_analysis(None)
        self.draw_board()
        self.schedule_ai_reply()

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
            messagebox.showerror(
                "Save PGN",
                str(exc),
                parent=self.root,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chessboard GUI with optional model assistance"
    )
    parser.add_argument("--model", default="none")
    parser.add_argument("--device", default=DEVICE)
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
    parser.add_argument("--progress-interval-ms", type=int, default=750)
    parser.add_argument("--root-topn", type=int, default=8)
    return parser.parse_args()


def main():
    mp.freeze_support()
    args = parse_args()
    model_path = resolve_model_path(args.model)
    config = EngineConfig(
        model_path=model_path,
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
        progress_interval_ms=args.progress_interval_ms,
        root_topn=args.root_topn,
    )
    engine = ChessEngine(
        config,
        load_weights=bool(model_path),
    )

    root = tk.Tk()
    ChessBoardApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
