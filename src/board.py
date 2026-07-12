"""GUI and CLI chessboard simulator with optional model-assisted play."""

from __future__ import annotations

import argparse
import dataclasses
import io
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    count_pieces,
    safe_san,
    select_move,
)


@dataclasses.dataclass
class EngineConfig:
    model_path: Optional[str] = None
    device: str = DEVICE

    mcts_sims: int = 100
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    movetime_ms: int = 3000
    c_puct: float = 1.5
    alpha_beta_depth: int = 4
    alpha_beta_topk: int = 4
    alpha_beta_nodes: int = 20000
    alpha_beta_quiescence: int = 3
    alpha_beta_margin: float = 0.10
    alpha_beta_time_fraction: float = 0.25
    mate_guard_plies: int = 3
    q_tiebreak: bool = True
    q_tiebreak_min_visits: int = 32
    q_tiebreak_p_ratio: float = 0.90
    q_tiebreak_visit_ratio: float = 0.80
    q_tiebreak_margin: float = 0.25
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
    "mcts_sims": int,
    "mcts_min_sims": int,
    "mcts_batch_size": int,
    "movetime_ms": int,
    "c_puct": float,
    "alpha_beta_depth": int,
    "alpha_beta_topk": int,
    "alpha_beta_nodes": int,
    "alpha_beta_quiescence": int,
    "alpha_beta_margin": float,
    "alpha_beta_time_fraction": float,
    "mate_guard_plies": int,
    "q_tiebreak": bool_from_text,
    "q_tiebreak_min_visits": int,
    "q_tiebreak_p_ratio": float,
    "q_tiebreak_visit_ratio": float,
    "q_tiebreak_margin": float,
    "root_topn": int,
}


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
            mcts_sims=self.config.mcts_sims,
            mcts_min_sims=self.config.mcts_min_sims,
            mcts_batch_size=self.config.mcts_batch_size,
            time_limit=(
                self.config.movetime_ms / 1000.0
                if self.config.movetime_ms > 0
                else None
            ),
            c_puct=self.config.c_puct,
            alpha_beta_depth=self.config.alpha_beta_depth,
            alpha_beta_topk=self.config.alpha_beta_topk,
            alpha_beta_nodes=self.config.alpha_beta_nodes,
            alpha_beta_quiescence=self.config.alpha_beta_quiescence,
            alpha_beta_margin=self.config.alpha_beta_margin,
            alpha_beta_time_fraction=self.config.alpha_beta_time_fraction,
            mate_guard_plies=self.config.mate_guard_plies,
            q_tiebreak=self.config.q_tiebreak,
            q_tiebreak_min_visits=self.config.q_tiebreak_min_visits,
            q_tiebreak_p_ratio=self.config.q_tiebreak_p_ratio,
            q_tiebreak_visit_ratio=self.config.q_tiebreak_visit_ratio,
            q_tiebreak_margin=self.config.q_tiebreak_margin,
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
                    "alpha_beta_depth",
                    "alpha_beta_topk",
                    "alpha_beta_nodes",
                    "alpha_beta_quiescence",
                    "mate_guard_plies",
                    "q_tiebreak_min_visits",
                    "root_topn",
                } and value < 0:
                    raise ValueError(f"{name} must be non-negative")
                if name == "mcts_batch_size" and value < 1:
                    raise ValueError("mcts_batch_size must be at least 1")
                if name == "root_topn" and value < 1:
                    raise ValueError("root_topn must be at least 1")
                if (
                    name in {
                        "q_tiebreak_p_ratio",
                        "q_tiebreak_visit_ratio",
                    }
                    and not 0.0 <= value <= 1.0
                ):
                    raise ValueError(f"{name} must be between 0 and 1")
                if name == "q_tiebreak_margin" and value < 0.0:
                    raise ValueError("q_tiebreak_margin must be non-negative")
                if (
                    name == "alpha_beta_time_fraction"
                    and not 0.0 <= value <= 0.9
                ):
                    raise ValueError(
                        "alpha_beta_time_fraction must be between 0 and 0.9"
                    )
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
        if not self.model_loaded:
            raise RuntimeError("load a model and apply its parameters first")
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
            and self.model_loaded
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

    def choose_ai_move(self) -> Tuple[chess.Move, Dict]:
        if not self.model_loaded:
            raise RuntimeError("load a model and apply its parameters first")
        if not self.playing_with_ai:
            raise RuntimeError("start Play first")
        if not self.is_ai_turn():
            raise RuntimeError("the AI side is not to move")
        if self.game_over():
            raise RuntimeError("game is already over")

        live_fen = self.board.fen()
        move, info = select_move(
            self.board.copy(stack=True),
            self.model,
            self.search_options(),
            device=self.config.device,
        )
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

    def suggestions(self, topn: int = 8) -> Tuple[List[Dict], Dict]:
        if not self.model_loaded:
            raise RuntimeError("load a model and apply its parameters first")
        options = dataclasses.replace(
            self.search_options(),
            root_topn=max(1, int(topn)),
        )
        board_copy = self.board.copy(stack=True)
        move, info = select_move(
            board_copy,
            self.model,
            options,
            device=self.config.device,
        )
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
            "mcts_sims": "MCTS sims soft cap",
            "mcts_min_sims": "MCTS minimum sims",
            "mcts_batch_size": "MCTS batch size",
            "movetime_ms": "Movetime (ms)",
            "c_puct": "C-PUCT",
            "alpha_beta_depth": "Alpha-Beta depth",
            "alpha_beta_topk": "Alpha-Beta root candidates",
            "alpha_beta_nodes": "Alpha-Beta node cap",
            "alpha_beta_quiescence": "Quiescence depth",
            "alpha_beta_margin": "Alpha-Beta override margin",
            "alpha_beta_time_fraction": "Alpha-Beta time fraction",
            "mate_guard_plies": "Mate guard plies",
            "q_tiebreak": "Q tiebreak",
            "q_tiebreak_min_visits": "Q tiebreak min visits",
            "q_tiebreak_p_ratio": "Q tiebreak p ratio",
            "q_tiebreak_visit_ratio": "Q tiebreak visit ratio",
            "q_tiebreak_margin": "Q tiebreak margin",
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
        self.square_size = 72
        self.board_size = self.square_size * 8
        self.flipped = False
        self.selected_square: Optional[chess.Square] = None
        self.legal_targets: List[chess.Square] = []
        self.last_move = self.engine.last_move()

        self.ai_thinking = False
        self.ai_worker = None
        self.ai_task: Optional[str] = None
        self.pending_ai_after_id = None
        self.buttons: List[ttk.Button] = []

        self.light = "#EEEED2"
        self.dark = "#769656"
        self.selected_color = "#F6F669"
        self.target_color = "#BACA44"
        self.lastmove_color = "#CDD26A"

        self._build_ui()
        self.draw_board()
        self.refresh_controls()
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
        base_state = tk.DISABLED if self.ai_thinking else tk.NORMAL
        for button in self.buttons:
            try:
                button.configure(state=base_state)
            except Exception:
                pass

        if not self.ai_thinking:
            model_state = (
                tk.NORMAL
                if self.engine.model_loaded
                else tk.DISABLED
            )
            self.play_ai_button.configure(state=model_state)
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
            if self.engine.mode == "simulator" and self.ai_task != "move"
            else tk.DISABLED
        )
        self.ai_suggest_button.configure(text=label, state=state)

    def cancel_pending_ai(self):
        if self.pending_ai_after_id is not None:
            try:
                self.root.after_cancel(self.pending_ai_after_id)
            except Exception:
                pass
            self.pending_ai_after_id = None

    def schedule_ai_reply(self):
        self.cancel_pending_ai()
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
            and self.engine.model_loaded
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
            f"MCTS sims: {info.get('sims_completed')}/"
            f"{info.get('mcts_dynamic_target')}/"
            f"{info.get('mcts_soft_cap')}\n"
            f"Uncertainty: {info.get('uncertainty')}\n"
            f"Value: {info.get('value')}\n"
            f"Expanded nodes: {info.get('nodes')}\n"
            f"NN batches: {info.get('nn_batches')}\n"
            f"Alpha-Beta nodes: {info.get('alpha_beta_nodes')}\n"
            f"Alpha-Beta override: "
            f"{info.get('alpha_beta_overrode_mcts')}\n"
            f"Q tiebreak: {info.get('q_tiebreak_overrode')} "
            f"{info.get('q_tiebreak_move')}\n"
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
        if self.ai_thinking or self.engine.game_over():
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
        if self.ai_thinking:
            return
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
        if self.ai_thinking or not self.engine.is_ai_turn():
            return

        self.cancel_pending_ai()
        before_board = self.engine.board.copy(stack=True)
        before_fen = self.engine.board.fen()
        before_turn = self.engine.board.turn
        before_length = len(self.engine.board.move_stack)
        self.ai_thinking = True
        self.ai_task = "move"
        self.clear_selection()
        self.draw_board()

        def worker():
            payload = {
                "ok": False,
                "move": None,
                "info": None,
                "error": None,
                "before_board": before_board,
                "before_fen": before_fen,
                "before_turn": before_turn,
                "before_length": before_length,
            }
            try:
                move, info = self.engine.choose_ai_move()
                payload["move"] = move.uci()
                payload["info"] = info
                payload["ok"] = True
            except Exception as exc:
                payload["error"] = str(exc)
            self.root.after(
                0,
                lambda result=payload: self.finish_ai_move(result),
            )

        self.ai_worker = threading.Thread(target=worker, daemon=True)
        self.ai_worker.start()

    def finish_ai_move(self, payload):
        try:
            if self.engine.board.fen() != payload.get("before_fen"):
                raise RuntimeError("board changed during AI search")
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
            self.ai_thinking = False
            self.ai_worker = None
            self.ai_task = None
            self.draw_board()
            self.schedule_ai_reply()

    def start_simulator_suggestion(self):
        if self.ai_thinking:
            return
        if (
            self.engine.mode != "simulator"
            or self.engine.game_over()
            or not self.engine.model_loaded
            or not self.engine.ai_suggest_open
        ):
            return

        self.cancel_pending_ai()
        before_fen = self.engine.board.fen()
        self.ai_thinking = True
        self.ai_task = "suggestion"
        self.draw_board()

        def worker():
            payload = {
                "ok": False,
                "info": None,
                "error": None,
                "before_fen": before_fen,
            }
            try:
                _suggestions, info = self.engine.suggestions(
                    topn=self.engine.config.root_topn
                )
                payload["info"] = info
                payload["ok"] = True
            except Exception as exc:
                payload["error"] = str(exc)
            self.root.after(
                0,
                lambda result=payload: self.finish_suggestions(result),
            )

        self.ai_worker = threading.Thread(target=worker, daemon=True)
        self.ai_worker.start()

    def finish_suggestions(self, payload):
        error_message = None
        try:
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
            error_message = f"Model analysis error: {exc}"
        finally:
            self.ai_thinking = False
            self.ai_worker = None
            self.ai_task = None
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
        if self.ai_thinking:
            return
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
        if not self.engine.model_loaded:
            messagebox.showinfo(
                "Play",
                "Load a model and apply its parameters first.",
                parent=self.root,
            )
            return

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
        if self.ai_thinking:
            return
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
        if self.ai_thinking:
            return
        self.cancel_pending_ai()
        undone = self.engine.undo()
        self.clear_selection()
        self.sync_last_move()
        self.update_analysis(None)
        self.draw_board()
        self.schedule_ai_reply()

    def import_pgn(self):
        if self.ai_thinking:
            return
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


def print_cli_help():
    print(
        """
Commands:
  <move>
      Play the legal move for the side to move. UCI and SAN are accepted.

  model <path>
      Load or reload a model with the current search parameters.

  unload
      Unload the model and enter simulator mode.

  params
      Show the current model search parameters.

  set <name> <value>
      Update one search parameter and reload the current model.

  play
      Choose White or Black and choose current position or startpos.

  simulator
      Return to two-sided board simulation.

  close
      Close simulator AI suggest output.

  open
      Open simulator AI suggest output and analyze the current position.

  reset [fen]
      Restore the supplied FEN. Empty input restores startpos.

  pgn <path>
      Load the first PGN mainline and display its final position.

  undo
      Undo one simulator ply or the latest human/AI turn pair.

  save <path>
      Save the current board history as PGN.

  state
      Show mode, model, FEN, side to move, and result.

  board
      Show the ASCII board.

  help
      Show this command list.

  quit
      Exit.
"""
    )


def print_search_info(info: Dict):
    print(
        "search=", info.get("search_type"),
        "best=", info.get("best_san"),
        info.get("best_move"),
        "sims=", (
            info.get("sims_completed"),
            info.get("mcts_dynamic_target"),
            info.get("mcts_soft_cap"),
        ),
        "uncertainty=", info.get("uncertainty"),
        "value=", info.get("value"),
        "q_tiebreak=", info.get("q_tiebreak_move"),
    )


def print_suggestions(suggestions: List[Dict]):
    for index, row in enumerate(suggestions, 1):
        print(
            f"{index}. {row.get('san')} {row.get('move')} "
            f"p={row.get('p', 0.0):.4f} "
            f"visits={row.get('visits', 0)} "
            f"q={row.get('q', 0.0):+.3f}"
        )


def cli_simulator_suggest(engine: ChessEngine):
    if (
        engine.mode != "simulator"
        or not engine.model_loaded
        or not engine.ai_suggest_open
        or engine.game_over()
    ):
        return
    try:
        suggestions, info = engine.suggestions(
            topn=engine.config.root_topn
        )
        print_search_info(info)
        print_suggestions(suggestions)
    except Exception as exc:
        print("Suggestion error:", exc)


def cli_ai_reply(engine: ChessEngine):
    if not engine.is_ai_turn() or engine.game_over():
        return
    output = engine.make_ai_move()
    move = output["move"]
    print(f"AI: {move['san']} ({move['uci']})")
    print_search_info(output["info"])


def cli_after_position_change(engine: ChessEngine):
    if engine.is_ai_turn():
        cli_ai_reply(engine)
    else:
        cli_simulator_suggest(engine)


def prompt_ai_setup(engine: ChessEngine):
    if not engine.model_loaded:
        print("Load a model and apply its parameters first.")
        return

    side = input("Your side [white/black]: ").strip().lower()
    if side not in {"white", "w", "black", "b"}:
        print("Expected white or black.")
        return
    user_side = "white" if side in {"white", "w"} else "black"

    start = input(
        "Starting position [current/startpos]: "
    ).strip().lower()
    if start not in {"current", "c", "startpos", "start", "s", ""}:
        print("Expected current or startpos.")
        return
    from_current = start in {"current", "c"}

    engine.start_ai_game(
        user_side=user_side,
        from_current_position=from_current,
    )
    print(
        f"Play: human={side_name(engine.user_color)}, "
        f"ai={side_name(engine.ai_color)}, "
        f"start={'current' if from_current else 'startpos'}"
    )
    cli_ai_reply(engine)


def run_cli_app(engine: ChessEngine):
    print("Chessboard Simulator CLI")
    print("Mode: simulator")
    print("Type 'help' for commands.")
    cli_simulator_suggest(engine)

    while True:
        if engine.game_over():
            print(engine.outcome_text())

        raw = input("board> ")
        text = raw.strip()
        if not text:
            continue
        lower = text.lower()

        if lower in {"q", "quit", "exit"}:
            break
        if lower in {"help", "h", "?"}:
            print_cli_help()
            continue
        if lower == "board":
            print(engine.board)
            continue
        if lower == "state":
            state = engine.state()
            for key, value in state.items():
                print(f"{key}: {value}")
            continue
        if lower == "params":
            for key, value in engine.parameter_dict().items():
                print(f"{key}: {value}")
            continue
        if lower in {"play", "playai", "ai game"}:
            try:
                prompt_ai_setup(engine)
            except Exception as exc:
                print("Play error:", exc)
            continue
        if lower in {"simulator", "simulate"}:
            engine.enter_simulator()
            print("Mode: simulator")
            cli_simulator_suggest(engine)
            continue
        if lower == "close":
            engine.close_ai_suggest()
            print("AI suggest: closed")
            continue
        if lower == "open":
            engine.open_ai_suggest()
            print("AI suggest: open")
            cli_simulator_suggest(engine)
            continue
        if lower == "unload":
            engine.unload_model()
            print("Model unloaded. Mode: simulator")
            continue
        if lower == "undo":
            print(f"undone {engine.undo()} ply")
            cli_after_position_change(engine)
            continue
        if lower == "reset":
            try:
                engine.reset("startpos")
                print("FEN:", engine.board.fen())
                cli_after_position_change(engine)
            except Exception as exc:
                print("Reset error:", exc)
            continue
        if lower.startswith("reset "):
            try:
                fen = strip_wrapping_quotes(text[6:].strip())
                engine.reset(fen or "startpos")
                print("FEN:", engine.board.fen())
                cli_after_position_change(engine)
            except Exception as exc:
                print("Reset error:", exc)
            continue
        if lower.startswith("model "):
            path = strip_wrapping_quotes(text[6:].strip())
            try:
                loaded = engine.load_model(path)
                print("Model loaded:", loaded)
                cli_simulator_suggest(engine)
            except Exception as exc:
                print("Model error:", exc)
            continue
        if lower.startswith("set "):
            parts = text.split(maxsplit=2)
            if len(parts) != 3:
                print("usage: set <name> <value>")
                continue
            name, value = parts[1], parts[2]
            if name not in SEARCH_PARAMETER_TYPES:
                print(
                    "Available parameters:",
                    ", ".join(SEARCH_PARAMETER_TYPES),
                )
                continue
            if not engine.model_loaded:
                print("Load a model before applying parameters.")
                continue
            try:
                converter = SEARCH_PARAMETER_TYPES[name]
                converted = converter(value)
                engine.configure_model(
                    engine.model_path,
                    {name: converted},
                )
                print(
                    f"{name}={getattr(engine.config, name)}; "
                    "model reloaded"
                )
                cli_simulator_suggest(engine)
            except Exception as exc:
                print("Parameter error:", exc)
            continue
        if lower.startswith("pgn "):
            path = strip_wrapping_quotes(text[4:].strip())
            try:
                engine.load_pgn_file(path)
                print("FEN:", engine.board.fen())
                cli_after_position_change(engine)
            except Exception as exc:
                print("PGN error:", exc)
            continue
        if lower.startswith("save "):
            path = strip_wrapping_quotes(text[5:].strip())
            try:
                engine.save_pgn(path)
                print("saved:", path)
            except Exception as exc:
                print("Save error:", exc)
            continue

        try:
            result = engine.make_human_move(text)
            print(f"Move: {result['san']} ({result['uci']})")
            cli_after_position_change(engine)
        except Exception as exc:
            print("Move error:", exc)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chessboard simulator with optional model assistance"
    )
    parser.add_argument("--model", default="none")
    parser.add_argument(
        "--gui",
        type=int,
        choices=[0, 1],
        default=1,
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--mcts-min-sims", type=int, default=0)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=3000)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--alpha-beta-depth", type=int, default=4)
    parser.add_argument("--alpha-beta-topk", type=int, default=4)
    parser.add_argument("--alpha-beta-nodes", type=int, default=20000)
    parser.add_argument("--alpha-beta-quiescence", type=int, default=3)
    parser.add_argument("--alpha-beta-margin", type=float, default=0.10)
    parser.add_argument(
        "--alpha-beta-time-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument("--mate-guard-plies", type=int, default=3)
    parser.add_argument("--q-tiebreak", action="store_true", default=True)
    parser.add_argument("--no-q-tiebreak", dest="q_tiebreak", action="store_false")
    parser.add_argument("--q-tiebreak-min-visits", type=int, default=32)
    parser.add_argument("--q-tiebreak-p-ratio", type=float, default=0.90)
    parser.add_argument("--q-tiebreak-visit-ratio", type=float, default=0.80)
    parser.add_argument("--q-tiebreak-margin", type=float, default=0.25)
    parser.add_argument("--root-topn", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model)
    config = EngineConfig(
        model_path=model_path,
        device=args.device,
        mcts_sims=args.mcts_sims,
        mcts_min_sims=args.mcts_min_sims,
        mcts_batch_size=args.mcts_batch_size,
        movetime_ms=args.movetime_ms,
        c_puct=args.c_puct,
        alpha_beta_depth=args.alpha_beta_depth,
        alpha_beta_topk=args.alpha_beta_topk,
        alpha_beta_nodes=args.alpha_beta_nodes,
        alpha_beta_quiescence=args.alpha_beta_quiescence,
        alpha_beta_margin=args.alpha_beta_margin,
        alpha_beta_time_fraction=args.alpha_beta_time_fraction,
        mate_guard_plies=args.mate_guard_plies,
        q_tiebreak=args.q_tiebreak,
        q_tiebreak_min_visits=args.q_tiebreak_min_visits,
        q_tiebreak_p_ratio=args.q_tiebreak_p_ratio,
        q_tiebreak_visit_ratio=args.q_tiebreak_visit_ratio,
        q_tiebreak_margin=args.q_tiebreak_margin,
        root_topn=args.root_topn,
    )
    engine = ChessEngine(
        config,
        load_weights=bool(model_path),
    )

    if args.gui == 0:
        run_cli_app(engine)
        return

    root = tk.Tk()
    ChessBoardApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
