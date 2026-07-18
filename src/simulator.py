"""GUI chessboard simulator with optional model analysis."""

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
from gui import ChessGUIBase
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
        repetition_policy_penalty=float(
            parameters.get("repetition_policy_penalty", 0.0) or 0.0
        ),
        instant_mate_first=bool_from_text(
            parameters.get("instant_mate_first", False)
        ),
        progress_interval_sec=max(
            0.0,
            float(parameters.get("progress_interval_ms", 750) or 0) / 1000.0,
        ),
        root_topn=max(1, int(root_topn or parameters.get("root_topn", 8) or 8)),
    )


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


def run_simulator_search_job(
    job: Dict,
    output_queue,
    cancel_event,
    model,
):
    try:
        board = chess.Board(str(job["fen"]))
        parameters = dict(job["parameters"])
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
    model = None
    model_key = None

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
        model_path = str(job.get("model_path") or "").strip()
        if model_path.lower() == "none":
            model_path = ""
        device = str(parameters.get("device", DEVICE))
        model_revision = int(job.get("model_revision", 0))
        next_model_key = (model_path, device, model_revision)

        try:
            if next_model_key != model_key:
                model = load_model(model_path, device=device) if model_path else None
                model_key = next_model_key
            if cancel_event.is_set():
                continue
            run_simulator_search_job(job, output_queue, cancel_event, model)
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


class SimulatorState:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        load_weights: bool = False,
    ):
        self.config = config or EngineConfig()
        self.board = chess.Board()
        self.model = None
        self.model_path: Optional[str] = None
        self.model_revision = 0
        self.last_ai_info: Optional[Dict] = None
        self.last_suggestions: List[Dict] = []
        self.ai_suggest_open = True

        if load_weights and self.config.model_path:
            self.load_model(self.config.model_path)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and bool(self.model_path)

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
            repetition_policy_penalty=self.config.repetition_policy_penalty,
            instant_mate_first=self.config.instant_mate_first,
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
        self.model_revision += 1
        self.clear_analysis()
        return resolved

    def unload_model(self):
        self.model = None
        self.model_path = None
        self.config.model_path = None
        self.model_revision += 1
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

    def undo(self) -> int:
        if not self.board.move_stack:
            return 0
        self.board.pop()
        self.clear_analysis()
        return 1

    def last_move(self) -> Optional[chess.Move]:
        return self.board.move_stack[-1] if self.board.move_stack else None

    def game_over(self) -> bool:
        return bool(
            self.board.is_game_over(claim_draw=False)
            or self.board.is_repetition(3)
            or self.board.is_fifty_moves()
        )

    def result(self) -> str:
        if self.board.is_game_over(claim_draw=False):
            return self.board.result(claim_draw=False)
        if self.board.is_repetition(3) or self.board.is_fifty_moves():
            return "1/2-1/2"
        return "*"

    def outcome_text(self) -> str:
        if not self.game_over():
            return "Game in progress"
        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            return f"{self.result()} - {outcome.termination.name}"
        if self.board.is_repetition(3):
            return "1/2-1/2 - THREEFOLD_REPETITION"
        if self.board.is_fifty_moves():
            return "1/2-1/2 - FIFTY_MOVES"
        return self.result()

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
        game.headers["Event"] = "Gadidae Simulator"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
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
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "ai_suggest": "open" if self.ai_suggest_open else "closed",
            "piece_count": count_pieces(self.board),
            "legal_moves": len(list(self.board.legal_moves)),
            "game_over": self.game_over(),
            "result": self.result() if self.game_over() else "*",
            "outcome": self.outcome_text(),
        }


class ModelSettingsDialog(tk.Toplevel):
    def __init__(self, parent, engine: SimulatorState, on_applied):
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
            "repetition_policy_penalty": "Repetition policy penalty",
            "instant_mate_first": "Instant Mate First",
            "progress_interval_ms": "Progress interval (ms)",
            "root_topn": "Suggestion count",
        }
        current = engine.parameter_dict()
        for name, converter in SEARCH_PARAMETER_TYPES.items():
            ttk.Label(self, text=labels[name]).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=3,
            )
            if converter is bool_from_text:
                variable = tk.BooleanVar(value=bool(current[name]))
                control = ttk.Checkbutton(self, variable=variable)
            else:
                variable = tk.StringVar(value=str(current[name]))
                control = ttk.Entry(self, textvariable=variable, width=24)
            self.variables[name] = variable
            control.grid(
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
        parameters = {}
        for name, variable in self.variables.items():
            value = variable.get()
            parameters[name] = value.strip() if isinstance(value, str) else value
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


class SimulatorApp(ChessGUIBase):
    def __init__(self, root: tk.Tk, engine: SimulatorState):
        self.root = root
        self.engine = engine

        self.root.title("Gadidae Simulator")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.initialize_board_gui()

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
        self._build_ui()
        self.draw_board()
        self.refresh_controls()
        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)
        self.schedule_ai_reply()

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
        self.cancel_pending_ai()
        self.stop_ai_worker()
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
            "model_path": self.engine.model_path,
            "model_revision": self.engine.model_revision,
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

        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)

    def schedule_ai_reply(self):
        self.cancel_pending_ai(cancel_running=False)
        if (
            self.ai_thinking
            or self.engine.game_over()
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
                f"score={row.get('decision_score', row.get('p', 0.0)):.4f}  "
                f"visits={row.get('visits', 0)}  "
                f"q={row.get('q', 0.0):+.3f}",
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
            error_message = f"Model analysis error: {exc}"
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
            self.schedule_ai_reply()
            messagebox.showinfo(
                "Model",
                "Model unloaded.",
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chess position simulator with optional model analysis"
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
    parser.add_argument("--repetition-policy-penalty", type=float, default=0.0)
    parser.add_argument(
        "--instant-mate-first",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
        repetition_policy_penalty=args.repetition_policy_penalty,
        instant_mate_first=args.instant_mate_first,
        progress_interval_ms=args.progress_interval_ms,
        root_topn=args.root_topn,
    )
    engine = SimulatorState(
        config,
        load_weights=bool(model_path),
    )

    root = tk.Tk()
    SimulatorApp(root, engine)
    root.mainloop()


if __name__ == "__main__":
    main()
