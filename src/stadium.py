from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import chess
import chess.engine
import chess.pgn
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from gui import ChessGUIBase


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


def strip_wrapping_quotes(text: str) -> str:
    value = str(text).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


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
    configured_options: Dict[str, object],
) -> int:
    option = engine.options.get("MultiPV")
    if option is None:
        return 1
    value = option.default
    for name, configured_value in configured_options.items():
        if str(name).lower() == "multipv":
            value = configured_value
            break
    count = int(value)
    if option.min is not None:
        count = max(int(option.min), count)
    if option.max is not None:
        count = min(int(option.max), count)
    return max(1, count)


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
):
    analysis = engine.analysis(
        board,
        chess.engine.Limit(time=movetime_ms / 1000.0),
        multipv=max(1, int(multipv)),
        info=chess.engine.INFO_ALL,
    )
    best = analysis.wait()
    return best.move, [dict(info) for info in analysis.multipv]


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


class StadiumSettingsDialog(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Stadium Settings")
        self.resizable(True, False)
        self.transient(parent)
        self.grab_set()

        self.values = {}
        fields = (
            ("white_uci", "White UCI command"),
            ("white_options", "White UCI options (JSON)"),
            ("white_movetime_ms", "White move time (ms)"),
            ("black_uci", "Black UCI command"),
            ("black_options", "Black UCI options (JSON)"),
            ("black_movetime_ms", "Black move time (ms)"),
            ("delay_ms", "Display delay (ms)"),
            ("max_plies", "Max plies"),
        )
        for row, (name, label) in enumerate(fields):
            ttk.Label(self, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=8,
                pady=5,
            )
            variable = tk.StringVar(value=str(app.settings[name]))
            self.values[name] = variable
            entry = ttk.Entry(self, textvariable=variable, width=72)
            entry.grid(row=row, column=1, sticky="ew", padx=8, pady=5)
            if name in {"white_uci", "black_uci"}:
                ttk.Button(
                    self,
                    text="Browse",
                    command=lambda target=entry: app.browse_uci(target),
                ).grid(row=row, column=2, padx=8, pady=5)

        actions = ttk.Frame(self)
        actions.grid(row=len(fields), column=0, columnspan=3, sticky="e", padx=8, pady=8)
        ttk.Button(actions, text="Apply", command=self.apply).pack(side=tk.LEFT, padx=3)
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=3)
        self.columnconfigure(1, weight=1)

    def apply(self):
        try:
            settings = {
                "white_uci": self.values["white_uci"].get().strip(),
                "white_options": self.values["white_options"].get().strip(),
                "white_movetime_ms": int(self.values["white_movetime_ms"].get()),
                "black_uci": self.values["black_uci"].get().strip(),
                "black_options": self.values["black_options"].get().strip(),
                "black_movetime_ms": int(self.values["black_movetime_ms"].get()),
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
    ):
        self.root = root
        self.engine = engine
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.initialize_board_gui()
        self.settings = {
            "white_uci": str(defaults.get("white_uci") or ""),
            "white_options": str(defaults.get("white_options") or "{}"),
            "white_movetime_ms": int(defaults.get("white_movetime_ms", 1000)),
            "black_uci": str(defaults.get("black_uci") or ""),
            "black_options": str(defaults.get("black_options") or "{}"),
            "black_movetime_ms": int(defaults.get("black_movetime_ms", 1000)),
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
        self._build_ui()
        self.draw_board()
        self.refresh_controls()
        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)

    @property
    def arena_engine(self) -> StadiumBoardState:
        return self.engine

    def _build_ui(self):
        self.root.title("Gadidae Stadium")
        container = ttk.Frame(self.root, padding=8)
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
            elif kind == "move":
                self.apply_uci_move(payload)
            elif kind == "finished":
                self.finish_game(payload["result"], payload["termination"])
            elif kind == "error":
                self.fail_game(payload)
        self.root.after(self.ui_poll_interval_ms, self.process_ui_events)

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
        black_movetime_ms = int(self.settings["black_movetime_ms"])
        delay_ms = int(self.settings["delay_ms"])
        max_plies = int(self.settings["max_plies"])
        if white_movetime_ms <= 0:
            raise ValueError("white move time must be positive")
        if black_movetime_ms <= 0:
            raise ValueError("black move time must be positive")
        if delay_ms < 0:
            raise ValueError("display delay must be non-negative")
        if max_plies <= 0:
            raise ValueError("max plies must be positive")
        return (
            white,
            white_options,
            white_movetime_ms,
            black,
            black_options,
            black_movetime_ms,
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
                black,
                black_options,
                black_movetime_ms,
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
                black,
                black_options,
                black_movetime_ms,
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
        black_command: Sequence[str],
        black_options: Dict[str, object],
        black_movetime_ms: int,
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
            if white_options:
                white.configure(white_options)
            if black_options:
                black.configure(black_options)
            white_multipv = engine_multipv_count(white, white_options)
            black_multipv = engine_multipv_count(black, black_options)
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
            while not board.is_game_over(claim_draw=True) and ply < max_plies:
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
                move, multipv_infos = analyse_uci_turn(
                    uci,
                    board,
                    movetime_ms=movetime_ms,
                    multipv=multipv,
                )
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if move is None or move not in board.legal_moves:
                    raise RuntimeError(
                        f"{name} returned an illegal move: "
                        f"{move.uci() if move is not None else 'none'}"
                    )
                san = safe_san(board, move)
                primary_info = primary_info_for_move(multipv_infos, move)
                detail = engine_info_text(name, board, move, primary_info, elapsed_ms)
                move_rows = multipv_move_rows(board, multipv_infos, move)
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
                        "detail": detail,
                        "multipv": move_rows,
                    },
                ))
                if delay_ms > 0 and self.stop_event.wait(delay_ms / 1000.0):
                    break

            if self.stop_event.is_set():
                result_text, termination = "*", "stopped"
            else:
                outcome = board.outcome(claim_draw=True)
                if outcome is not None:
                    result_text = board.result(claim_draw=True)
                    termination = outcome.termination.name.lower().replace("_", " ")
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
        move_number = (int(payload["ply"]) + 1) // 2
        side = "White" if int(payload["ply"]) % 2 else "Black"
        self.moves_list.delete(0, tk.END)
        move_rows = list(payload.get("multipv") or [])
        if not move_rows:
            move_rows = [
                f"*1. {payload['san']} ({move.uci()})  {side} move {move_number}"
            ]
        for row in move_rows:
            self.moves_list.insert(tk.END, row)
        self.moves_list.see(tk.END)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, payload["detail"])
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
        self.stop_event.set()
        self.resume_event.set()
        with self.uci_lock:
            engines = list(self.uci_engines)
        for uci in engines:
            try:
                uci.close()
            except Exception:
                pass
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watch one game between any two UCI engines"
    )
    parser.add_argument("--white-uci", default="")
    parser.add_argument("--white-options", default="{}")
    parser.add_argument("--white-movetime-ms", type=int, default=1000)
    parser.add_argument("--black-uci", default="")
    parser.add_argument("--black-options", default="{}")
    parser.add_argument("--black-movetime-ms", type=int, default=1000)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--delay-ms", type=int, default=250)
    parser.add_argument("--max-plies", type=int, default=240)
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    StadiumApp(
        root,
        StadiumBoardState(),
        {
            "white_uci": args.white_uci,
            "white_options": args.white_options,
            "white_movetime_ms": args.white_movetime_ms,
            "black_uci": args.black_uci,
            "black_options": args.black_options,
            "black_movetime_ms": args.black_movetime_ms,
            "fen": args.fen,
            "delay_ms": args.delay_ms,
            "max_plies": args.max_plies,
        },
    )
    root.mainloop()


if __name__ == "__main__":
    main()
