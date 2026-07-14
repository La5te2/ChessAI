from __future__ import annotations

import argparse
import contextlib
import shlex
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import chess

from config import CPUCT, DEFAULT_SIMS, DEVICE
from model import load_model
from move_encoder import move_to_index
from search import SearchOptions, UnifiedSearch, VALID_SEARCH_TYPES


def uci_print(text: str):
    print(text, flush=True)


def as_int(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def as_float(value, default: float) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def as_bool(value, default: bool) -> bool:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return bool(default)


def normalize_option_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


@dataclass
class EngineConfig:
    model_path: str = ""
    device: str = DEVICE
    search_type: str = "only-mcts"
    mcts_sims: int = DEFAULT_SIMS
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    movetime_ms: int = 1000
    move_overhead_ms: int = 50
    min_movetime_ms: int = 50
    max_movetime_ms: int = 10000
    time_divisor: float = 30.0
    increment_fraction: float = 0.75
    c_puct: float = CPUCT
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_reduction: float = 0.15
    virtual_loss: float = 0.0
    mcts_time_fraction: float = 0.90
    mate_plies: int = 0
    mate_topk: int = 4
    mate_nodes: int = 20000
    root_topn: int = 5
    log_search: bool = False


class UCIEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.board = chess.Board()
        self.debug = False
        self.model = None
        self.loaded_model_path: Optional[str] = None
        self.loaded_device: Optional[str] = None

    def option_lines(self) -> List[str]:
        cfg = self.config
        return [
            f"option name ModelPath type string default {cfg.model_path}",
            f"option name Device type string default {cfg.device}",
            f"option name SearchType type combo default {cfg.search_type} var closed var only-mcts var mcts-mate",
            f"option name MCTSSims type spin default {cfg.mcts_sims} min 0 max 1000000",
            f"option name MCTSMinSims type spin default {cfg.mcts_min_sims} min 0 max 1000000",
            f"option name MCTSBatchSize type spin default {cfg.mcts_batch_size} min 1 max 4096",
            f"option name MoveTimeMS type spin default {cfg.movetime_ms} min 0 max 3600000",
            f"option name MoveOverheadMS type spin default {cfg.move_overhead_ms} min 0 max 60000",
            f"option name MinMoveTimeMS type spin default {cfg.min_movetime_ms} min 0 max 3600000",
            f"option name MaxMoveTimeMS type spin default {cfg.max_movetime_ms} min 0 max 3600000",
            f"option name TimeDivisor type string default {cfg.time_divisor}",
            f"option name IncrementFraction type string default {cfg.increment_fraction}",
            f"option name CPuct type string default {cfg.c_puct}",
            f"option name CPuctBase type string default {cfg.c_puct_base}",
            f"option name CPuctFactor type string default {cfg.c_puct_factor}",
            f"option name FPUReduction type string default {cfg.fpu_reduction}",
            f"option name VirtualLoss type string default {cfg.virtual_loss}",
            f"option name MCTSTimeFraction type string default {cfg.mcts_time_fraction}",
            f"option name MatePlies type spin default {cfg.mate_plies} min 0 max 64",
            f"option name MateTopK type spin default {cfg.mate_topk} min 0 max 256",
            f"option name MateNodes type spin default {cfg.mate_nodes} min 0 max 10000000",
            f"option name RootTopN type spin default {cfg.root_topn} min 1 max 256",
            f"option name LogSearch type check default {'true' if cfg.log_search else 'false'}",
        ]

    def handle_uci(self):
        uci_print("id name Gadidae")
        uci_print("id author La5te2")
        for line in self.option_lines():
            uci_print(line)
        uci_print("uciok")

    def _mark_model_dirty(self):
        self.model = None
        self.loaded_model_path = None
        self.loaded_device = None

    def set_option(self, name: str, value: str):
        key = normalize_option_name(name)
        cfg = self.config
        previous_model = (cfg.model_path, cfg.device)
        if key == "modelpath":
            cfg.model_path = str(value).strip()
        elif key == "device":
            cfg.device = str(value).strip() or DEVICE
        elif key == "searchtype":
            candidate = str(value).strip().lower()
            if candidate in VALID_SEARCH_TYPES:
                cfg.search_type = candidate
            else:
                uci_print(f"info string invalid SearchType: {value}")
        elif key == "mctssims":
            cfg.mcts_sims = max(0, as_int(value, cfg.mcts_sims))
        elif key == "mctsminsims":
            cfg.mcts_min_sims = max(0, as_int(value, cfg.mcts_min_sims))
        elif key == "mctsbatchsize":
            cfg.mcts_batch_size = max(1, as_int(value, cfg.mcts_batch_size))
        elif key == "movetimems":
            cfg.movetime_ms = max(0, as_int(value, cfg.movetime_ms))
        elif key == "moveoverheadms":
            cfg.move_overhead_ms = max(0, as_int(value, cfg.move_overhead_ms))
        elif key == "minmovetimems":
            cfg.min_movetime_ms = max(0, as_int(value, cfg.min_movetime_ms))
        elif key == "maxmovetimems":
            cfg.max_movetime_ms = max(0, as_int(value, cfg.max_movetime_ms))
        elif key == "timedivisor":
            cfg.time_divisor = max(1.0, as_float(value, cfg.time_divisor))
        elif key == "incrementfraction":
            cfg.increment_fraction = max(0.0, as_float(value, cfg.increment_fraction))
        elif key == "cpuct":
            cfg.c_puct = max(0.0, as_float(value, cfg.c_puct))
        elif key == "cpuctbase":
            cfg.c_puct_base = max(1.0, as_float(value, cfg.c_puct_base))
        elif key == "cpuctfactor":
            cfg.c_puct_factor = max(0.0, as_float(value, cfg.c_puct_factor))
        elif key == "fpureduction":
            cfg.fpu_reduction = max(0.0, as_float(value, cfg.fpu_reduction))
        elif key == "virtualloss":
            cfg.virtual_loss = max(0.0, as_float(value, cfg.virtual_loss))
        elif key == "mctstimefraction":
            cfg.mcts_time_fraction = max(0.0, min(1.0, as_float(value, cfg.mcts_time_fraction)))
        elif key == "mateplies":
            cfg.mate_plies = max(0, as_int(value, cfg.mate_plies))
        elif key == "matetopk":
            cfg.mate_topk = max(0, as_int(value, cfg.mate_topk))
        elif key == "matenodes":
            cfg.mate_nodes = max(0, as_int(value, cfg.mate_nodes))
        elif key == "roottopn":
            cfg.root_topn = max(1, as_int(value, cfg.root_topn))
        elif key == "logsearch":
            cfg.log_search = as_bool(value, cfg.log_search)
        else:
            uci_print(f"info string unknown option: {name}")
            return

        if (cfg.model_path, cfg.device) != previous_model:
            self._mark_model_dirty()

    def parse_setoption(self, line: str):
        tokens = line.split()
        if "name" not in tokens:
            return
        name_index = tokens.index("name") + 1
        value_index = tokens.index("value") if "value" in tokens else len(tokens)
        name = " ".join(tokens[name_index:value_index]).strip()
        value = " ".join(tokens[value_index + 1:]).strip() if value_index < len(tokens) else ""
        if name:
            self.set_option(name, value)

    def ensure_model(self):
        path = self.config.model_path.strip()
        device = self.config.device.strip() or DEVICE
        if path.lower() == "none":
            self.model = None
            self.loaded_model_path = path
            self.loaded_device = device
            return None
        if not path:
            return None
        if (
            self.model is not None
            and self.loaded_model_path == path
            and self.loaded_device == device
        ):
            return self.model

        print(f"loading model {path} on {device}", file=sys.stderr, flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            self.model = load_model(path, device=device)
        self.loaded_model_path = path
        self.loaded_device = device
        print("model ready", file=sys.stderr, flush=True)
        return self.model

    def set_position(self, line: str):
        tokens = shlex.split(line)
        if len(tokens) < 2:
            return

        moves_index = tokens.index("moves") if "moves" in tokens else -1
        moves = tokens[moves_index + 1:] if moves_index >= 0 else []

        try:
            if tokens[1] == "startpos":
                board = chess.Board()
            elif tokens[1] == "fen":
                fen_tokens = tokens[2:moves_index] if moves_index >= 0 else tokens[2:]
                fen = " ".join(fen_tokens)
                board = chess.Board(fen)
            else:
                uci_print(f"info string unsupported position command: {line}")
                return

            for move_text in moves:
                move = chess.Move.from_uci(move_text)
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move {move_text} in {board.fen()}")
                board.push(move)
            self.board = board
        except Exception as exc:
            uci_print(f"info string position error: {exc}")

    def search_options(self, movetime_ms: int, sims_override: Optional[int] = None) -> SearchOptions:
        cfg = self.config
        sims = cfg.mcts_sims if sims_override is None else max(0, int(sims_override))
        return SearchOptions(
            search_type=cfg.search_type,
            mcts_sims=sims,
            mcts_min_sims=cfg.mcts_min_sims,
            mcts_batch_size=cfg.mcts_batch_size,
            time_limit=(movetime_ms / 1000.0) if movetime_ms > 0 else None,
            c_puct=cfg.c_puct,
            c_puct_base=cfg.c_puct_base,
            c_puct_factor=cfg.c_puct_factor,
            fpu_reduction=cfg.fpu_reduction,
            virtual_loss=cfg.virtual_loss,
            mcts_time_fraction=cfg.mcts_time_fraction,
            mate_plies=cfg.mate_plies,
            mate_topk=cfg.mate_topk,
            mate_nodes=cfg.mate_nodes,
            root_topn=cfg.root_topn,
        )

    def parse_go(self, line: str) -> Dict[str, object]:
        tokens = line.split()[1:]
        args: Dict[str, object] = {}
        searchmoves: List[str] = []
        i = 0
        go_keys = {
            "wtime", "btime", "winc", "binc", "movetime", "depth",
            "nodes", "mate", "infinite", "ponder", "searchmoves",
        }
        while i < len(tokens):
            key = tokens[i]
            if key == "searchmoves":
                i += 1
                while i < len(tokens) and tokens[i] not in go_keys:
                    searchmoves.append(tokens[i])
                    i += 1
                continue
            if key in ("infinite", "ponder"):
                args[key] = True
                i += 1
                continue
            if key in go_keys and i + 1 < len(tokens):
                args[key] = tokens[i + 1]
                i += 2
                continue
            i += 1
        if searchmoves:
            args["searchmoves"] = searchmoves
        return args

    def choose_movetime_ms(self, go_args: Dict[str, object]) -> int:
        cfg = self.config
        if "movetime" in go_args:
            return max(0, as_int(go_args["movetime"], cfg.movetime_ms))

        time_key = "wtime" if self.board.turn == chess.WHITE else "btime"
        inc_key = "winc" if self.board.turn == chess.WHITE else "binc"
        if time_key in go_args:
            remaining = max(0, as_int(go_args[time_key], 0))
            increment = max(0, as_int(go_args.get(inc_key, 0), 0))
            budget = (
                remaining / max(1.0, float(cfg.time_divisor))
                + increment * max(0.0, float(cfg.increment_fraction))
                - max(0, int(cfg.move_overhead_ms))
            )
            if cfg.max_movetime_ms > 0:
                budget = min(budget, float(cfg.max_movetime_ms))
            if cfg.min_movetime_ms > 0:
                budget = max(budget, float(cfg.min_movetime_ms))
            if remaining > 0:
                budget = min(budget, max(1.0, float(remaining - cfg.move_overhead_ms)))
            return max(0, int(budget))

        return max(0, int(cfg.movetime_ms))

    def allowed_moves(self, go_args: Dict[str, object]) -> Optional[Set[chess.Move]]:
        texts = go_args.get("searchmoves")
        if not texts:
            return None
        allowed: Set[chess.Move] = set()
        for text in texts:
            try:
                move = chess.Move.from_uci(str(text))
            except ValueError:
                continue
            if move in self.board.legal_moves:
                allowed.add(move)
        return allowed or None

    def fallback_move(self, allowed: Optional[Set[chess.Move]] = None) -> str:
        legal = list(self.board.legal_moves)
        if allowed is not None:
            legal = [move for move in legal if move in allowed]
        if not legal:
            return "0000"
        return max(legal, key=lambda move: move.uci()).uci()

    def handle_go(self, line: str):
        if self.board.is_game_over(claim_draw=True):
            uci_print("bestmove 0000")
            return

        go_args = self.parse_go(line)
        movetime_ms = self.choose_movetime_ms(go_args)
        sims_override = None
        if "nodes" in go_args:
            sims_override = min(self.config.mcts_sims, max(0, as_int(go_args["nodes"], self.config.mcts_sims)))

        allowed = self.allowed_moves(go_args)
        try:
            model = self.ensure_model()
            if model is None and self.config.model_path.strip().lower() != "none":
                uci_print("info string model is not loaded; set ModelPath or pass --model")

            result = UnifiedSearch(
                model,
                self.search_options(movetime_ms, sims_override=sims_override),
                device=self.config.device,
            ).search(self.board)
            move = result.move

            if allowed is not None and move not in allowed:
                candidates = sorted(
                    allowed,
                    key=lambda candidate: (
                        float(result.policy[move_to_index(candidate)]),
                        candidate.uci(),
                    ),
                    reverse=True,
                )
                move = candidates[0] if candidates else None

            if self.config.log_search or self.debug:
                info = result.info
                uci_print(
                    "info string "
                    f"best={info.get('best_san')} "
                    f"move={move.uci() if move is not None else '0000'} "
                    f"value={info.get('value')} "
                    f"sims={info.get('sims_completed')}/{info.get('mcts_dynamic_target')}/{info.get('mcts_soft_cap')} "
                    f"elapsed_ms={info.get('elapsed_ms')}"
                )
                for row in info.get("root", [])[: max(1, int(self.config.root_topn))]:
                    uci_print(
                        "info string "
                        f"{row.get('san')} {row.get('move')} "
                        f"p={float(row.get('p', 0.0)):.5f} "
                        f"visits={int(row.get('visits', 0))} "
                        f"q={float(row.get('q', 0.0)):+.4f} "
                        f"mate={row.get('mate')}"
                    )

            uci_print(f"bestmove {move.uci() if move is not None else self.fallback_move(allowed)}")
        except Exception as exc:
            uci_print(f"info string search error: {exc}")
            uci_print(f"bestmove {self.fallback_move(allowed)}")

    def loop(self):
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            command = line.split(maxsplit=1)[0]

            if command == "uci":
                self.handle_uci()
            elif command == "debug":
                self.debug = line.lower().endswith(" on")
            elif command == "isready":
                try:
                    self.ensure_model()
                except Exception as exc:
                    uci_print(f"info string model load error: {exc}")
                uci_print("readyok")
            elif command == "setoption":
                self.parse_setoption(line)
            elif command == "ucinewgame":
                self.board = chess.Board()
            elif command == "position":
                self.set_position(line)
            elif command == "go":
                self.handle_go(line)
            elif command == "stop":
                uci_print(f"bestmove {self.fallback_move()}")
            elif command == "quit":
                break
            elif command in ("ponderhit", "register"):
                continue
            else:
                uci_print(f"info string unknown command: {line}")


def parse_args():
    parser = argparse.ArgumentParser(description="ChessAI UCI engine wrapper")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--search-type",
        choices=sorted(VALID_SEARCH_TYPES),
        default="only-mcts",
    )
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--mcts-min-sims", type=int, default=0)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=1000)
    parser.add_argument("--move-overhead-ms", type=int, default=50)
    parser.add_argument("--min-movetime-ms", type=int, default=50)
    parser.add_argument("--max-movetime-ms", type=int, default=10000)
    parser.add_argument("--time-divisor", type=float, default=30.0)
    parser.add_argument("--increment-fraction", type=float, default=0.75)
    parser.add_argument("--c-puct", type=float, default=CPUCT)
    parser.add_argument("--c-puct-base", type=float, default=19652.0)
    parser.add_argument("--c-puct-factor", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.15)
    parser.add_argument("--virtual-loss", type=float, default=0.0)
    parser.add_argument("--mcts-time-fraction", type=float, default=0.90)
    parser.add_argument("--mate-plies", type=int, default=0)
    parser.add_argument("--mate-topk", type=int, default=4)
    parser.add_argument("--mate-nodes", type=int, default=20000)
    parser.add_argument("--root-topn", type=int, default=5)
    parser.add_argument("--log-search", action="store_true", default=False)
    return parser.parse_args()


def config_from_args(args) -> EngineConfig:
    return EngineConfig(
        model_path=str(args.model),
        device=str(args.device),
        search_type=str(args.search_type),
        mcts_sims=int(args.mcts_sims),
        mcts_min_sims=int(args.mcts_min_sims),
        mcts_batch_size=int(args.mcts_batch_size),
        movetime_ms=int(args.movetime_ms),
        move_overhead_ms=int(args.move_overhead_ms),
        min_movetime_ms=int(args.min_movetime_ms),
        max_movetime_ms=int(args.max_movetime_ms),
        time_divisor=float(args.time_divisor),
        increment_fraction=float(args.increment_fraction),
        c_puct=float(args.c_puct),
        c_puct_base=float(args.c_puct_base),
        c_puct_factor=float(args.c_puct_factor),
        fpu_reduction=float(args.fpu_reduction),
        virtual_loss=float(args.virtual_loss),
        mcts_time_fraction=float(args.mcts_time_fraction),
        mate_plies=int(args.mate_plies),
        mate_topk=int(args.mate_topk),
        mate_nodes=int(args.mate_nodes),
        root_topn=int(args.root_topn),
        log_search=bool(args.log_search),
    )


def main():
    args = parse_args()
    UCIEngine(config_from_args(args)).loop()


if __name__ == "__main__":
    main()
