"""Stockfish teacher used only by self-learning and model-quality validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import chess.engine
import numpy as np

from config import NUM_ACTIONS, STOCKFISH_PATH
from move_encoder import move_to_index

MATE_SCORE_CP = 100000
CACHE_VERSION = "teacher-v2"


@dataclass
class TeacherConfig:
    uci: str = STOCKFISH_PATH
    depth: int = 10
    movetime_ms: int = 0
    multipv: int = 8
    threads: int = 4
    hash_mb: int = 512
    policy_temperature_cp: float = 80.0
    cache_path: Optional[str] = "data/selflearn/teacher_cache.sqlite"


class TeacherCache:
    def __init__(self, path: Optional[str]):
        self.conn = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path, timeout=60)
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS teacher_cache "
                "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self.conn.commit()

    def get(self, key):
        if self.conn is None:
            return None
        row = self.conn.execute(
            "SELECT payload FROM teacher_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, payload):
        if self.conn is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO teacher_cache(cache_key, payload) VALUES (?, ?)",
            (key, json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        )
        self.conn.commit()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class StockfishTeacher:
    def __init__(self, config: TeacherConfig):
        self.config = config
        if not os.path.exists(config.uci):
            raise FileNotFoundError(f"UCI engine not found: {config.uci}")
        self.engine = chess.engine.SimpleEngine.popen_uci(config.uci)
        options = {}
        if config.threads > 0:
            options["Threads"] = int(config.threads)
        if config.hash_mb > 0:
            options["Hash"] = int(config.hash_mb)
        if options:
            try:
                self.engine.configure(options)
            except Exception:
                pass
        self.cache = TeacherCache(config.cache_path)

    def close(self):
        try:
            self.engine.quit()
        finally:
            self.cache.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _limit(self):
        kwargs = {}
        if self.config.depth > 0:
            kwargs["depth"] = int(self.config.depth)
        if self.config.movetime_ms > 0:
            kwargs["time"] = float(self.config.movetime_ms) / 1000.0
        if not kwargs:
            kwargs["depth"] = 10
        return chess.engine.Limit(**kwargs)

    @staticmethod
    def _score_cp(info, pov):
        score = info.get("score")
        if score is None:
            return None
        try:
            return int(score.pov(pov).score(mate_score=MATE_SCORE_CP))
        except Exception:
            return None

    @staticmethod
    def _value(info, pov, cp):
        score = info.get("score")
        if score is not None:
            try:
                expectation = float(score.pov(pov).wdl(model="sf16").expectation())
                return max(-1.0, min(1.0, 2.0 * expectation - 1.0))
            except Exception:
                pass
        if cp is None:
            return 0.0
        return float(np.tanh(float(cp) / 600.0))

    def _score_child_move(self, board, move, pov):
        child = board.copy(stack=False)
        child.push(move)
        info = self.engine.analyse(child, self._limit())
        return self._score_cp(info, pov)

    def _policy_from_move_scores(self, move_scores):
        rows = [
            {"move": move, "score_cp": int(score)}
            for move, score in move_scores.items()
        ]
        rows.sort(key=lambda row: (-row["score_cp"], row["move"]))
        if not rows:
            return {}

        scores = np.asarray([row["score_cp"] for row in rows], dtype=np.float64)
        temperature = max(1e-6, float(self.config.policy_temperature_cp))
        logits = (scores - scores.max()) / temperature
        probabilities = np.exp(logits)
        probabilities /= max(1e-12, probabilities.sum())
        return {
            row["move"]: float(probability)
            for row, probability in zip(rows, probabilities)
        }

    def _cache_key(self, board, played_move):
        raw = "|".join([
            CACHE_VERSION,
            board.fen(),
            played_move.uci() if played_move else "-",
            str(self.config.depth),
            str(self.config.movetime_ms),
            str(self.config.multipv),
            str(self.config.policy_temperature_cp),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def analyse(self, board: chess.Board, played_move: Optional[chess.Move] = None) -> Dict:
        if played_move is not None and played_move not in board.legal_moves:
            raise ValueError(f"played move is illegal: {played_move.uci()}")

        cache_key = self._cache_key(board, played_move)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        mover = board.turn
        legal_count = board.legal_moves.count()
        multipv = max(1, min(int(self.config.multipv), legal_count))
        infos = self.engine.analyse(board, self._limit(), multipv=multipv)
        if isinstance(infos, dict):
            infos = [infos]

        rows = []
        info_by_move = {}
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            if move not in board.legal_moves:
                continue
            cp = self._score_cp(info, mover)
            if cp is None:
                continue
            row = {"move": move.uci(), "score_cp": int(cp)}
            rows.append(row)
            info_by_move[move.uci()] = info

        if not rows:
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            probability = 1.0 / max(1, len(legal))
            payload = {
                "policy_moves": {move.uci(): probability for move in legal},
                "move_scores_cp": {move.uci(): 0 for move in legal},
                "value": 0.0,
                "best_move": legal[0].uci() if legal else None,
                "best_score_cp": 0,
                "played_score_cp": 0,
                "regret_cp": 0,
                "margin_cp": 0,
            }
            self.cache.put(cache_key, payload)
            return payload

        rows.sort(key=lambda row: row["score_cp"], reverse=True)
        move_scores = {row["move"]: int(row["score_cp"]) for row in rows}
        policy_moves = self._policy_from_move_scores(move_scores)

        best_move = rows[0]["move"]
        best_score = int(rows[0]["score_cp"])
        margin = best_score - int(rows[1]["score_cp"]) if len(rows) > 1 else 300
        best_info = info_by_move.get(best_move, infos[0])
        teacher_value = self._value(best_info, mover, best_score)

        played_score = best_score
        if played_move is not None:
            played_uci = played_move.uci()
            if played_uci in move_scores:
                played_score = int(move_scores[played_uci])
            else:
                child_score = self._score_child_move(board, played_move, mover)
                played_score = int(child_score) if child_score is not None else best_score

        payload = {
            "policy_moves": policy_moves,
            "move_scores_cp": move_scores,
            "value": float(teacher_value),
            "best_move": best_move,
            "best_score_cp": best_score,
            "played_score_cp": int(played_score),
            "regret_cp": int(max(0, best_score - int(played_score))),
            "margin_cp": int(max(0, margin)),
        }
        self.cache.put(cache_key, payload)
        return payload

    def analyse_candidates(
        self,
        board: chess.Board,
        candidate_moves: List[chess.Move],
        played_move: Optional[chess.Move] = None,
    ) -> Dict:
        result = dict(self.analyse(board, played_move=played_move))
        legal_moves = set(board.legal_moves)
        candidates = []
        seen = set()
        for move in candidate_moves:
            if move not in legal_moves:
                continue
            uci = move.uci()
            if uci in seen:
                continue
            seen.add(uci)
            candidates.append(move)

        if not candidates:
            result["teacher_label_moves"] = []
            result["teacher_label_topk"] = 0
            return result

        mover = board.turn
        move_scores = {
            str(move): int(score)
            for move, score in (result.get("move_scores_cp") or {}).items()
        }
        scored_candidates = []
        for move in candidates:
            uci = move.uci()
            if uci not in move_scores:
                child_score = self._score_child_move(board, move, mover)
                if child_score is not None:
                    move_scores[uci] = int(child_score)
            if uci in move_scores:
                scored_candidates.append(uci)

        if move_scores:
            result["move_scores_cp"] = move_scores
            result["policy_moves"] = self._policy_from_move_scores(move_scores)
            ordered = sorted(
                move_scores.items(),
                key=lambda item: (-int(item[1]), item[0]),
            )
            best_move, best_score = ordered[0]
            previous_best = result.get("best_move")
            result["best_move"] = best_move
            result["best_score_cp"] = int(best_score)
            if best_move != previous_best:
                result["value"] = self._value({}, mover, int(best_score))
            if len(ordered) > 1:
                result["margin_cp"] = int(
                    max(0, int(ordered[0][1]) - int(ordered[1][1]))
                )
            played_uci = played_move.uci() if played_move is not None else best_move
            if played_uci in move_scores:
                played_score = int(move_scores[played_uci])
                result["played_score_cp"] = played_score
                result["regret_cp"] = int(max(0, int(best_score) - played_score))

        result["teacher_label_moves"] = scored_candidates
        result["teacher_label_topk"] = len(scored_candidates)
        return result

    @staticmethod
    def dense_policy(board: chess.Board, result: Dict) -> np.ndarray:
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for uci, probability in result.get("policy_moves", {}).items():
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    policy[move_to_index(move)] = float(probability)
            except Exception:
                continue
        total = float(policy.sum())
        if total <= 0:
            legal = list(board.legal_moves)
            if legal:
                probability = 1.0 / len(legal)
                for move in legal:
                    policy[move_to_index(move)] = probability
        else:
            policy /= total
        return policy


def teacher_weight_from_result(result: Dict) -> float:
    regret = max(0.0, float(result.get("regret_cp", 0.0)))
    margin = max(0.0, float(result.get("margin_cp", 0.0)))

    if regret <= 30:
        base = 0.05
    elif regret <= 150:
        base = 0.05 + 0.45 * (regret - 30.0) / 120.0
    elif regret <= 300:
        base = 0.50 + 0.35 * (regret - 150.0) / 150.0
    else:
        base = 0.95

    confidence = min(1.0, max(0.25, margin / 120.0))
    return float(min(0.95, max(0.0, base * confidence)))


def acceptable_moves(
    result: Dict,
    tolerance_cp: int = 35,
    max_answers: int = 8,
) -> List[str]:
    scores = result.get("move_scores_cp") or {}
    if not scores:
        best = result.get("best_move")
        return [best] if best else []

    best_score = max(int(score) for score in scores.values())
    answers = [
        move
        for move, score in scores.items()
        if best_score - int(score) <= max(0, int(tolerance_cp))
    ]
    answers.sort(key=lambda move: (-int(scores[move]), move))
    if not answers and result.get("best_move"):
        answers = [result["best_move"]]
    return answers[: max(1, int(max_answers))]


def move_accuracy_from_regret(regret_cp: float) -> float:
    regret = max(0.0, float(regret_cp))
    return float(max(0.0, min(100.0, 100.0 * np.exp(-regret / 350.0))))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a position with the Stockfish teacher")
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--played-move", default=None)
    parser.add_argument("--uci", default=STOCKFISH_PATH)
    parser.add_argument("--uci-depth", type=int, default=10)
    parser.add_argument("--uci-movetime-ms", type=int, default=0)
    parser.add_argument("--uci-multipv", type=int, default=8)
    parser.add_argument("--uci-threads", type=int, default=4)
    parser.add_argument("--uci-hash-mb", type=int, default=512)
    parser.add_argument("--teacher-policy-temperature-cp", type=float, default=80.0)
    parser.add_argument("--teacher-cache", default="data/selflearn/teacher_cache.sqlite")
    parser.add_argument("--answer-tolerance-cp", type=int, default=35)
    return parser.parse_args()


def main():
    args = parse_args()
    board = chess.Board() if args.fen == "startpos" else chess.Board(args.fen)
    played_move = chess.Move.from_uci(args.played_move) if args.played_move else None
    config = TeacherConfig(
        uci=args.uci,
        depth=args.uci_depth,
        movetime_ms=args.uci_movetime_ms,
        multipv=args.uci_multipv,
        threads=args.uci_threads,
        hash_mb=args.uci_hash_mb,
        policy_temperature_cp=args.teacher_policy_temperature_cp,
        cache_path=args.teacher_cache,
    )
    with StockfishTeacher(config) as teacher:
        result = teacher.analyse(board, played_move=played_move)
    result["teacher_weight"] = teacher_weight_from_result(result)
    result["acceptable_moves"] = acceptable_moves(
        result,
        tolerance_cp=args.answer_tolerance_cp,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
