from __future__ import annotations

import argparse
import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from config import (
    CPUCT,
    DEFAULT_SIMS,
    DEVICE,
    MODEL_PATH,
    NUM_ACTIONS,
)
from evaluator import BatchedEvaluator
from model import load_model
from move_encoder import move_to_index, policy_to_legal_distribution


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def count_pieces(board: chess.Board) -> int:
    return len(board.piece_map())


def safe_san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


def model_device(model: torch.nn.Module):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def terminal_value_side_to_move(board: chess.Board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def normalize_policy(policy: np.ndarray, board: chess.Board) -> np.ndarray:
    output = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal = list(board.legal_moves)
    if not legal:
        return output

    total = 0.0
    for move in legal:
        index = move_to_index(move)
        value = max(0.0, float(policy[index])) if index < len(policy) else 0.0
        output[index] = value
        total += value

    if total <= 0:
        probability = 1.0 / len(legal)
        for move in legal:
            output[move_to_index(move)] = probability
    else:
        output /= total
    return output


@dataclass
class SearchOptions:
    search_type: str = "mcts-mate"
    # mcts_sims is a soft upper budget. Confident positions may stop earlier.
    mcts_sims: int = DEFAULT_SIMS
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    c_puct: float = CPUCT
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_reduction: float = 0.15
    time_limit: Optional[float] = None
    mcts_time_fraction: float = 0.90
    root_topn: int = 10
    virtual_loss: float = 0.0

    mate_plies: int = 3
    mate_topk: int = 4
    mate_nodes: int = 20000
    mate_hash_mb: int = 16


VALID_SEARCH_TYPES = {"closed", "only-mcts", "mcts-mate"}


def normalize_search_type(search_type: str) -> str:
    value = str(search_type).strip().lower()
    if value not in VALID_SEARCH_TYPES:
        raise ValueError(
            f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}, got {search_type!r}"
        )
    return value


class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_visits = 0
        self.children: Dict[chess.Move, "MCTSNode"] = {}

    @property
    def q(self):
        if self.visit_count <= 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self):
        return bool(self.children)


@dataclass
class SearchResult:
    move: chess.Move
    policy: np.ndarray
    value: float
    mcts_policy: np.ndarray
    info: Dict[str, Any]


def root_uncertainty(root: MCTSNode) -> float:
    children = list(root.children.values())
    if len(children) <= 1:
        return 0.0

    counts = np.asarray([child.visit_count for child in children], dtype=np.float64)
    if counts.sum() > 0:
        distribution = counts / counts.sum()
    else:
        priors = np.asarray([max(0.0, child.prior) for child in children], dtype=np.float64)
        distribution = priors / max(1e-12, priors.sum())

    entropy = -float(np.sum(distribution * np.log(np.maximum(distribution, 1e-12))))
    entropy /= max(1e-12, math.log(len(children)))

    ordered = sorted(children, key=lambda child: (child.visit_count, child.prior), reverse=True)
    first = float(ordered[0].visit_count)
    second = float(ordered[1].visit_count)
    visit_margin_uncertainty = 1.0 - abs(first - second) / max(1.0, first + second)

    q_first = -float(ordered[0].q)
    q_second = -float(ordered[1].q)
    q_gap_uncertainty = 1.0 - min(1.0, abs(q_first - q_second) / 0.50)

    uncertainty = (
        0.50 * entropy
        + 0.35 * visit_margin_uncertainty
        + 0.15 * q_gap_uncertainty
    )
    return float(max(0.0, min(1.0, uncertainty)))


class MCTS:
    """PUCT MCTS with batched neural evaluation and uncertainty-sized budget."""

    def __init__(self, model, options: SearchOptions, device=DEVICE):
        self.model = model
        self.options = options
        self.device = str(device)
        self.evaluator = BatchedEvaluator(
            model,
            device=self.device,
            batch_size=max(1, int(options.mcts_batch_size)),
        )
        self.expanded_nodes = 0
        self.nn_batches = 0

    def _scheduled_c_puct(self, parent: MCTSNode) -> float:
        visits = max(0.0, float(parent.visit_count + parent.virtual_visits))
        base = max(1.0, float(self.options.c_puct_base))
        growth = max(0.0, float(self.options.c_puct_factor)) * math.log(
            (visits + base + 1.0) / base
        )
        return max(0.0, float(self.options.c_puct) + growth)

    def _fpu_value(self, parent: MCTSNode) -> float:
        visited_policy_mass = sum(
            max(0.0, float(child.prior))
            for child in parent.children.values()
            if child.visit_count > 0
        )
        parent_q = float(parent.q) if parent.visit_count > 0 else 0.0
        reduction = max(0.0, float(self.options.fpu_reduction))
        return float(
            max(-1.0, min(1.0, parent_q - reduction * math.sqrt(visited_policy_mass)))
        )

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode):
        q_from_parent = -child.q if child.visit_count > 0 else self._fpu_value(parent)
        visits = child.visit_count + child.virtual_visits
        exploration = (
            self._scheduled_c_puct(parent)
            * child.prior
            * np.sqrt(parent.visit_count + parent.virtual_visits + 1.0)
            / (1.0 + visits)
        )
        penalty = float(self.options.virtual_loss) * child.virtual_visits
        return q_from_parent + exploration - penalty

    def _select_child(self, node: MCTSNode):
        return max(
            node.children.items(),
            key=lambda item: (
                self._ucb_score(node, item[1]),
                item[1].prior,
                item[0].uci(),
            ),
        )

    def _expand_from_policy(self, node: MCTSNode, board: chess.Board, policy):
        priors = policy_to_legal_distribution(policy, board, normalize=True)
        if not node.expanded():
            for move, prior in priors.items():
                node.children[move] = MCTSNode(prior=prior)
            self.expanded_nodes += 1

    @staticmethod
    def _backpropagate(path, value):
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += float(value)
            value = -value

    @staticmethod
    def _clear_virtual(path):
        for node in path[1:]:
            node.virtual_visits = max(0, node.virtual_visits - 1)

    def _select_leaf(self, root: MCTSNode, root_board: chess.Board):
        node = root
        board = root_board.copy(stack=False)
        path = [root]

        while node.expanded():
            move, node = self._select_child(node)
            node.virtual_visits += 1
            board.push(move)
            path.append(node)

            terminal = terminal_value_side_to_move(board)
            if terminal is not None:
                return node, board, path, terminal

        return node, board, path, None

    def run(self, board: chess.Board, deadline=None):
        root = MCTSNode(0.0)
        terminal = terminal_value_side_to_move(board)
        if terminal is not None:
            return np.zeros(NUM_ACTIONS, dtype=np.float32), terminal, {
                "sims_completed": 0,
                "dynamic_target": 0,
                "uncertainty": 0.0,
                "expanded_nodes": 0,
                "nn_batches": 0,
                "timeout": False,
                "root_node": root,
                "root_c_puct": self._scheduled_c_puct(root),
                "root_fpu": self._fpu_value(root),
            }

        root_policy, root_value = self.evaluator.evaluate_one(board)
        self.nn_batches += 1
        self._expand_from_policy(root, board, root_policy)

        soft_cap = max(0, int(self.options.mcts_sims))
        batch_size = max(1, int(self.options.mcts_batch_size))
        if soft_cap <= 0:
            policy = normalize_policy(root_policy, board)
            return policy, float(root_value), {
                "sims_completed": 0,
                "dynamic_target": 0,
                "uncertainty": root_uncertainty(root),
                "expanded_nodes": self.expanded_nodes,
                "nn_batches": self.nn_batches,
                "timeout": False,
                "root_node": root,
                "root_c_puct": self._scheduled_c_puct(root),
                "root_fpu": self._fpu_value(root),
            }

        configured_minimum = int(self.options.mcts_min_sims)
        minimum = configured_minimum if configured_minimum > 0 else max(batch_size, soft_cap // 4)
        minimum = max(1, min(soft_cap, minimum))
        dynamic_target = minimum
        sims_completed = 0
        uncertainty = 1.0

        while sims_completed < soft_cap and sims_completed < dynamic_target:
            if deadline is not None and time.monotonic() >= deadline:
                break

            wanted = min(
                batch_size,
                soft_cap - sims_completed,
                max(1, dynamic_target - sims_completed),
            )
            selected = []
            selected_nodes = set()
            attempts = 0
            max_attempts = max(wanted * 5, wanted + 8)

            while (
                len(selected) < wanted
                and attempts < max_attempts
                and sims_completed + len(selected) < soft_cap
                and sims_completed + len(selected) < dynamic_target
            ):
                attempts += 1
                if deadline is not None and time.monotonic() >= deadline:
                    break

                leaf, leaf_board, path, terminal = self._select_leaf(root, board)
                if terminal is not None:
                    self._clear_virtual(path)
                    self._backpropagate(path, terminal)
                    sims_completed += 1
                    if sims_completed >= dynamic_target or sims_completed >= soft_cap:
                        break
                    continue

                leaf_identity = id(leaf)
                if leaf_identity in selected_nodes:
                    self._clear_virtual(path)
                    continue
                selected_nodes.add(leaf_identity)
                selected.append((leaf, leaf_board, path))

            if selected:
                policies, values = self.evaluator.evaluate_boards(
                    [item[1] for item in selected]
                )
                self.nn_batches += 1
                for (leaf, leaf_board, path), policy, value in zip(
                    selected, policies, values
                ):
                    self._expand_from_policy(leaf, leaf_board, policy)
                    self._clear_virtual(path)
                    self._backpropagate(path, float(value))
                    sims_completed += 1

            if sims_completed >= minimum:
                uncertainty = root_uncertainty(root)
                desired = minimum + int(
                    math.ceil(uncertainty * max(0, soft_cap - minimum))
                )
                dynamic_target = max(minimum, min(soft_cap, desired))

            if not selected and attempts >= max_attempts:
                break

        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for move, child in root.children.items():
            # Preserve the network prior as the root tie-breaker. With small
            # search budgets many legal moves can finish with identical visit
            # counts; using visits alone then falls through to UCI ordering.
            policy[move_to_index(move)] = (
                float(child.visit_count) + max(0.0, float(child.prior))
            )
        if float(policy.sum()) <= 0:
            for move, child in root.children.items():
                policy[move_to_index(move)] = float(child.prior)
        policy = normalize_policy(policy, board)

        return policy, float(root.q if root.visit_count else root_value), {
            "sims_completed": int(sims_completed),
            "dynamic_target": int(dynamic_target),
            "uncertainty": float(root_uncertainty(root)),
            "expanded_nodes": int(self.expanded_nodes),
            "nn_batches": int(self.nn_batches),
            "timeout": bool(deadline is not None and time.monotonic() >= deadline),
            "root_node": root,
            "root_c_puct": float(self._scheduled_c_puct(root)),
            "root_fpu": float(self._fpu_value(root)),
        }


class MateLimit(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class MateFinder:
    """Bounded proof search for forcing mates by the side to move.

    The finder is intentionally one-way: it may override the root move when it
    proves a mate for the mover inside the configured attacker-move budget. It
    does not ban ordinary moves or try to defend against opponent threats.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module],
        options: SearchOptions,
        deadline: Optional[float],
        device: str,
        root_policy: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.options = options
        self.deadline = deadline
        self.device = str(device)
        self.root_policy = root_policy
        self.nodes = 0
        self.evaluator = None
        self.policy_cache: Dict[Any, Tuple[np.ndarray, float]] = {}
        self.proof_cache: Dict[Tuple[Any, chess.Color, int], Tuple[bool, List[chess.Move]]] = {}
        self.max_proof_cache_entries = max(
            0,
            int(max(0, int(options.mate_hash_mb)) * 1024 * 1024 / 256),
        )
        if self.model is not None:
            self.evaluator = BatchedEvaluator(
                self.model,
                device=self.device,
                batch_size=max(1, int(options.mcts_batch_size)),
            )

    def _check_limit(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise MateLimit("time_limit")
        if self.options.mate_nodes > 0 and self.nodes >= int(self.options.mate_nodes):
            raise MateLimit("node_limit")

    def _cache_proof(
        self,
        key: Tuple[Any, chess.Color, int],
        result: Tuple[bool, List[chess.Move]],
    ):
        if self.max_proof_cache_entries <= 0:
            return
        if len(self.proof_cache) >= self.max_proof_cache_entries:
            return
        self.proof_cache[key] = result

    @staticmethod
    def _board_cache_key(board: chess.Board):
        try:
            return board._transposition_key()
        except Exception:
            return board.fen()

    @staticmethod
    def _move_order_score(board: chess.Board, move: chess.Move, prior=0.0):
        score = float(prior) * 1000.0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim:
                score += 100.0 * PIECE_VALUES.get(victim.piece_type, 0)
            if attacker:
                score -= 10.0 * PIECE_VALUES.get(attacker.piece_type, 0)
        if move.promotion:
            score += 100.0 * PIECE_VALUES.get(move.promotion, 0)
        try:
            if board.gives_check(move):
                score += 80.0
                board.push(move)
                if board.is_checkmate():
                    score += 10000.0
                board.pop()
        except Exception:
            pass
        return score

    def _policy_for(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        if self.root_policy is not None and not board.move_stack:
            return normalize_policy(self.root_policy, board), 0.0
        key = self._board_cache_key(board)
        if key in self.policy_cache:
            return self.policy_cache[key]
        if self.evaluator is None:
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            legal = list(board.legal_moves)
            if legal:
                probability = 1.0 / len(legal)
                for move in legal:
                    policy[move_to_index(move)] = probability
            value = 0.0
        else:
            raw_policy, value = self.evaluator.evaluate_one(board)
            policy = normalize_policy(raw_policy, board)
        payload = (policy, float(value))
        self.policy_cache[key] = payload
        return payload

    def _ordered_moves(
        self,
        board: chess.Board,
        *,
        topk: Optional[int] = None,
        root_policy: Optional[np.ndarray] = None,
    ):
        policy = root_policy
        if policy is None:
            policy, _value = self._policy_for(board)
        moves = list(board.legal_moves)
        moves.sort(
            key=lambda move: (
                self._move_order_score(board, move, policy[move_to_index(move)]),
                move.uci(),
            ),
            reverse=True,
        )
        if topk is None or int(topk) <= 0:
            return moves
        return moves[: min(int(topk), len(moves))]

    def _ordered_defenses(self, board: chess.Board):
        policy, _value = self._policy_for(board)
        king_square = board.king(board.turn)

        def score(move: chess.Move):
            piece = board.piece_at(move.from_square)
            value = self._move_order_score(board, move, policy[move_to_index(move)])
            if piece and piece.piece_type == chess.KING:
                value += 250.0
            if board.is_capture(move):
                value += 180.0
            if king_square is not None and board.is_check():
                value += 120.0
            try:
                if board.gives_check(move):
                    value += 60.0
            except Exception:
                pass
            return value

        moves = list(board.legal_moves)
        moves.sort(key=lambda move: (score(move), move.uci()), reverse=True)
        return moves

    def _side_can_force_checkmate(
        self,
        board: chess.Board,
        attacker: chess.Color,
        attacker_moves_left: int,
    ) -> Tuple[bool, List[chess.Move]]:
        self._check_limit()
        self.nodes += 1

        if board.is_checkmate():
            return board.turn != attacker, []
        if board.is_game_over(claim_draw=True):
            return False, []

        cache_key = (self._board_cache_key(board), attacker, int(attacker_moves_left))
        cached = self.proof_cache.get(cache_key)
        if cached is not None:
            return cached

        if board.turn == attacker:
            if attacker_moves_left <= 0:
                self._cache_proof(cache_key, (False, []))
                return False, []
            for move in self._ordered_moves(
                board,
                topk=max(1, int(self.options.mate_topk)),
            ):
                board.push(move)
                forced, child_pv = self._side_can_force_checkmate(
                    board,
                    attacker,
                    attacker_moves_left - 1,
                )
                board.pop()
                if forced:
                    result = (True, [move] + child_pv)
                    self._cache_proof(cache_key, result)
                    return result
            self._cache_proof(cache_key, (False, []))
            return False, []

        replies = self._ordered_defenses(board)
        if not replies:
            self._cache_proof(cache_key, (False, []))
            return False, []
        representative_pv: List[chess.Move] = []
        for reply in replies:
            board.push(reply)
            still_forced, child_pv = self._side_can_force_checkmate(
                board,
                attacker,
                attacker_moves_left,
            )
            board.pop()
            if not still_forced:
                self._cache_proof(cache_key, (False, []))
                return False, []
            if not representative_pv:
                representative_pv = [reply] + child_pv
        result = (True, representative_pv)
        self._cache_proof(cache_key, result)
        return result

    def _find_forcing_mate_move(
        self,
        board: chess.Board,
        attacker_moves: int,
        root_candidates: List[chess.Move],
    ):
        attacker = board.turn
        for move in root_candidates:
            self._check_limit()
            self.nodes += 1
            if move not in board.legal_moves:
                continue

            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move, "mate_in_1", [move]
            forced, child_pv = self._side_can_force_checkmate(
                board,
                attacker,
                attacker_moves - 1,
            )
            board.pop()
            if forced:
                return move, f"forces_mate_within_{attacker_moves}_moves", [move] + child_pv
        return None, None, []

    def analyze(self, board: chess.Board, candidates: List[chess.Move]):
        forced_move = None
        pv: List[chess.Move] = []
        reasons: Dict[str, str] = {}
        completed = True
        status = "not_found"
        mate_plies = max(0, int(self.options.mate_plies))

        if mate_plies <= 0:
            return {
                "forced_move": None,
                "pv": [],
                "reasons": {},
                "nodes": 0,
                "completed": True,
                "status": "disabled",
                "cache_entries": 0,
            }

        try:
            forced_move, reason, pv = self._find_forcing_mate_move(
                board,
                mate_plies,
                candidates,
            )
            if forced_move is not None and reason is not None:
                reasons[forced_move.uci()] = reason
                status = "proved"
        except MateLimit as exc:
            completed = False
            status = exc.reason

        return {
            "forced_move": forced_move,
            "pv": pv,
            "reasons": reasons,
            "nodes": int(self.nodes),
            "completed": bool(completed),
            "status": status,
            "cache_entries": int(len(self.proof_cache)),
        }


class UnifiedSearch:
    """Neural policy, MCTS, and optional active mate search."""

    def __init__(self, model: Optional[torch.nn.Module], options=None, device=None):
        self.model = model
        self.options = options or SearchOptions()
        if device is None and self.model is not None:
            device = str(model_device(self.model))
        elif device is None:
            device = DEVICE
        self.device = str(device)
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _policy_only(self, board: chess.Board):
        legal = list(board.legal_moves)
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        if not legal:
            return policy, 0.0
        if self.model is None:
            probability = 1.0 / len(legal)
            for move in legal:
                policy[move_to_index(move)] = probability
            return policy, 0.0

        evaluator = BatchedEvaluator(
            self.model,
            device=str(model_device(self.model)),
            batch_size=max(1, int(self.options.mcts_batch_size)),
        )
        raw_policy, value = evaluator.evaluate_one(board)
        return normalize_policy(raw_policy, board), float(value)

    @staticmethod
    def _select_top_move(board, policy):
        legal = list(board.legal_moves)
        if not legal:
            return None
        return max(
            legal,
            key=lambda move: (
                max(0.0, float(policy[move_to_index(move)])),
                move.uci(),
            ),
        )

    def search(self, board: chess.Board) -> SearchResult:
        if board.is_game_over(claim_draw=True):
            raise RuntimeError("game is already over")

        search_type = normalize_search_type(self.options.search_type)
        root_board = board.copy(stack=False)
        start = time.monotonic()
        final_deadline = None
        mcts_deadline = None
        if self.options.time_limit is not None and float(self.options.time_limit) > 0:
            total = float(self.options.time_limit)
            final_deadline = start + total
            mcts_fraction = max(
                0.0,
                min(1.0, float(self.options.mcts_time_fraction)),
            )
            if search_type != "mcts-mate" or int(self.options.mate_plies) <= 0:
                mcts_fraction = 1.0
            mcts_deadline = start + total * mcts_fraction

        root_node = None
        if (
            search_type == "closed"
            or self.model is None
            or int(self.options.mcts_sims) <= 0
        ):
            mcts_policy, value = self._policy_only(root_board)
            stats = {
                "sims_completed": 0,
                "dynamic_target": 0,
                "uncertainty": 0.0,
                "expanded_nodes": 0,
                "nn_batches": 1 if self.model is not None else 0,
                "timeout": False,
                "root_node": None,
            }
        else:
            mcts = MCTS(
                self.model,
                self.options,
                device=str(model_device(self.model)),
            )
            mcts_policy, value, stats = mcts.run(
                root_board,
                deadline=mcts_deadline,
            )
            root_node = stats.get("root_node")

        legal = list(root_board.legal_moves)
        mcts_move = self._select_top_move(root_board, mcts_policy)
        final_policy = mcts_policy.copy()
        mate_forced = None
        mate_pv: List[chess.Move] = []
        mate_reasons: Dict[str, str] = {}
        mate_nodes = 0
        mate_completed = True
        mate_status = "disabled" if search_type != "mcts-mate" else "not_found"
        mate_cache_entries = 0

        if legal and search_type == "mcts-mate" and int(self.options.mate_plies) > 0:
            mate_topk = max(1, int(self.options.mate_topk))
            ranked = sorted(
                legal,
                key=lambda move: (
                    float(mcts_policy[move_to_index(move)]),
                    root_node.children.get(move).visit_count
                    if root_node is not None and move in root_node.children
                    else 0,
                    root_node.children.get(move).prior
                    if root_node is not None and move in root_node.children
                    else 0.0,
                    move.uci(),
                ),
                reverse=True,
            )
            candidates = ranked[: min(mate_topk, len(ranked))]
            if mcts_move is not None and mcts_move not in candidates:
                candidates[-1] = mcts_move

            mate = MateFinder(
                self.model,
                self.options,
                deadline=final_deadline,
                device=str(model_device(self.model)) if self.model is not None else self.device,
                root_policy=mcts_policy,
            )
            mate_info = mate.analyze(root_board, candidates)
            mate_forced = mate_info["forced_move"]
            mate_pv = list(mate_info.get("pv") or [])
            mate_reasons = dict(mate_info["reasons"])
            mate_nodes = int(mate_info["nodes"])
            mate_completed = bool(mate_info["completed"])
            mate_status = str(mate_info.get("status") or "not_found")
            mate_cache_entries = int(mate_info.get("cache_entries", 0))

            if mate_forced is not None and mate_forced in legal:
                final_policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                final_policy[move_to_index(mate_forced)] = 1.0

        move = self._select_top_move(root_board, final_policy)
        if move is None or move not in legal:
            if not legal:
                raise RuntimeError("no legal moves")
            move = max(
                legal,
                key=lambda candidate: (
                    float(final_policy[move_to_index(candidate)]),
                    candidate.uci(),
                ),
            )

        root_lines = []
        for candidate in legal:
            index = move_to_index(candidate)
            child = (
                root_node.children.get(candidate)
                if root_node is not None
                else None
            )
            root_lines.append({
                "move": candidate.uci(),
                "san": safe_san(root_board, candidate),
                "selected": candidate == move,
                "p": float(final_policy[index]),
                "mcts_p": float(mcts_policy[index]),
                "visits": int(child.visit_count) if child is not None else 0,
                "prior": float(child.prior) if child is not None else float(mcts_policy[index]),
                "q": float(-child.q) if child is not None else 0.0,
                "mate": mate_reasons.get(candidate.uci()),
            })
        root_lines.sort(
            key=lambda row: (
                row["selected"],
                row["p"],
                row["visits"],
                row["q"],
                row["prior"],
            ),
            reverse=True,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0
        effective_mcts_sims = 0 if search_type == "closed" else int(self.options.mcts_sims)
        effective_mate_plies = (
            int(self.options.mate_plies) if search_type == "mcts-mate" else 0
        )
        effective_mate_topk = (
            int(self.options.mate_topk) if search_type == "mcts-mate" else 0
        )
        effective_mate_hash_mb = (
            int(self.options.mate_hash_mb) if search_type == "mcts-mate" else 0
        )
        info = {
            "search_type": search_type,
            "value": float(value),
            "mcts_soft_cap": int(effective_mcts_sims),
            "mcts_dynamic_target": int(stats.get("dynamic_target", 0)),
            "sims_completed": int(stats.get("sims_completed", 0)),
            "uncertainty": float(stats.get("uncertainty", 0.0)),
            "nodes": int(stats.get("expanded_nodes", 0)),
            "nn_batches": int(stats.get("nn_batches", 0)),
            "c_puct_initial": float(self.options.c_puct),
            "c_puct_base": float(self.options.c_puct_base),
            "c_puct_factor": float(self.options.c_puct_factor),
            "c_puct_root": float(stats.get("root_c_puct", self.options.c_puct)),
            "fpu_reduction": float(self.options.fpu_reduction),
            "fpu_root": float(stats.get("root_fpu", 0.0)),
            "virtual_loss": float(self.options.virtual_loss),
            "mcts_time_fraction": float(self.options.mcts_time_fraction),
            "mate_plies": int(effective_mate_plies),
            "mate_topk": int(effective_mate_topk),
            "mate_nodes": int(mate_nodes),
            "mate_hash_mb": int(effective_mate_hash_mb),
            "mate_completed": bool(mate_completed),
            "mate_status": str(mate_status),
            "mate_forced_move": mate_forced.uci() if mate_forced is not None else None,
            "mate_pv": [move.uci() for move in mate_pv],
            "mate_cache_entries": int(mate_cache_entries),
            "mate_reasons": dict(mate_reasons),
            "mcts_move": mcts_move.uci() if mcts_move else None,
            "piece_count": count_pieces(root_board),
            "best_move": move.uci(),
            "best_san": safe_san(root_board, move),
            "timeout": bool(
                final_deadline is not None
                and time.monotonic() >= final_deadline
            ),
            "elapsed_ms": round(elapsed_ms, 2),
            "selection": {
                "selection_mode": "top1",
                "selected_move": move.uci(),
            },
            "root": root_lines[: max(1, int(self.options.root_topn))],
        }

        return SearchResult(
            move=move,
            policy=final_policy.astype(np.float32),
            value=float(value),
            mcts_policy=mcts_policy.astype(np.float32),
            info=info,
        )


def select_move(board, model, options=None, device=None) -> Tuple[chess.Move, Dict]:
    result = UnifiedSearch(model, options or SearchOptions(), device=device).search(board)
    return result.move, result.info


def get_suggestions(board, model, options=None, topn=5, device=None) -> List[Dict]:
    options = options or SearchOptions(root_topn=topn)
    options = dataclasses.replace(options, root_topn=topn)
    result = UnifiedSearch(model, options, device=device).search(board)
    return result.info.get("root", [])[:topn]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChessAI policy/MCTS search with optional active mate search"
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--search-type",
        choices=sorted(VALID_SEARCH_TYPES),
        default="mcts-mate",
    )
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--mcts-min-sims", type=int, default=0)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=5000)
    parser.add_argument("--c-puct", type=float, default=CPUCT)
    parser.add_argument("--c-puct-base", type=float, default=19652.0)
    parser.add_argument("--c-puct-factor", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.15)
    parser.add_argument("--virtual-loss", type=float, default=0.0)
    parser.add_argument("--mcts-time-fraction", type=float, default=0.90)
    parser.add_argument("--mate-plies", type=int, default=3)
    parser.add_argument("--mate-topk", type=int, default=4)
    parser.add_argument("--mate-nodes", type=int, default=20000)
    parser.add_argument("--mate-hash-mb", type=int, default=16)
    parser.add_argument("--root-topn", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    board = chess.Board() if args.fen == "startpos" else chess.Board(args.fen)
    model = None if str(args.model).lower() == "none" else load_model(
        args.model,
        device=args.device,
    )
    options = SearchOptions(
        search_type=args.search_type,
        mcts_sims=args.mcts_sims,
        mcts_min_sims=args.mcts_min_sims,
        mcts_batch_size=args.mcts_batch_size,
        time_limit=(args.movetime_ms / 1000.0) if args.movetime_ms > 0 else None,
        c_puct=args.c_puct,
        c_puct_base=args.c_puct_base,
        c_puct_factor=args.c_puct_factor,
        fpu_reduction=args.fpu_reduction,
        virtual_loss=args.virtual_loss,
        mcts_time_fraction=args.mcts_time_fraction,
        mate_plies=args.mate_plies,
        mate_topk=args.mate_topk,
        mate_nodes=args.mate_nodes,
        mate_hash_mb=args.mate_hash_mb,
        root_topn=args.root_topn,
    )
    result = UnifiedSearch(model, options, device=args.device).search(board)
    print("fen:", board.fen())
    print("best:", result.info["best_san"], result.move.uci())
    print("value:", result.value)
    print(
        "mcts:",
        result.info["sims_completed"],
        "/",
        result.info["mcts_dynamic_target"],
        "/",
        result.info["mcts_soft_cap"],
    )
    print("uncertainty:", result.info["uncertainty"])
    print("c_puct_root:", result.info["c_puct_root"])
    print("fpu_root:", result.info["fpu_root"])
    print("virtual_loss:", result.info["virtual_loss"])
    print("mate_nodes:", result.info["mate_nodes"])
    print("mate_hash_mb:", result.info["mate_hash_mb"])
    print("mate_completed:", result.info["mate_completed"])
    print("mate_status:", result.info["mate_status"])
    print("mate_forced_move:", result.info["mate_forced_move"])
    print("mate_pv:", " ".join(result.info.get("mate_pv") or []))
    print("mate_cache_entries:", result.info["mate_cache_entries"])
    print("elapsed_ms:", result.info["elapsed_ms"])
    print("root:")
    for index, row in enumerate(result.info.get("root", []), 1):
        marker = "*" if row.get("selected") else " "
        print(
            f"{marker}{index:2d}. {row['san']:8s} {row['move']:5s} "
            f"p={row['p']:.5f} mcts={row['mcts_p']:.5f} "
            f"visits={row['visits']:4d} q={row['q']:+.4f} "
            f"mate={row['mate']}"
        )


if __name__ == "__main__":
    main()
