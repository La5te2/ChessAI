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
    # mcts_sims is a soft upper budget. Confident positions may stop earlier.
    mcts_sims: int = DEFAULT_SIMS
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    c_puct: float = CPUCT
    time_limit: Optional[float] = None
    root_topn: int = 10
    virtual_loss: float = 1.0

    # When enabled, Alpha-Beta verifies MCTS root candidates directly; there is
    # no position-stage classifier.
    alpha_beta_depth: int = 4
    alpha_beta_topk: int = 4
    alpha_beta_nodes: int = 20000
    alpha_beta_quiescence: int = 3
    alpha_beta_margin: float = 0.10
    alpha_beta_time_fraction: float = 0.25
    alpha_beta_policy_weight: float = 0.35
    mate_guard_plies: int = 3

    q_tiebreak: bool = True
    q_tiebreak_min_visits: int = 32
    q_tiebreak_p_ratio: float = 0.90
    q_tiebreak_visit_ratio: float = 0.80
    q_tiebreak_margin: float = 0.25


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

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode):
        q_from_parent = -child.q
        visits = child.visit_count + child.virtual_visits
        exploration = (
            float(self.options.c_puct)
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
            policy[move_to_index(move)] = float(child.visit_count)
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
        }


class VerificationLimit(Exception):
    pass


class AlphaBetaVerifier:
    """Universal root-candidate verifier with neural leaves and bounded search."""

    def __init__(
        self,
        evaluator: BatchedEvaluator,
        options: SearchOptions,
        deadline: Optional[float],
    ):
        self.evaluator = evaluator
        self.options = options
        self.deadline = deadline
        self.nodes = 0
        self.leaf_cache: Dict[str, float] = {}

    def _check_limit(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise VerificationLimit()
        if self.options.alpha_beta_nodes > 0 and self.nodes >= int(
            self.options.alpha_beta_nodes
        ):
            raise VerificationLimit()

    def _leaf_value(self, board: chess.Board) -> float:
        key = board.fen()
        if key not in self.leaf_cache:
            _policy, value = self.evaluator.evaluate_one(board)
            self.leaf_cache[key] = float(value)
        return self.leaf_cache[key]

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

    def _side_can_force_checkmate(self, board: chess.Board, plies: int) -> bool:
        self._check_limit()
        self.nodes += 1

        terminal = terminal_value_side_to_move(board)
        if terminal is not None:
            return float(terminal) > 0.0
        if plies <= 0:
            return False

        checking_moves = [
            move
            for move in self._ordered_moves(board)
            if board.gives_check(move)
        ]
        for move in checking_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return True

            legal_replies = list(self._ordered_moves(board))
            forced = bool(legal_replies)
            for reply in legal_replies:
                board.push(reply)
                reply_allows_mate = self._side_can_force_checkmate(
                    board,
                    plies - 2,
                )
                board.pop()
                if not reply_allows_mate:
                    forced = False
                    break

            board.pop()
            if forced:
                return True
        return False

    def _root_tactical_score(self, child: chess.Board):
        terminal = terminal_value_side_to_move(child)
        if terminal is not None:
            return -float(terminal)

        guard_plies = max(0, int(self.options.mate_guard_plies))
        if guard_plies > 0 and self._side_can_force_checkmate(child, guard_plies):
            return -1.0
        return None

    def _ordered_moves(self, board: chess.Board, priors=None):
        priors = priors or {}
        moves = list(board.legal_moves)
        moves.sort(
            key=lambda move: (
                self._move_order_score(board, move, priors.get(move, 0.0)),
                move.uci(),
            ),
            reverse=True,
        )
        return moves

    def _stability_moves(self, board: chess.Board):
        if board.is_check():
            return self._ordered_moves(board)
        moves = []
        for move in board.legal_moves:
            include = board.is_capture(move) or bool(move.promotion)
            if not include:
                try:
                    include = board.gives_check(move)
                except Exception:
                    include = False
            if include:
                moves.append(move)
        moves.sort(
            key=lambda move: (self._move_order_score(board, move), move.uci()),
            reverse=True,
        )
        return moves

    def _quiescence(self, board, alpha, beta, depth):
        self._check_limit()
        self.nodes += 1

        terminal = terminal_value_side_to_move(board)
        if terminal is not None:
            return float(terminal)

        stand_pat = self._leaf_value(board)
        in_check = board.is_check()
        if not in_check:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            if depth <= 0:
                return alpha

        for move in self._stability_moves(board):
            board.push(move)
            if in_check and depth <= 0:
                terminal = terminal_value_side_to_move(board)
                child_value = (
                    float(terminal)
                    if terminal is not None
                    else self._leaf_value(board)
                )
                score = -child_value
            else:
                score = -self._quiescence(
                    board,
                    -beta,
                    -alpha,
                    depth - 1,
                )
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _negamax(self, board, depth, alpha, beta):
        self._check_limit()
        self.nodes += 1

        terminal = terminal_value_side_to_move(board)
        if terminal is not None:
            return float(terminal)

        if depth <= 0:
            return self._quiescence(
                board,
                alpha,
                beta,
                max(0, int(self.options.alpha_beta_quiescence)),
            )

        best = -1.1
        for move in self._ordered_moves(board):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return float(best)

    def verify(self, board: chess.Board, candidates: List[chess.Move]):
        scores = {}
        completed = True
        pending = []

        for move in candidates:
            try:
                child = board.copy(stack=False)
                child.push(move)
                quick_score = self._root_tactical_score(child)
                if quick_score is not None:
                    scores[move] = float(quick_score)
                    continue
                scores[move] = float(-self._leaf_value(child))
                pending.append((move, child))
            except VerificationLimit:
                completed = False
                break

        for move, child in pending:
            try:
                self._check_limit()
                score = -self._negamax(
                    child,
                    max(0, int(self.options.alpha_beta_depth) - 1),
                    -1.1,
                    1.1,
                )
                scores[move] = float(score)
            except VerificationLimit:
                completed = False
                break
        return scores, completed


class UnifiedSearch:
    """Neural MCTS with uncertainty-sized budget and Alpha-Beta verification."""

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

    @staticmethod
    def _promote_policy(policy, board: chess.Board, selected, incumbent):
        if selected is None or incumbent is None or selected == incumbent:
            return policy
        promoted = policy.copy()
        selected_index = move_to_index(selected)
        incumbent_index = move_to_index(incumbent)
        selected_value = float(promoted[selected_index])
        incumbent_value = float(promoted[incumbent_index])
        if selected_value <= incumbent_value:
            promoted[selected_index] = incumbent_value + max(
                1e-6,
                abs(incumbent_value) * 1e-6,
            )
        return normalize_policy(promoted, board)

    @staticmethod
    def _q_tiebreak_required_gain(
        margin: float,
        p_ratio_actual: float,
        visit_ratio_actual: float,
        p_ratio_floor: float,
        visit_ratio_floor: float,
    ) -> Tuple[float, Dict[str, float]]:
        margin = max(0.0, float(margin))

        def deficit(actual: float, floor: float) -> float:
            actual = max(0.0, float(actual))
            floor = max(0.0, float(floor))
            if actual >= 1.0:
                return 0.0
            if floor >= 1.0:
                return 1.0
            floor = min(floor, 1.0 - 1e-9)
            if actual <= floor:
                return 1.0
            return (1.0 - actual) / max(1e-9, 1.0 - floor)

        p_deficit = deficit(p_ratio_actual, p_ratio_floor)
        visit_deficit = deficit(visit_ratio_actual, visit_ratio_floor)
        closeness_penalty = max(p_deficit, visit_deficit)
        dynamic_gain = margin * closeness_penalty

        # If policy and visits are effectively tied, require only a small Q edge
        # so deterministic tie cases can be ordered by value without amplifying
        # pure floating-point noise.
        min_gain = 1e-6 if margin <= 0.0 else min(0.01, margin * 0.10)
        required = max(min_gain, dynamic_gain)
        return required, {
            "p_ratio_actual": float(p_ratio_actual),
            "visit_ratio_actual": float(visit_ratio_actual),
            "p_deficit": float(p_deficit),
            "visit_deficit": float(visit_deficit),
            "closeness_penalty": float(closeness_penalty),
        }

    @staticmethod
    def _q_tiebreak_effective_min_visits(
        configured_min_visits: int,
        incumbent_visits: int,
        visit_ratio: float,
    ) -> int:
        configured = max(0, int(configured_min_visits))
        if configured <= 0:
            return 0
        incumbent = max(0, int(incumbent_visits))
        if incumbent <= 0:
            return configured
        floor = min(8, configured)
        relative = int(math.ceil(incumbent * max(0.0, float(visit_ratio))))
        return max(0, min(configured, max(floor, relative)))

    def _q_tiebreak_move(
        self,
        board: chess.Board,
        final_policy,
        comparison_policy,
        root_node: Optional[MCTSNode],
        incumbent,
        verifier_scores: Optional[Dict[chess.Move, float]] = None,
    ):
        if (
            not bool(self.options.q_tiebreak)
            or root_node is None
            or incumbent is None
        ):
            return incumbent, None

        legal = list(board.legal_moves)
        if incumbent not in legal:
            return incumbent, None

        incumbent_child = root_node.children.get(incumbent)
        if incumbent_child is None:
            return incumbent, None

        min_visits = max(0, int(self.options.q_tiebreak_min_visits))
        incumbent_visits = int(incumbent_child.visit_count)
        incumbent_index = move_to_index(incumbent)
        incumbent_p = float(comparison_policy[incumbent_index])
        incumbent_final_p = float(final_policy[incumbent_index])
        incumbent_q = float(-incumbent_child.q)
        p_ratio = max(0.0, float(self.options.q_tiebreak_p_ratio))
        visit_ratio = max(0.0, float(self.options.q_tiebreak_visit_ratio))
        margin = max(0.0, float(self.options.q_tiebreak_margin))
        verifier_scores = verifier_scores or {}
        incumbent_verifier_score = verifier_scores.get(incumbent)
        effective_min_visits = self._q_tiebreak_effective_min_visits(
            configured_min_visits=min_visits,
            incumbent_visits=incumbent_visits,
            visit_ratio=visit_ratio,
        )
        if incumbent_visits < effective_min_visits:
            return incumbent, None

        best_move = None
        best_info = None
        best_key = None
        for candidate in legal:
            if candidate == incumbent:
                continue
            child = root_node.children.get(candidate)
            if child is None:
                continue

            candidate_verifier_score = verifier_scores.get(candidate)
            if incumbent_verifier_score is not None:
                if candidate_verifier_score is None:
                    continue
                if candidate_verifier_score + 1e-6 < incumbent_verifier_score:
                    continue

            visits = int(child.visit_count)
            if visits < effective_min_visits:
                continue
            visit_ratio_actual = (
                visits / max(1.0, float(incumbent_visits))
                if incumbent_visits > 0
                else 1.0
            )
            if incumbent_visits > 0 and visit_ratio_actual < visit_ratio:
                continue

            candidate_index = move_to_index(candidate)
            candidate_p = float(comparison_policy[candidate_index])
            p_ratio_actual = (
                candidate_p / max(1e-12, incumbent_p)
                if incumbent_p > 0.0
                else 1.0
            )
            if incumbent_p > 0.0 and p_ratio_actual < p_ratio:
                continue

            candidate_q = float(-child.q)
            q_gain = candidate_q - incumbent_q
            required_gain, dynamic_info = self._q_tiebreak_required_gain(
                margin=margin,
                p_ratio_actual=p_ratio_actual,
                visit_ratio_actual=visit_ratio_actual,
                p_ratio_floor=p_ratio,
                visit_ratio_floor=visit_ratio,
            )
            q_surplus = q_gain - required_gain
            if q_surplus < 0.0:
                continue

            key = (
                q_surplus,
                candidate_q,
                q_gain,
                candidate_p,
                visits,
                candidate.uci(),
            )
            if best_key is None or key > best_key:
                best_move = candidate
                best_key = key
                best_info = {
                    "from": incumbent.uci(),
                    "to": candidate.uci(),
                    "from_p": incumbent_p,
                    "to_p": candidate_p,
                    "from_final_p": incumbent_final_p,
                    "to_final_p": float(final_policy[candidate_index]),
                    "from_visits": incumbent_visits,
                    "to_visits": visits,
                    "from_alpha_beta_value": (
                        float(incumbent_verifier_score)
                        if incumbent_verifier_score is not None
                        else None
                    ),
                    "to_alpha_beta_value": (
                        float(candidate_verifier_score)
                        if candidate_verifier_score is not None
                        else None
                    ),
                    "min_visits": min_visits,
                    "effective_min_visits": effective_min_visits,
                    "from_q": incumbent_q,
                    "to_q": candidate_q,
                    "q_gain": q_gain,
                    "required_q_gain": required_gain,
                    "q_surplus": q_surplus,
                    **dynamic_info,
                }

        return (best_move, best_info) if best_move is not None else (incumbent, None)

    def search(self, board: chess.Board) -> SearchResult:
        if board.is_game_over(claim_draw=True):
            raise RuntimeError("game is already over")

        root_board = board.copy(stack=False)
        start = time.monotonic()
        final_deadline = None
        mcts_deadline = None
        if self.options.time_limit is not None and float(self.options.time_limit) > 0:
            total = float(self.options.time_limit)
            final_deadline = start + total
            reserve = max(0.0, min(0.9, float(self.options.alpha_beta_time_fraction)))
            mcts_deadline = start + total * (1.0 - reserve)

        root_node = None
        if self.model is None or int(self.options.mcts_sims) <= 0:
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
            evaluator = None
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
            evaluator = mcts.evaluator

        legal = list(root_board.legal_moves)
        mcts_move = self._select_top_move(root_board, mcts_policy)
        final_policy = mcts_policy.copy()
        verifier_scores: Dict[chess.Move, float] = {}
        verifier_completed = True
        verifier_nodes = 0
        overridden = False
        alpha_beta_decisive = False
        selected_by_verifier = None

        if (
            self.model is not None
            and evaluator is not None
            and int(self.options.alpha_beta_depth) > 0
            and root_node is not None
            and legal
        ):
            uncertainty = float(stats.get("uncertainty", 0.0))
            maximum_topk = max(2, int(self.options.alpha_beta_topk))
            requested_topk = 2 + int(
                round(uncertainty * max(0, maximum_topk - 2))
            )
            requested_topk = min(maximum_topk, max(2, requested_topk), len(legal))

            ranked = sorted(
                legal,
                key=lambda move: (
                    root_node.children.get(move).visit_count
                    if move in root_node.children
                    else 0,
                    root_node.children.get(move).prior
                    if move in root_node.children
                    else float(mcts_policy[move_to_index(move)]),
                    move.uci(),
                ),
                reverse=True,
            )
            candidates = ranked[:requested_topk]
            if mcts_move is not None and mcts_move not in candidates:
                candidates[-1] = mcts_move

            verifier = AlphaBetaVerifier(
                evaluator=evaluator,
                options=self.options,
                deadline=final_deadline,
            )
            verifier_scores, verifier_completed = verifier.verify(
                root_board,
                candidates,
            )
            verifier_nodes = verifier.nodes

            if verifier_scores:
                selected_by_verifier = max(
                    verifier_scores,
                    key=lambda move: (verifier_scores[move], move.uci()),
                )
                mcts_score = verifier_scores.get(mcts_move)
                best_score = verifier_scores[selected_by_verifier]
                decisive = (
                    best_score >= 0.999
                    or (
                        mcts_score is not None
                        and mcts_score <= -0.999
                    )
                )
                alpha_beta_decisive = bool(decisive)
                if (
                    mcts_score is None
                    or best_score - mcts_score
                    >= float(self.options.alpha_beta_margin)
                ):
                    overridden = selected_by_verifier != mcts_move
                    if overridden:
                        one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
                        one_hot[move_to_index(selected_by_verifier)] = 1.0
                        if decisive:
                            final_policy = one_hot
                        else:
                            gap = best_score - (
                                mcts_score if mcts_score is not None else -1.0
                            )
                            base_weight = max(
                                0.05,
                                min(
                                    float(self.options.alpha_beta_policy_weight),
                                    gap
                                    / max(
                                        1e-6,
                                        4.0 * float(self.options.alpha_beta_margin),
                                    )
                                    * float(self.options.alpha_beta_policy_weight),
                                ),
                            )
                            final_policy = normalize_policy(
                                (1.0 - base_weight) * final_policy
                                + base_weight * one_hot,
                                root_board,
                            )

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

        q_tiebreak = None
        if not alpha_beta_decisive:
            selected, q_tiebreak = self._q_tiebreak_move(
                root_board,
                final_policy,
                mcts_policy,
                root_node,
                move,
                verifier_scores,
            )
            if q_tiebreak is not None and selected in legal:
                final_policy = self._promote_policy(
                    final_policy,
                    root_board,
                    selected,
                    move,
                )
                move = selected

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
                "alpha_beta_value": (
                    float(verifier_scores[candidate])
                    if candidate in verifier_scores
                    else None
                ),
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
        info = {
            "search_type": "uncertainty_mcts_alpha_beta",
            "value": float(value),
            "mcts_soft_cap": int(self.options.mcts_sims),
            "mcts_dynamic_target": int(stats.get("dynamic_target", 0)),
            "sims_completed": int(stats.get("sims_completed", 0)),
            "uncertainty": float(stats.get("uncertainty", 0.0)),
            "nodes": int(stats.get("expanded_nodes", 0)),
            "nn_batches": int(stats.get("nn_batches", 0)),
            "alpha_beta_depth": int(self.options.alpha_beta_depth),
            "alpha_beta_nodes": int(verifier_nodes),
            "alpha_beta_completed": bool(verifier_completed),
            "alpha_beta_overrode_mcts": bool(overridden),
            "alpha_beta_decisive": bool(alpha_beta_decisive),
            "q_tiebreak_enabled": bool(self.options.q_tiebreak),
            "q_tiebreak_overrode": q_tiebreak is not None,
            "q_tiebreak": q_tiebreak,
            "q_tiebreak_move": (
                q_tiebreak["to"] if q_tiebreak is not None else None
            ),
            "mcts_move": mcts_move.uci() if mcts_move else None,
            "verifier_move": (
                selected_by_verifier.uci()
                if selected_by_verifier is not None
                else None
            ),
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
        description="ChessAI uncertainty MCTS with Alpha-Beta verification"
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_SIMS)
    parser.add_argument("--mcts-min-sims", type=int, default=0)
    parser.add_argument("--mcts-batch-size", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=5000)
    parser.add_argument("--c-puct", type=float, default=CPUCT)
    parser.add_argument("--alpha-beta-depth", type=int, default=4)
    parser.add_argument("--alpha-beta-topk", type=int, default=4)
    parser.add_argument("--alpha-beta-nodes", type=int, default=20000)
    parser.add_argument("--alpha-beta-quiescence", type=int, default=3)
    parser.add_argument("--alpha-beta-margin", type=float, default=0.10)
    parser.add_argument("--alpha-beta-time-fraction", type=float, default=0.25)
    parser.add_argument("--mate-guard-plies", type=int, default=3)
    parser.add_argument("--q-tiebreak", action="store_true", default=True)
    parser.add_argument("--no-q-tiebreak", dest="q_tiebreak", action="store_false")
    parser.add_argument("--q-tiebreak-min-visits", type=int, default=32)
    parser.add_argument("--q-tiebreak-p-ratio", type=float, default=0.90)
    parser.add_argument("--q-tiebreak-visit-ratio", type=float, default=0.80)
    parser.add_argument("--q-tiebreak-margin", type=float, default=0.25)
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
        mcts_sims=args.mcts_sims,
        mcts_min_sims=args.mcts_min_sims,
        mcts_batch_size=args.mcts_batch_size,
        time_limit=(args.movetime_ms / 1000.0) if args.movetime_ms > 0 else None,
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
    print("alpha_beta_nodes:", result.info["alpha_beta_nodes"])
    print("alpha_beta_overrode_mcts:", result.info["alpha_beta_overrode_mcts"])
    print("q_tiebreak_overrode:", result.info["q_tiebreak_overrode"])
    print("q_tiebreak:", result.info["q_tiebreak"])
    print("elapsed_ms:", result.info["elapsed_ms"])
    print("root:")
    for index, row in enumerate(result.info.get("root", []), 1):
        marker = "*" if row.get("selected") else " "
        print(
            f"{marker}{index:2d}. {row['san']:8s} {row['move']:5s} "
            f"p={row['p']:.5f} mcts={row['mcts_p']:.5f} "
            f"visits={row['visits']:4d} q={row['q']:+.4f} "
            f"ab={row['alpha_beta_value']}"
        )


if __name__ == "__main__":
    main()
