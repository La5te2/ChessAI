from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from config import (
    CPUCT,
    DEFAULT_SIMS,
    DEVICE,
    MODEL_PATH,
)
from decision import profile_for_model
from evaluator import BatchedEvaluator
from model import load_model

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
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def normalize_policy(policy: np.ndarray, board: chess.Board, codec) -> np.ndarray:
    output = np.zeros(codec.action_size, dtype=np.float32)
    legal = list(board.legal_moves)
    if not legal:
        return output

    total = 0.0
    for move in legal:
        index = codec.move_to_index(move)
        value = max(0.0, float(policy[index])) if index < len(policy) else 0.0
        output[index] = value
        total += value

    if total <= 0:
        probability = 1.0 / len(legal)
        for move in legal:
            output[codec.move_to_index(move)] = probability
    else:
        output /= total
    return output


@dataclass
class SearchOptions:
    search_type: str = "only-mcts"
    # mcts_sims is a soft upper budget. Confident positions may stop earlier.
    mcts_sims: int = DEFAULT_SIMS
    mcts_min_sims: int = 0
    mcts_batch_size: int = 32
    c_puct: float = CPUCT
    c_puct_base: float = 19652.0
    c_puct_factor: float = 1.0
    fpu_reduction: float = 0.15
    time_limit: Optional[float] = None
    progress_interval_sec: float = 0.75
    root_topn: int = 10
    virtual_loss: float = 0.0


VALID_SEARCH_TYPES = {"closed", "only-mcts"}


def normalize_search_type(search_type: str) -> str:
    value = str(search_type).strip().lower()
    if value not in VALID_SEARCH_TYPES:
        raise ValueError(
            f"search_type must be one of {sorted(VALID_SEARCH_TYPES)}, got {search_type!r}"
        )
    return value


class MCTSNode:
    def __init__(self, prior=0.0, profile_state: Optional[Dict[str, Any]] = None):
        self.prior = float(prior)
        self.profile_state = dict(profile_state or {})
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
        self.profile = profile_for_model(model)
        self.codec = self.profile.move_codec
        self.action_size = self.codec.action_size
        self.evaluator = BatchedEvaluator(
            model,
            device=self.device,
            batch_size=max(1, int(options.mcts_batch_size)),
        )
        self.expanded_nodes = 0
        self.nn_batches = 0
        self.max_leaf_depth = 0
        self.total_leaf_depth = 0
        self.leaf_samples = 0

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
        fpu = self._fpu_value(parent)
        q_from_parent = (
            -child.q
            if child.visit_count > 0
            else self.profile.unvisited_q_from_parent(fpu, child)
        )
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

    def _expand_from_policy(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy,
        expansion_payload=None,
    ):
        priors = self.codec.policy_to_legal_distribution(policy, board, normalize=True)
        if not node.expanded():
            for move, prior in priors.items():
                profile_state = self.profile.child_profile_state(expansion_payload, move)
                node.children[move] = MCTSNode(prior=prior, profile_state=profile_state)
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

    def _record_leaf_depth(self, path):
        depth = max(0, len(path) - 1)
        self.max_leaf_depth = max(self.max_leaf_depth, depth)
        self.total_leaf_depth += depth
        self.leaf_samples += 1

    def _depth_stats(self):
        average = (
            self.total_leaf_depth / self.leaf_samples
            if self.leaf_samples > 0
            else 0.0
        )
        return {
            "leaf_samples": int(self.leaf_samples),
            "avg_leaf_depth": float(average),
            "max_leaf_depth": int(self.max_leaf_depth),
        }

    def _policy_from_root(self, root: MCTSNode, board: chess.Board, root_policy):
        policy = np.zeros(self.action_size, dtype=np.float32)
        for move, child in root.children.items():
            policy[self.codec.move_to_index(move)] = (
                float(child.visit_count) + max(0.0, float(child.prior))
            )
        if float(policy.sum()) <= 0:
            policy = np.asarray(root_policy, dtype=np.float32)
        return normalize_policy(policy, board, self.codec)

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

    @staticmethod
    def _cancelled(cancel_event) -> bool:
        return bool(cancel_event is not None and cancel_event.is_set())

    def run(
        self,
        board: chess.Board,
        deadline=None,
        cancel_event=None,
        progress_callback: Optional[Callable[[np.ndarray, float, Dict[str, Any]], None]] = None,
        progress_interval_sec: Optional[float] = None,
        progress_interval_sims: int = 0,
    ):
        root = MCTSNode(0.0)
        terminal = terminal_value_side_to_move(board)
        if terminal is not None:
            stats = {
                "sims_completed": 0,
                "dynamic_target": 0,
                "decision_profile": self.profile.name,
                "uncertainty": 0.0,
                "expanded_nodes": 0,
                "nn_batches": 0,
                "timeout": False,
                "cancelled": self._cancelled(cancel_event),
                "root_node": root,
                "root_c_puct": self._scheduled_c_puct(root),
                "root_fpu": self._fpu_value(root),
            }
            stats.update(self._depth_stats())
            return np.zeros(self.action_size, dtype=np.float32), terminal, stats

        root_policy, root_value, root_payload = self.evaluator.evaluate_one_full(board)
        self.nn_batches += 1
        self._expand_from_policy(root, board, root_policy, root_payload)

        soft_cap = max(0, int(self.options.mcts_sims))
        batch_size = max(1, int(self.options.mcts_batch_size))
        if soft_cap <= 0:
            policy = normalize_policy(root_policy, board, self.codec)
            stats = {
                "sims_completed": 0,
                "dynamic_target": 0,
                "decision_profile": self.profile.name,
                "uncertainty": root_uncertainty(root),
                "expanded_nodes": self.expanded_nodes,
                "nn_batches": self.nn_batches,
                "timeout": False,
                "cancelled": self._cancelled(cancel_event),
                "root_node": root,
                "root_c_puct": self._scheduled_c_puct(root),
                "root_fpu": self._fpu_value(root),
            }
            stats.update(self._depth_stats())
            if progress_callback is not None:
                progress_callback(policy, float(root_value), dict(stats))
            return policy, float(root_value), stats

        configured_minimum = int(self.options.mcts_min_sims)
        minimum = configured_minimum if configured_minimum > 0 else max(batch_size, soft_cap // 4)
        minimum = max(1, min(soft_cap, minimum))
        dynamic_target = minimum
        sims_completed = 0
        uncertainty = 1.0
        last_progress_time = 0.0
        last_progress_sims = 0
        progress_interval = (
            float(self.options.progress_interval_sec)
            if progress_interval_sec is None
            else float(progress_interval_sec)
        )

        def emit_progress(force=False):
            nonlocal last_progress_time, last_progress_sims
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force:
                if (
                    progress_interval_sims > 0
                    and sims_completed - last_progress_sims < progress_interval_sims
                ):
                    return
                if (
                    progress_interval > 0
                    and now - last_progress_time < progress_interval
                ):
                    return
            policy = self._policy_from_root(root, board, root_policy)
            stats = {
                "sims_completed": int(sims_completed),
                "dynamic_target": int(dynamic_target),
                "decision_profile": self.profile.name,
                "uncertainty": float(root_uncertainty(root)),
                "expanded_nodes": int(self.expanded_nodes),
                "nn_batches": int(self.nn_batches),
                "timeout": bool(deadline is not None and now >= deadline),
                "cancelled": self._cancelled(cancel_event),
                "root_node": root,
                "root_c_puct": float(self._scheduled_c_puct(root)),
                "root_fpu": float(self._fpu_value(root)),
            }
            stats.update(self._depth_stats())
            last_progress_time = now
            last_progress_sims = sims_completed
            progress_callback(
                policy,
                float(root.q if root.visit_count else root_value),
                stats,
            )

        while (
            sims_completed < soft_cap
            and sims_completed < dynamic_target
            and not self._cancelled(cancel_event)
        ):
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
                if self._cancelled(cancel_event):
                    break

                leaf, leaf_board, path, terminal = self._select_leaf(root, board)
                self._record_leaf_depth(path)
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
                evaluation = self.evaluator.evaluate_boards_full(
                    [item[1] for item in selected]
                )
                self.nn_batches += 1
                for index, (leaf, leaf_board, path) in enumerate(selected):
                    policy = evaluation.policies[index]
                    value = evaluation.values[index]
                    payload = self.profile.payload_for_index(
                        evaluation.expansion_payload,
                        index,
                    )
                    self._expand_from_policy(leaf, leaf_board, policy, payload)
                    self._clear_virtual(path)
                    self._backpropagate(path, float(value))
                    sims_completed += 1

            if sims_completed >= minimum:
                uncertainty = root_uncertainty(root)
                desired = minimum + int(
                    math.ceil(uncertainty * max(0, soft_cap - minimum))
                )
                dynamic_target = max(minimum, min(soft_cap, desired))

            emit_progress()

            if not selected and attempts >= max_attempts:
                break

        policy = self._policy_from_root(root, board, root_policy)

        stats = {
            "sims_completed": int(sims_completed),
            "dynamic_target": int(dynamic_target),
            "decision_profile": self.profile.name,
            "uncertainty": float(root_uncertainty(root)),
            "expanded_nodes": int(self.expanded_nodes),
            "nn_batches": int(self.nn_batches),
            "timeout": bool(deadline is not None and time.monotonic() >= deadline),
            "cancelled": self._cancelled(cancel_event),
            "root_node": root,
            "root_c_puct": float(self._scheduled_c_puct(root)),
            "root_fpu": float(self._fpu_value(root)),
        }
        stats.update(self._depth_stats())
        emit_progress(force=progress_callback is not None)
        return policy, float(root.q if root.visit_count else root_value), stats

    def run_many(self, boards: List[chess.Board], deadline=None):
        """Run independent MCTS trees while batching leaves across root positions."""
        if not boards:
            return []

        batch_size = max(1, int(self.options.mcts_batch_size))
        soft_cap = max(0, int(self.options.mcts_sims))
        configured_minimum = int(self.options.mcts_min_sims)
        minimum = (
            configured_minimum
            if configured_minimum > 0
            else max(batch_size, soft_cap // 4)
        )
        minimum = max(1, min(soft_cap, minimum)) if soft_cap > 0 else 0
        deadlines = (
            list(deadline)
            if isinstance(deadline, (list, tuple))
            else [deadline] * len(boards)
        )
        if len(deadlines) != len(boards):
            raise ValueError("deadline count must match board count")

        states = []
        root_eval_indices = []
        root_eval_boards = []
        for index, source_board in enumerate(boards):
            board = source_board.copy(stack=False)
            root = MCTSNode(0.0)
            terminal = terminal_value_side_to_move(board)
            state = {
                "board": board,
                "root": root,
                "root_policy": np.zeros(self.action_size, dtype=np.float32),
                "root_value": float(terminal or 0.0),
                "sims_completed": 0,
                "dynamic_target": minimum,
                "expanded_nodes": 0,
                "nn_batches": 0,
                "total_leaf_depth": 0,
                "leaf_samples": 0,
                "max_leaf_depth": 0,
                "terminal": terminal is not None,
                "deadline": deadlines[index],
            }
            states.append(state)
            if terminal is None:
                root_eval_indices.append(index)
                root_eval_boards.append(board)

        if root_eval_boards:
            evaluation = self.evaluator.evaluate_boards_full(root_eval_boards)
            for batch_index, state_index in enumerate(root_eval_indices):
                state = states[state_index]
                policy = evaluation.policies[batch_index]
                value = float(evaluation.values[batch_index])
                payload = self.profile.payload_for_index(
                    evaluation.expansion_payload,
                    batch_index,
                )
                self._expand_from_policy_without_counters(
                    state["root"],
                    state["board"],
                    policy,
                    payload,
                )
                state["root_policy"] = policy
                state["root_value"] = value
                state["expanded_nodes"] = 1
                state["nn_batches"] = 1

        if soft_cap > 0:
            while True:
                selected = []
                active = False
                now = time.monotonic()
                for state_index, state in enumerate(states):
                    if state["terminal"]:
                        continue
                    if state["sims_completed"] >= soft_cap:
                        continue
                    if state["sims_completed"] >= state["dynamic_target"]:
                        continue
                    if state["deadline"] is not None and now >= state["deadline"]:
                        continue
                    active = True

                    wanted = min(
                        batch_size,
                        soft_cap - state["sims_completed"],
                        max(1, state["dynamic_target"] - state["sims_completed"]),
                    )
                    selected_nodes = set()
                    local_selected = 0
                    attempts = 0
                    max_attempts = max(wanted * 5, wanted + 8)
                    while (
                        local_selected < wanted
                        and attempts < max_attempts
                        and state["sims_completed"] + local_selected < soft_cap
                        and state["sims_completed"] + local_selected
                        < state["dynamic_target"]
                    ):
                        attempts += 1
                        if (
                            state["deadline"] is not None
                            and time.monotonic() >= state["deadline"]
                        ):
                            break
                        leaf, leaf_board, path, terminal = self._select_leaf(
                            state["root"],
                            state["board"],
                        )
                        depth = max(0, len(path) - 1)
                        state["max_leaf_depth"] = max(
                            state["max_leaf_depth"], depth
                        )
                        state["total_leaf_depth"] += depth
                        state["leaf_samples"] += 1
                        if terminal is not None:
                            self._clear_virtual(path)
                            self._backpropagate(path, terminal)
                            state["sims_completed"] += 1
                            continue
                        leaf_identity = id(leaf)
                        if leaf_identity in selected_nodes:
                            self._clear_virtual(path)
                            continue
                        selected_nodes.add(leaf_identity)
                        selected.append(
                            (local_selected, state_index, leaf, leaf_board, path)
                        )
                        local_selected += 1

                if selected:
                    selected.sort(key=lambda item: (item[0], item[1]))
                    for start in range(0, len(selected), batch_size):
                        chunk = selected[start:start + batch_size]
                        ready = []
                        for item in chunk:
                            state = states[item[1]]
                            if (
                                state["deadline"] is not None
                                and time.monotonic() >= state["deadline"]
                            ):
                                self._clear_virtual(item[4])
                            else:
                                ready.append(item)
                        if not ready:
                            continue
                        evaluation = self.evaluator.evaluate_boards_full(
                            [item[3] for item in ready]
                        )
                        evaluated_states = set()
                        for batch_index, (_, state_index, leaf, leaf_board, path) in enumerate(ready):
                            state = states[state_index]
                            policy = evaluation.policies[batch_index]
                            value = float(evaluation.values[batch_index])
                            payload = self.profile.payload_for_index(
                                evaluation.expansion_payload,
                                batch_index,
                            )
                            was_expanded = leaf.expanded()
                            self._expand_from_policy_without_counters(
                                leaf,
                                leaf_board,
                                policy,
                                payload,
                            )
                            if not was_expanded:
                                state["expanded_nodes"] += 1
                            evaluated_states.add(state_index)
                            self._clear_virtual(path)
                            self._backpropagate(path, value)
                            state["sims_completed"] += 1
                        for state_index in evaluated_states:
                            states[state_index]["nn_batches"] += 1

                for state in states:
                    if not state["terminal"]:
                        self._update_batch_dynamic_target(state, minimum, soft_cap)

                if not active:
                    break

        outputs = []
        now = time.monotonic()
        for state in states:
            root = state["root"]
            board = state["board"]
            policy = (
                normalize_policy(state["root_policy"], board, self.codec)
                if soft_cap <= 0
                else self._policy_from_root(root, board, state["root_policy"])
            )
            leaf_samples = int(state["leaf_samples"])
            stats = {
                "sims_completed": int(state["sims_completed"]),
                "dynamic_target": int(state["dynamic_target"] if soft_cap > 0 else 0),
                "decision_profile": self.profile.name,
                "uncertainty": float(root_uncertainty(root)),
                "expanded_nodes": int(state["expanded_nodes"]),
                "nn_batches": int(state["nn_batches"]),
                "timeout": bool(
                    state["deadline"] is not None and now >= state["deadline"]
                ),
                "cancelled": False,
                "root_node": root,
                "root_c_puct": float(self._scheduled_c_puct(root)),
                "root_fpu": float(self._fpu_value(root)),
                "leaf_samples": leaf_samples,
                "avg_leaf_depth": float(
                    state["total_leaf_depth"] / leaf_samples if leaf_samples else 0.0
                ),
                "max_leaf_depth": int(state["max_leaf_depth"]),
            }
            value = float(root.q if root.visit_count else state["root_value"])
            outputs.append((policy, value, stats))
        return outputs

    def _expand_from_policy_without_counters(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy,
        expansion_payload=None,
    ):
        priors = self.codec.policy_to_legal_distribution(policy, board, normalize=True)
        if node.expanded():
            return
        for move, prior in priors.items():
            profile_state = self.profile.child_profile_state(expansion_payload, move)
            node.children[move] = MCTSNode(prior=prior, profile_state=profile_state)

    @staticmethod
    def _update_batch_dynamic_target(state, minimum: int, soft_cap: int):
        if state["sims_completed"] < minimum:
            return
        uncertainty = root_uncertainty(state["root"])
        desired = minimum + int(
            math.ceil(uncertainty * max(0, soft_cap - minimum))
        )
        state["dynamic_target"] = max(minimum, min(soft_cap, desired))


class UnifiedSearch:
    """Neural policy and MCTS search."""

    def __init__(self, model: Optional[torch.nn.Module], options=None, device=None):
        self.model = model
        self.options = options or SearchOptions()
        if device is None and self.model is not None:
            device = str(model_device(self.model))
        elif device is None:
            device = DEVICE
        self.device = str(device)
        self.profile = profile_for_model(self.model) if self.model is not None else None
        self.codec = self.profile.move_codec if self.profile is not None else None
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _uniform_probability(self, board: chess.Board) -> float:
        return 1.0 / max(1, board.legal_moves.count())

    def _policy_value_for_move(
        self,
        board: chess.Board,
        policy: np.ndarray,
        move: chess.Move,
    ) -> float:
        if self.codec is None:
            return self._uniform_probability(board)
        index = self.codec.move_to_index(move)
        return float(policy[index]) if index < len(policy) else 0.0

    def _policy_only(self, board: chess.Board):
        legal = list(board.legal_moves)
        policy = np.zeros(self.codec.action_size if self.codec is not None else 0, dtype=np.float32)
        if not legal:
            return policy, 0.0
        if self.model is None:
            return policy, 0.0

        evaluator = BatchedEvaluator(
            self.model,
            device=str(model_device(self.model)),
            batch_size=max(1, int(self.options.mcts_batch_size)),
        )
        raw_policy, value = evaluator.evaluate_one(board)
        return normalize_policy(raw_policy, board, self.codec), float(value)

    def _select_top_move(self, board, policy):
        legal = list(board.legal_moves)
        if not legal:
            return None
        if self.codec is None:
            return random.choice(legal)
        return max(
            legal,
            key=lambda move: (
                max(0.0, self._policy_value_for_move(board, policy, move)),
                move.uci(),
            ),
        )

    def _make_result(
        self,
        root_board: chess.Board,
        search_type: str,
        policy: np.ndarray,
        value: float,
        stats: Dict[str, Any],
        start_time: float,
        final_deadline=None,
        *,
        partial: bool = False,
    ) -> SearchResult:
        legal = list(root_board.legal_moves)
        if not legal:
            raise RuntimeError("no legal moves")

        root_node = stats.get("root_node")
        mcts_move = self._select_top_move(root_board, policy)
        move = mcts_move
        if move is None or move not in legal:
            move = max(
                legal,
                key=lambda candidate: (
                    self._policy_value_for_move(root_board, policy, candidate),
                    candidate.uci(),
                ),
            )

        root_lines = []
        for candidate in legal:
            policy_value = self._policy_value_for_move(root_board, policy, candidate)
            child = (
                root_node.children.get(candidate)
                if root_node is not None
                else None
            )
            row = {
                "move": candidate.uci(),
                "san": safe_san(root_board, candidate),
                "selected": candidate == move,
                "p": float(policy_value),
                "mcts_p": float(policy_value),
                "visits": int(child.visit_count) if child is not None else 0,
                "prior": float(child.prior) if child is not None else float(policy_value),
                "q": float(-child.q) if child is not None else 0.0,
            }
            row.update(self.profile.root_row_fields(child) if self.profile is not None else {})
            root_lines.append(row)
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

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        effective_mcts_sims = 0 if search_type == "closed" else int(self.options.mcts_sims)
        timed_out = bool(stats.get("timeout", False)) or bool(
            final_deadline is not None
            and time.monotonic() >= final_deadline
        )

        info = {
            "search_type": search_type,
            "decision_profile": str(
                stats.get(
                    "decision_profile",
                    self.profile.name if self.profile is not None else "uniform_legal",
                )
            ),
            "partial": bool(partial),
            "cancelled": bool(stats.get("cancelled", False)),
            "value": float(value),
            "mcts_soft_cap": int(effective_mcts_sims),
            "mcts_dynamic_target": int(stats.get("dynamic_target", 0)),
            "sims_completed": int(stats.get("sims_completed", 0)),
            "uncertainty": float(stats.get("uncertainty", 0.0)),
            "nodes": int(stats.get("expanded_nodes", 0)),
            "nn_batches": int(stats.get("nn_batches", 0)),
            "leaf_samples": int(stats.get("leaf_samples", 0)),
            "avg_leaf_depth": float(stats.get("avg_leaf_depth", 0.0)),
            "max_leaf_depth": int(stats.get("max_leaf_depth", 0)),
            "c_puct_initial": float(self.options.c_puct),
            "c_puct_base": float(self.options.c_puct_base),
            "c_puct_factor": float(self.options.c_puct_factor),
            "c_puct_root": float(stats.get("root_c_puct", self.options.c_puct)),
            "fpu_reduction": float(self.options.fpu_reduction),
            "fpu_root": float(stats.get("root_fpu", 0.0)),
            "virtual_loss": float(self.options.virtual_loss),
            "mcts_move": mcts_move.uci() if mcts_move else None,
            "piece_count": count_pieces(root_board),
            "best_move": move.uci(),
            "best_san": safe_san(root_board, move),
            "timeout": bool(timed_out),
            "elapsed_ms": round(elapsed_ms, 2),
            "selection": {
                "selection_mode": "uniform_legal" if self.codec is None else "top1",
                "selected_move": move.uci(),
            },
            "root": root_lines[: max(1, int(self.options.root_topn))],
        }

        return SearchResult(
            move=move,
            policy=policy.astype(np.float32),
            value=float(value),
            mcts_policy=policy.astype(np.float32),
            info=info,
        )

    def search(
        self,
        board: chess.Board,
        cancel_event=None,
        progress_callback: Optional[Callable[[SearchResult], None]] = None,
    ) -> SearchResult:
        if board.is_game_over(claim_draw=False):
            raise RuntimeError("game is already over")

        search_type = normalize_search_type(self.options.search_type)
        root_board = board.copy(stack=False)
        start = time.monotonic()
        final_deadline = None
        mcts_deadline = None
        if self.options.time_limit is not None and float(self.options.time_limit) > 0:
            total = float(self.options.time_limit)
            final_deadline = start + total
            mcts_deadline = final_deadline

        if (
            search_type == "closed"
            or self.model is None
            or int(self.options.mcts_sims) <= 0
        ):
            mcts_policy, value = self._policy_only(root_board)
            stats = {
                "sims_completed": 0,
                "dynamic_target": 0,
                "decision_profile": self.profile.name if self.profile is not None else "uniform_legal",
                "uncertainty": 0.0,
                "expanded_nodes": 0,
                "nn_batches": 1 if self.model is not None else 0,
                "timeout": False,
                "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
                "root_node": None,
            }
        else:
            mcts = MCTS(
                self.model,
                self.options,
                device=str(model_device(self.model)),
            )

            def on_mcts_progress(policy, progress_value, progress_stats):
                if progress_callback is None:
                    return
                try:
                    progress_callback(
                        self._make_result(
                            root_board,
                            search_type,
                            policy,
                            progress_value,
                            progress_stats,
                            start,
                            final_deadline,
                            partial=True,
                        )
                    )
                except Exception:
                    pass

            mcts_policy, value, stats = mcts.run(
                root_board,
                deadline=mcts_deadline,
                cancel_event=cancel_event,
                progress_callback=on_mcts_progress if progress_callback else None,
                progress_interval_sec=float(self.options.progress_interval_sec),
            )

        cancelled = bool(cancel_event is not None and cancel_event.is_set())
        if cancelled:
            stats["cancelled"] = True

        return self._make_result(
            root_board,
            search_type,
            mcts_policy,
            value,
            stats,
            start,
            final_deadline,
            partial=False,
        )

    def search_many(self, boards: List[chess.Board]) -> List[SearchResult]:
        """Search several independent positions with shared neural batches."""
        if not boards:
            return []
        for board in boards:
            if board.is_game_over(claim_draw=False):
                raise RuntimeError("game is already over")

        search_type = normalize_search_type(self.options.search_type)
        root_boards = [board.copy(stack=False) for board in boards]
        start_times = [time.monotonic()] * len(root_boards)
        deadlines = [None] * len(root_boards)
        if self.options.time_limit is not None and float(self.options.time_limit) > 0:
            deadline = time.monotonic() + float(self.options.time_limit)
            deadlines = [deadline] * len(root_boards)

        if self.model is None:
            raw_results = []
            for board in root_boards:
                raw_results.append((
                    np.zeros(0, dtype=np.float32),
                    0.0,
                    {
                        "sims_completed": 0,
                        "dynamic_target": 0,
                        "decision_profile": "uniform_legal",
                        "uncertainty": 0.0,
                        "expanded_nodes": 0,
                        "nn_batches": 0,
                        "timeout": False,
                        "cancelled": False,
                        "root_node": None,
                    },
                ))
        elif search_type == "closed" or int(self.options.mcts_sims) <= 0:
            evaluator = BatchedEvaluator(
                self.model,
                device=str(model_device(self.model)),
                batch_size=max(1, int(self.options.mcts_batch_size)),
            )
            evaluation = evaluator.evaluate_boards_full(root_boards)
            raw_results = []
            for index, board in enumerate(root_boards):
                policy = normalize_policy(
                    evaluation.policies[index],
                    board,
                    self.codec,
                )
                raw_results.append((
                    policy,
                    float(evaluation.values[index]),
                    {
                        "sims_completed": 0,
                        "dynamic_target": 0,
                        "decision_profile": self.profile.name,
                        "uncertainty": 0.0,
                        "expanded_nodes": 0,
                        "nn_batches": 1,
                        "timeout": False,
                        "cancelled": False,
                        "root_node": None,
                    },
                ))
        else:
            mcts = MCTS(
                self.model,
                self.options,
                device=str(model_device(self.model)),
            )
            raw_results = mcts.run_many(root_boards, deadline=deadlines)

        return [
            self._make_result(
                board,
                search_type,
                policy,
                value,
                stats,
                start_times[index],
                deadlines[index],
                partial=False,
            )
            for index, (board, (policy, value, stats)) in enumerate(
                zip(root_boards, raw_results)
            )
        ]


def select_move(
    board,
    model,
    options=None,
    device=None,
    cancel_event=None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[chess.Move, Dict]:
    def on_progress(result: SearchResult):
        if progress_callback is not None:
            progress_callback(result.info)

    result = UnifiedSearch(model, options or SearchOptions(), device=device).search(
        board,
        cancel_event=cancel_event,
        progress_callback=on_progress if progress_callback is not None else None,
    )
    return result.move, result.info


def get_suggestions(
    board,
    model,
    options=None,
    topn=5,
    device=None,
    cancel_event=None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict]:
    options = options or SearchOptions(root_topn=topn)
    options = dataclasses.replace(options, root_topn=topn)
    def on_progress(result: SearchResult):
        if progress_callback is not None:
            progress_callback(result.info)

    result = UnifiedSearch(model, options, device=device).search(
        board,
        cancel_event=cancel_event,
        progress_callback=on_progress if progress_callback is not None else None,
    )
    return result.info.get("root", [])[:topn]


def parse_args():
    parser = argparse.ArgumentParser(
        description="ChessAI policy/MCTS search"
    )
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--search-type",
        choices=sorted(VALID_SEARCH_TYPES),
        default="only-mcts",
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
        root_topn=args.root_topn,
    )
    result = UnifiedSearch(model, options, device=args.device).search(board)
    print("fen:", board.fen())
    print("best:", result.info["best_san"], result.move.uci())
    print("decision_profile:", result.info["decision_profile"])
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
    print("leaf_depth_avg:", result.info["avg_leaf_depth"])
    print("leaf_depth_max:", result.info["max_leaf_depth"])
    print("elapsed_ms:", result.info["elapsed_ms"])
    print("root:")
    for index, row in enumerate(result.info.get("root", []), 1):
        marker = "*" if row.get("selected") else " "
        extra = ""
        if "adv" in row:
            extra += f" adv={float(row['adv']):+.4f}"
        print(
            f"{marker}{index:2d}. {row['san']:8s} {row['move']:5s} "
            f"p={row['p']:.5f} mcts={row['mcts_p']:.5f} "
            f"visits={row['visits']:4d} q={row['q']:+.4f}{extra}"
        )


if __name__ == "__main__":
    main()
