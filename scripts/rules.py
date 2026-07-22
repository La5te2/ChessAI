from __future__ import annotations

from typing import Optional

import chess


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
