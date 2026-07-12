import chess
import numpy as np
from config import NUM_ACTIONS

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

KNIGHT_MOVES = [
    (-2, -1), (-2, 1),
    (-1, -2), (-1, 2),
    (1, -2),  (1, 2),
    (2, -1),  (2, 1),
]

UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
PROMOTION_DCS = [-1, 0, 1]

def move_to_index(move: chess.Move) -> int:
    f, t = move.from_square, move.to_square
    fr, fc = divmod(f, 8)
    tr, tc = divmod(t, 8)
    dr, dc = tr - fr, tc - fc

    # Underpromotions only. Queen promotions are queen-like moves.
    if move.promotion in UNDERPROMOTION_PIECES:
        if dc not in PROMOTION_DCS:
            raise ValueError(f"bad promotion direction: {move}")
        dir_idx = PROMOTION_DCS.index(dc)
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        return f * 73 + 64 + dir_idx * 3 + piece_idx

    # Queen-like moves: includes castling, pawn moves, captures, queen promotions.
    for d, (rr, cc) in enumerate(DIRECTIONS):
        for dist in range(1, 8):
            if dr == rr * dist and dc == cc * dist:
                return f * 73 + d * 7 + (dist - 1)

    # Knight
    for i, (rr, cc) in enumerate(KNIGHT_MOVES):
        if dr == rr and dc == cc:
            return f * 73 + 56 + i

    raise ValueError(f"cannot encode move: {move}")

def legal_move_map(board: chess.Board) -> dict:
    out = {}
    for m in board.legal_moves:
        out[move_to_index(m)] = m
    return out

def legal_indices(board: chess.Board) -> list:
    return list(legal_move_map(board).keys())

def index_to_legal_move(index: int, board: chess.Board):
    index = int(index)
    for m in board.legal_moves:
        if move_to_index(m) == index:
            return m
    return None

def policy_to_legal_distribution(policy: np.ndarray, board: chess.Board, normalize=True):
    d = {}
    s = 0.0
    for m in board.legal_moves:
        idx = move_to_index(m)
        p = float(policy[idx])
        if p < 0:
            p = 0.0
        d[m] = p
        s += p
    if normalize:
        if s <= 0:
            n = len(d)
            return {m: 1.0 / n for m in d} if n else {}
        return {m: p / s for m, p in d.items()}
    return d
