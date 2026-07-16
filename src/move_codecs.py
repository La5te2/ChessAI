from dataclasses import dataclass
from typing import Dict, Optional

import chess
import numpy as np


MOVE_ENCODING_AZ_64X73 = "alphazero_64x73"
MOVE_ENCODING_SD_64X64_UP9 = "sd_64x64_underpromo9"
BOARD_SQUARES = 64
POLICY_PLANES = 73

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
UNDERPROMOTION_PLANES = len(PROMOTION_DCS) * len(UNDERPROMOTION_PIECES)
AZ_64X73_ACTION_SIZE = BOARD_SQUARES * POLICY_PLANES
SD_64X64_UP9_ACTION_SIZE = BOARD_SQUARES * BOARD_SQUARES + BOARD_SQUARES * UNDERPROMOTION_PLANES


@dataclass(frozen=True)
class MoveCodec:
    name: str
    action_size: int

    def move_to_index(self, move: chess.Move) -> int:
        raise NotImplementedError

    def index_to_move(self, index: int, board: chess.Board) -> Optional[chess.Move]:
        for move in board.legal_moves:
            if self.move_to_index(move) == index:
                return move
        return None

    def policy_to_legal_distribution(
        self,
        policy: np.ndarray,
        board: chess.Board,
        normalize: bool = True,
    ) -> Dict[chess.Move, float]:
        distribution = {}
        total = 0.0
        for move in board.legal_moves:
            index = self.move_to_index(move)
            value = float(policy[index]) if index < len(policy) else 0.0
            if value < 0:
                value = 0.0
            distribution[move] = value
            total += value
        if normalize and total > 0:
            distribution = {move: value / total for move, value in distribution.items()}
        return distribution


class AlphaZero64x73Codec(MoveCodec):
    def __init__(self):
        super().__init__(MOVE_ENCODING_AZ_64X73, AZ_64X73_ACTION_SIZE)

    def move_to_index(self, move: chess.Move) -> int:
        from_square, to_square = move.from_square, move.to_square
        from_rank, from_file = divmod(from_square, 8)
        to_rank, to_file = divmod(to_square, 8)
        dr, dc = to_rank - from_rank, to_file - from_file

        if move.promotion in UNDERPROMOTION_PIECES:
            if dc not in PROMOTION_DCS:
                raise ValueError(f"bad promotion direction: {move}")
            dir_idx = PROMOTION_DCS.index(dc)
            piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
            return from_square * POLICY_PLANES + BOARD_SQUARES + dir_idx * 3 + piece_idx

        for direction, (rr, cc) in enumerate(DIRECTIONS):
            for distance in range(1, 8):
                if dr == rr * distance and dc == cc * distance:
                    return from_square * POLICY_PLANES + direction * 7 + (distance - 1)

        for offset, (rr, cc) in enumerate(KNIGHT_MOVES):
            if dr == rr and dc == cc:
                return from_square * POLICY_PLANES + 56 + offset

        raise ValueError(f"cannot encode move: {move}")


class SourceDestinationUnderpromotionCodec(MoveCodec):
    def __init__(self):
        super().__init__(MOVE_ENCODING_SD_64X64_UP9, SD_64X64_UP9_ACTION_SIZE)

    def move_to_index(self, move: chess.Move) -> int:
        if move.promotion in UNDERPROMOTION_PIECES:
            _, from_file = divmod(move.from_square, 8)
            _, to_file = divmod(move.to_square, 8)
            dc = to_file - from_file
            if dc not in PROMOTION_DCS:
                raise ValueError(f"bad promotion direction: {move}")
            dir_idx = PROMOTION_DCS.index(dc)
            piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
            return BOARD_SQUARES * BOARD_SQUARES + move.from_square * 9 + dir_idx * 3 + piece_idx
        return move.from_square * BOARD_SQUARES + move.to_square

    def index_to_move(self, index: int, board: chess.Board) -> Optional[chess.Move]:
        if index < 0 or index >= self.action_size:
            return None
        if index < BOARD_SQUARES * BOARD_SQUARES:
            from_square = index // BOARD_SQUARES
            to_square = index % BOARD_SQUARES
            for move in board.legal_moves:
                if move.from_square != from_square or move.to_square != to_square:
                    continue
                if move.promotion in UNDERPROMOTION_PIECES:
                    continue
                return move
            return None

        rem = index - BOARD_SQUARES * BOARD_SQUARES
        from_square = rem // 9
        underpromotion = rem % 9
        dir_idx = underpromotion // 3
        piece_idx = underpromotion % 3
        dc = PROMOTION_DCS[dir_idx]
        _, from_file = divmod(from_square, 8)
        to_file = from_file + dc
        if to_file < 0 or to_file >= 8:
            return None
        step = 8 if board.turn == chess.WHITE else -8
        to_square = from_square + step + dc
        promotion = UNDERPROMOTION_PIECES[piece_idx]
        move = chess.Move(from_square, to_square, promotion=promotion)
        return move if move in board.legal_moves else None


CODECS = {
    MOVE_ENCODING_AZ_64X73: AlphaZero64x73Codec(),
    MOVE_ENCODING_SD_64X64_UP9: SourceDestinationUnderpromotionCodec(),
}


def get_move_codec(name: str) -> MoveCodec:
    try:
        return CODECS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported move encoding {name!r}") from exc
