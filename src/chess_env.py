import chess
import numpy as np
from config import INPUT_CHANNELS

PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Absolute-board 18-plane representation.

    0-5 white P,N,B,R,Q,K
    6-11 black P,N,B,R,Q,K
    12 side-to-move, all 1 if white else 0
    13 white kingside castling
    14 white queenside castling
    15 black kingside castling
    16 black queenside castling
    17 en-passant file marker
    """
    x = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.uint8)

    for sq, piece in board.piece_map().items():
        r, c = divmod(sq, 8)
        idx = PIECE_MAP[piece.piece_type]
        if piece.color == chess.BLACK:
            idx += 6
        x[idx, r, c] = 1

    if board.turn == chess.WHITE:
        x[12, :, :] = 1

    if board.has_kingside_castling_rights(chess.WHITE):
        x[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        x[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        x[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        x[16, :, :] = 1

    if board.ep_square is not None:
        _, f = divmod(board.ep_square, 8)
        x[17, :, f] = 1

    return x

def board_to_packed(board: chess.Board) -> np.ndarray:
    return np.packbits(board_to_tensor(board), axis=-1).squeeze(-1).astype(np.uint8)

def packed_to_tensor(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape != (INPUT_CHANNELS, 8):
        raise ValueError(f"expected packed shape {(INPUT_CHANNELS, 8)}, got {packed.shape}")
    return np.unpackbits(packed[..., None], axis=-1).astype(np.float32)
