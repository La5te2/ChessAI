from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import chess
import numpy as np


STATE_ENCODING_RESNET_PV_LINEAR_18_PLANES = "resnet_pv_linear_18_planes"
STATE_ENCODING_RESNET_PVA_GAD_SQUARE_TOKENS = "resnet_pva_gad_square_tokens"
PLANES18_CHANNELS = 18
GAD_STATE_FEATURES = 67
GAD_SQUARES = 64

PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

PIECE_TOKEN_MAP = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


@dataclass(frozen=True)
class StateCodec:
    name: str
    storage_shape: Tuple[int, ...]
    tensor_shape: Tuple[int, ...]
    storage_dtype: str
    encode_board: Callable[[chess.Board], np.ndarray]
    decode_state: Callable[[np.ndarray], np.ndarray]
    tensor_from_board: Callable[[chess.Board], np.ndarray]

    @property
    def input_channels(self) -> int:
        return int(self.tensor_shape[0])


def _planes18_tensor_uint8(board: chess.Board) -> np.ndarray:
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
    x = np.zeros((PLANES18_CHANNELS, 8, 8), dtype=np.uint8)

    for sq, piece in board.piece_map().items():
        row, col = divmod(sq, 8)
        index = PIECE_MAP[piece.piece_type]
        if piece.color == chess.BLACK:
            index += 6
        x[index, row, col] = 1

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
        _, file_index = divmod(board.ep_square, 8)
        x[17, :, file_index] = 1

    return x


def _planes18_tensor_float(board: chess.Board) -> np.ndarray:
    return _planes18_tensor_uint8(board).astype(np.float32, copy=False)


def _planes18_packbits(board: chess.Board) -> np.ndarray:
    return np.packbits(_planes18_tensor_uint8(board), axis=-1).squeeze(-1).astype(np.uint8)


def _planes18_unpackbits(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    expected_shape = (PLANES18_CHANNELS, 8)
    if packed.shape != expected_shape:
        raise ValueError(f"expected packed state shape {expected_shape}, got {packed.shape}")
    return np.unpackbits(packed[..., None], axis=-1).astype(np.float32)


def _gad_square_tokens(board: chess.Board) -> np.ndarray:
    state = np.zeros((GAD_STATE_FEATURES,), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        state[square] = PIECE_TOKEN_MAP[(piece.color, piece.piece_type)]

    state[64] = 1 if board.turn == chess.WHITE else 0
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
    state[65] = castling

    if board.ep_square is not None:
        _, file_index = divmod(board.ep_square, 8)
        state[66] = file_index + 1
    return state


def _gad_decode_tokens(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.uint8)
    expected_shape = (GAD_STATE_FEATURES,)
    if encoded.shape != expected_shape:
        raise ValueError(f"expected square-token state shape {expected_shape}, got {encoded.shape}")
    return encoded.astype(np.int64, copy=False)


def _gad_tensor_from_board(board: chess.Board) -> np.ndarray:
    return _gad_square_tokens(board).astype(np.int64, copy=False)


STATE_CODECS: Dict[str, StateCodec] = {
    STATE_ENCODING_RESNET_PV_LINEAR_18_PLANES: StateCodec(
        name=STATE_ENCODING_RESNET_PV_LINEAR_18_PLANES,
        storage_shape=(PLANES18_CHANNELS, 8),
        tensor_shape=(PLANES18_CHANNELS, 8, 8),
        storage_dtype="uint8",
        encode_board=_planes18_packbits,
        decode_state=_planes18_unpackbits,
        tensor_from_board=_planes18_tensor_float,
    ),
    STATE_ENCODING_RESNET_PVA_GAD_SQUARE_TOKENS: StateCodec(
        name=STATE_ENCODING_RESNET_PVA_GAD_SQUARE_TOKENS,
        storage_shape=(GAD_STATE_FEATURES,),
        tensor_shape=(GAD_STATE_FEATURES,),
        storage_dtype="uint8",
        encode_board=_gad_square_tokens,
        decode_state=_gad_decode_tokens,
        tensor_from_board=_gad_tensor_from_board,
    ),
}

SUPPORTED_STATE_ENCODINGS = frozenset(STATE_CODECS)


def get_state_codec(state_encoding: str) -> StateCodec:
    try:
        return STATE_CODECS[str(state_encoding)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported state encoding {state_encoding!r}; "
            f"expected one of {sorted(SUPPORTED_STATE_ENCODINGS)}"
        ) from exc
