import os

import torch

PGN_PATH = "data/games.pgn"
H5_PATH = "data/games.h5"

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/chessnet.pth"

STOCKFISH_DIR = f"{MODEL_DIR}/stockfish"
STOCKFISH_BINARY = "stockfish.exe" if os.name == "nt" else "stockfish"
STOCKFISH_PATH = f"{STOCKFISH_DIR}/{STOCKFISH_BINARY}"

SELFLEARN_DIR = "data/selflearn"
REGRESSION_PATH = f"{SELFLEARN_DIR}/regression.json"

INPUT_CHANNELS = 18
BOARD_SIZE = 8
NUM_ACTIONS = 4672

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 0.25
NUM_WORKERS = 4

CPUCT = 1.5
DEFAULT_SIMS = 100

CONFIDENCE_Z = 1.96
