import os

import torch

PGN_PATH = "data/games.pgn"
H5_PATH = "data/games.h5"

MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/chessnet.pth"

UCI_DIR = f"{MODEL_DIR}/stockfish"
UCI_BINARY = "stockfish.exe" if os.name == "nt" else "stockfish"
UCI_PATH = f"{UCI_DIR}/{UCI_BINARY}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 0.25
NUM_WORKERS = 4

CPUCT = 0.5
DEFAULT_SIMS = 100

CONFIDENCE_Z = 1.96
