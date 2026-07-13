#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--foreground" ]]; then
  RUN_ID="${RUN_ID:-reinforce_$(date +%Y%m%d_%H%M%S)_$$}"
  DATA_RUN_DIR="data/runs/${RUN_ID}"
  mkdir -p "${DATA_RUN_DIR}"
  nohup bash "$0" --foreground "${RUN_ID}" > "${DATA_RUN_DIR}/info.log" 2>&1 < /dev/null &
  PID="$!"
  echo "${PID}" > "${DATA_RUN_DIR}/pid"
  echo "reinforce launched"
  echo "run_id=${RUN_ID}"
  echo "pid=${PID}"
  echo "log=${DATA_RUN_DIR}/info.log"
  echo "tail -f ${DATA_RUN_DIR}/info.log"
  exit 0
fi

RUN_ID="${2:?missing run id}"

MODEL="models/chessnet.pth"
SUPERVISED_DATA="data/games.h5"
FEN_SOURCE="data/games.pgn"
UCI="models/stockfish/stockfish"
DEVICE="cuda"

ITERATIONS=1
POSITIONS_PER_ITER=500
PARALLEL=10
SOURCE_MIN_PLY=0
SOURCE_MAX_PLY=160

SAMPLE_TOPK=8
REWARD_SCALE_CP=600
ACTOR_EXPLORATION_MIX=0.05
ADVANTAGE_CLIP=1.0

UCI_DEPTH=16
UCI_MOVETIME_MS=0
UCI_MULTIPV=4
UCI_THREADS=1
UCI_HASH_MB=512

EPOCHS=25
TRAIN_MAX_STEPS=2500
BATCH_SIZE=256
TRAIN_WORKERS=4
LR=0.00003
ACTOR_WEIGHT=1.0
CRITIC_WEIGHT=0.50
ENTROPY_WEIGHT=0.01
SUPERVISED_WEIGHT=0.35
KL_WEIGHT=0.10

EVAL_GAMES=100
EVAL_SIMS=0
EVAL_WORKERS=10
EVAL_MAX_PLIES=150
EVAL_OPENING_BOOK="data/openings.gen.bin"
EVAL_MOVETIME_MS=1500
EVAL_C_PUCT=0.5
EVAL_C_PUCT_BASE=19652
EVAL_C_PUCT_FACTOR=1.0
EVAL_FPU_REDUCTION=0.15
EVAL_MCTS_TIME_FRACTION=0.90
EVAL_MATE_GUARD_PLIES=3
EVAL_MATE_GUARD_TOPK=8
EVAL_MATE_GUARD_NODES=20000
EVAL_UCI_DEPTH=16
EVAL_UCI_MULTIPV=4
EVAL_MIN_NET_WINS=4
EVAL_MIN_ACPL_IMPROVEMENT=0.0
EVAL_MIN_ACCURACY_IMPROVEMENT=0.0

LOG_EVERY=50
SEED=2026

echo "reinforce foreground start"
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "seed=${SEED}"
echo "offline labels: fen_source=${FEN_SOURCE} positions_per_iter=${POSITIONS_PER_ITER} parallel=${PARALLEL} source_ply=${SOURCE_MIN_PLY}-${SOURCE_MAX_PLY}"
echo "actor actions: topk=${SAMPLE_TOPK} include_teacher_best=true exploration_mix=${ACTOR_EXPLORATION_MIX}"
echo "reward: continuous_tanh_cp scale_cp=${REWARD_SCALE_CP} advantage_clip=${ADVANTAGE_CLIP}"
echo "teacher: uci=${UCI} depth=${UCI_DEPTH} multipv=${UCI_MULTIPV} threads=${UCI_THREADS}"
echo "train: epochs=${EPOCHS} train_max_steps=${TRAIN_MAX_STEPS} batch_size=${BATCH_SIZE} actor_weight=${ACTOR_WEIGHT} critic_weight=${CRITIC_WEIGHT} entropy_weight=${ENTROPY_WEIGHT}"
echo "eval: games=${EVAL_GAMES} sims=${EVAL_SIMS} movetime_ms=${EVAL_MOVETIME_MS} c_puct=${EVAL_C_PUCT} fpu_reduction=${EVAL_FPU_REDUCTION} mcts_time_fraction=${EVAL_MCTS_TIME_FRACTION} mate_guard=${EVAL_MATE_GUARD_PLIES}/${EVAL_MATE_GUARD_TOPK}/${EVAL_MATE_GUARD_NODES} min_net_wins=${EVAL_MIN_NET_WINS} min_acpl_improvement=${EVAL_MIN_ACPL_IMPROVEMENT} min_accuracy_improvement=${EVAL_MIN_ACCURACY_IMPROVEMENT} opening_book=${EVAL_OPENING_BOOK}"

exec python src/reinforce.py \
  --run-id "${RUN_ID}" \
  --model "${MODEL}" \
  --supervised-data "${SUPERVISED_DATA}" \
  --fen-source "${FEN_SOURCE}" \
  --uci "${UCI}" \
  --device "${DEVICE}" \
  --iterations "${ITERATIONS}" \
  --positions-per-iter "${POSITIONS_PER_ITER}" \
  --parallel "${PARALLEL}" \
  --source-min-ply "${SOURCE_MIN_PLY}" \
  --source-max-ply "${SOURCE_MAX_PLY}" \
  --sample-topk "${SAMPLE_TOPK}" \
  --reward-scale-cp "${REWARD_SCALE_CP}" \
  --actor-exploration-mix "${ACTOR_EXPLORATION_MIX}" \
  --advantage-clip "${ADVANTAGE_CLIP}" \
  --uci-depth "${UCI_DEPTH}" \
  --uci-movetime-ms "${UCI_MOVETIME_MS}" \
  --uci-multipv "${UCI_MULTIPV}" \
  --uci-threads "${UCI_THREADS}" \
  --uci-hash-mb "${UCI_HASH_MB}" \
  --epochs "${EPOCHS}" \
  --train-max-steps "${TRAIN_MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-workers "${TRAIN_WORKERS}" \
  --lr "${LR}" \
  --actor-weight "${ACTOR_WEIGHT}" \
  --critic-weight "${CRITIC_WEIGHT}" \
  --entropy-weight "${ENTROPY_WEIGHT}" \
  --supervised-weight "${SUPERVISED_WEIGHT}" \
  --kl-weight "${KL_WEIGHT}" \
  --eval-games "${EVAL_GAMES}" \
  --eval-sims "${EVAL_SIMS}" \
  --eval-workers "${EVAL_WORKERS}" \
  --eval-max-plies "${EVAL_MAX_PLIES}" \
  --eval-opening-book "${EVAL_OPENING_BOOK}" \
  --eval-movetime-ms "${EVAL_MOVETIME_MS}" \
  --eval-c-puct "${EVAL_C_PUCT}" \
  --eval-c-puct-base "${EVAL_C_PUCT_BASE}" \
  --eval-c-puct-factor "${EVAL_C_PUCT_FACTOR}" \
  --eval-fpu-reduction "${EVAL_FPU_REDUCTION}" \
  --eval-mcts-time-fraction "${EVAL_MCTS_TIME_FRACTION}" \
  --eval-mate-guard-plies "${EVAL_MATE_GUARD_PLIES}" \
  --eval-mate-guard-topk "${EVAL_MATE_GUARD_TOPK}" \
  --eval-mate-guard-nodes "${EVAL_MATE_GUARD_NODES}" \
  --eval-uci-depth "${EVAL_UCI_DEPTH}" \
  --eval-uci-multipv "${EVAL_UCI_MULTIPV}" \
  --eval-min-net-wins "${EVAL_MIN_NET_WINS}" \
  --eval-min-acpl-improvement "${EVAL_MIN_ACPL_IMPROVEMENT}" \
  --eval-min-accuracy-improvement "${EVAL_MIN_ACCURACY_IMPROVEMENT}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}"
