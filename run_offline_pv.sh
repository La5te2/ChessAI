#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--foreground" ]]; then
  RUN_ID="offline_pv_$(date +%Y%m%d_%H%M%S)_$$"
  DATA_RUN_DIR="data/runs/${RUN_ID}"
  mkdir -p "${DATA_RUN_DIR}"
  nohup bash "$0" --foreground "${RUN_ID}" > "${DATA_RUN_DIR}/info.log" 2>&1 < /dev/null &
  PID="$!"
  echo "${PID}" > "${DATA_RUN_DIR}/pid"
  echo "offline-pv launched"
  echo "run_id=${RUN_ID}"
  echo "pid=${PID}"
  echo "log=${DATA_RUN_DIR}/info.log"
  echo "tail -f ${DATA_RUN_DIR}/info.log"
  exit 0
fi

RUN_ID="${2:?missing run id}"
export OFFLINE_PV_RUN_ID="${RUN_ID}"

MODEL="models/chessnet.pth"
FEN_SOURCE="data/games.pgn"
UCI="models/stockfish/stockfish"
DEVICE="cuda"

ITERATIONS=5
POSITIONS_PER_ITER=10000
PARALLEL=10
SOURCE_MIN_PLY=0
SOURCE_MAX_PLY=160
ARENA_REPLAY_WINDOW=1
ARENA_REPLAY_POSITIONS=-1
ARENA_REPLAY_POSITIONS_PER_ITER=10000

SAMPLE_TOPK=6
REWARD_SCALE_CP=600
TEACHER_POLICY_WEIGHT=0.10
TEACHER_RANK_WEIGHT=0.10
TEACHER_RANK_MIN_REWARD_GAP=0.0
TEACHER_VALUE_WEIGHT=0.50
TEACHER_POLICY_TEMP_CP=150
ACTOR_EXPLORATION_MIX=0.05
ADVANTAGE_CLIP=1.0

UCI_DEPTH=16
UCI_MOVETIME_MS=0
UCI_MULTIPV=1
UCI_THREADS=1
UCI_HASH_MB=512

EPOCHS=30
TRAIN_MAX_STEPS=2000
BATCH_SIZE=256
TRAIN_WORKERS=4
LR=0.00003
ACTOR_WEIGHT=1.0
CRITIC_WEIGHT=0.50
ENTROPY_WEIGHT=0.003
KL_WEIGHT=0.05

VALIDATION_SOURCE="data/games.pgn"
VALIDATION_POSITIONS=1000
VALIDATION_OFFSET="$((POSITIONS_PER_ITER * ITERATIONS))"
VALIDATION_MIN_PLY=0
VALIDATION_MAX_PLY=160
VALIDATION_TOPK=4
VALIDATION_WORKERS=10
VALIDATION_UCI_DEPTH=16
VALIDATION_UCI_MOVETIME_MS=0
VALIDATION_UCI_MULTIPV=1
VALIDATION_UCI_THREADS=1
VALIDATION_UCI_HASH_MB=512
VALIDATION_MAX_TOP1_REGRET_REGRESSION_CP=20
VALIDATION_MAX_COMPOSITE_REGRET_REGRESSION_CP=20
VALIDATION_MAX_VALUE_MAE_REGRESSION=0.02
VALIDATION_MAX_VALUE_RMSE_REGRESSION=0.02
VALIDATION_MIN_VALUE_SIGN_ACC_DELTA=-0.02
VALIDATION_MIN_TEACHER_BEST_TOPK_DELTA=-0.02

EVAL_GAMES=200
EVAL_SIMS=0
EVAL_WORKERS=10
EVAL_MAX_PLIES=160
EVAL_OPENING_BOOK="data/openings.gen.bin"
EVAL_MOVETIME_MS=0
EVAL_SEARCH_TYPE=closed
EVAL_C_PUCT=0.5
EVAL_C_PUCT_BASE=19652
EVAL_C_PUCT_FACTOR=1.0
EVAL_FPU_REDUCTION=0.15
EVAL_UCI_DEPTH=16
EVAL_UCI_MULTIPV=1
EVAL_MIN_NET_WINS=0
EVAL_MIN_ACPL_IMPROVEMENT=0.0
EVAL_MIN_ACCURACY_IMPROVEMENT=0.0

LOG_EVERY=50
SEED=2026

echo "offline-pv foreground start"
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "seed=${SEED}"
echo "offline labels: fen_source=${FEN_SOURCE} positions_per_iter=${POSITIONS_PER_ITER} parallel=${PARALLEL} source_ply=${SOURCE_MIN_PLY}-${SOURCE_MAX_PLY}"
echo "arena replay: window=${ARENA_REPLAY_WINDOW} positions=${ARENA_REPLAY_POSITIONS} positions_per_iter=${ARENA_REPLAY_POSITIONS_PER_ITER}"
echo "actor actions: topk=${SAMPLE_TOPK} include_teacher_best=true exploration_mix=${ACTOR_EXPLORATION_MIX}"
echo "reward: continuous_tanh_cp scale_cp=${REWARD_SCALE_CP} advantage_clip=${ADVANTAGE_CLIP}"
echo "offline-pv targets: policy_weight=${TEACHER_POLICY_WEIGHT} rank_weight=${TEACHER_RANK_WEIGHT} value_weight=${TEACHER_VALUE_WEIGHT} policy_temp_cp=${TEACHER_POLICY_TEMP_CP}"
echo "teacher: uci=${UCI} depth=${UCI_DEPTH} multipv=${UCI_MULTIPV} threads=${UCI_THREADS}"
echo "train: epochs=${EPOCHS} train_max_steps=${TRAIN_MAX_STEPS} batch_size=${BATCH_SIZE} actor_weight=${ACTOR_WEIGHT} critic_weight=${CRITIC_WEIGHT} entropy_weight=${ENTROPY_WEIGHT}"
echo "teacher validation: source=${VALIDATION_SOURCE} positions=${VALIDATION_POSITIONS} offset=${VALIDATION_OFFSET} topk=${VALIDATION_TOPK} uci_depth=${VALIDATION_UCI_DEPTH} uci_multipv=${VALIDATION_UCI_MULTIPV} top1_tol_cp=${VALIDATION_MAX_TOP1_REGRET_REGRESSION_CP} composite_tol_cp=${VALIDATION_MAX_COMPOSITE_REGRET_REGRESSION_CP}"
echo "eval: games=${EVAL_GAMES} search_type=${EVAL_SEARCH_TYPE} sims=${EVAL_SIMS} movetime_ms=${EVAL_MOVETIME_MS} c_puct=${EVAL_C_PUCT} fpu_reduction=${EVAL_FPU_REDUCTION} min_net_wins=${EVAL_MIN_NET_WINS} min_acpl_improvement=${EVAL_MIN_ACPL_IMPROVEMENT} min_accuracy_improvement=${EVAL_MIN_ACCURACY_IMPROVEMENT} opening_book=${EVAL_OPENING_BOOK}"

exec python src/offline_pv.py \
  --model "${MODEL}" \
  --fen-source "${FEN_SOURCE}" \
  --uci "${UCI}" \
  --device "${DEVICE}" \
  --iterations "${ITERATIONS}" \
  --positions-per-iter "${POSITIONS_PER_ITER}" \
  --parallel "${PARALLEL}" \
  --source-min-ply "${SOURCE_MIN_PLY}" \
  --source-max-ply "${SOURCE_MAX_PLY}" \
  --arena-replay-window "${ARENA_REPLAY_WINDOW}" \
  --arena-replay-positions "${ARENA_REPLAY_POSITIONS}" \
  --arena-replay-positions-per-iter "${ARENA_REPLAY_POSITIONS_PER_ITER}" \
  --sample-topk "${SAMPLE_TOPK}" \
  --reward-scale-cp "${REWARD_SCALE_CP}" \
  --teacher-policy-weight "${TEACHER_POLICY_WEIGHT}" \
  --teacher-rank-weight "${TEACHER_RANK_WEIGHT}" \
  --teacher-rank-min-reward-gap "${TEACHER_RANK_MIN_REWARD_GAP}" \
  --teacher-value-weight "${TEACHER_VALUE_WEIGHT}" \
  --teacher-policy-temp-cp "${TEACHER_POLICY_TEMP_CP}" \
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
  --kl-weight "${KL_WEIGHT}" \
  --validation-source "${VALIDATION_SOURCE}" \
  --validation-positions "${VALIDATION_POSITIONS}" \
  --validation-offset "${VALIDATION_OFFSET}" \
  --validation-min-ply "${VALIDATION_MIN_PLY}" \
  --validation-max-ply "${VALIDATION_MAX_PLY}" \
  --validation-topk "${VALIDATION_TOPK}" \
  --validation-workers "${VALIDATION_WORKERS}" \
  --validation-uci-depth "${VALIDATION_UCI_DEPTH}" \
  --validation-uci-movetime-ms "${VALIDATION_UCI_MOVETIME_MS}" \
  --validation-uci-multipv "${VALIDATION_UCI_MULTIPV}" \
  --validation-uci-threads "${VALIDATION_UCI_THREADS}" \
  --validation-uci-hash-mb "${VALIDATION_UCI_HASH_MB}" \
  --validation-max-top1-regret-regression-cp "${VALIDATION_MAX_TOP1_REGRET_REGRESSION_CP}" \
  --validation-max-composite-regret-regression-cp "${VALIDATION_MAX_COMPOSITE_REGRET_REGRESSION_CP}" \
  --validation-max-value-mae-regression "${VALIDATION_MAX_VALUE_MAE_REGRESSION}" \
  --validation-max-value-rmse-regression "${VALIDATION_MAX_VALUE_RMSE_REGRESSION}" \
  --validation-min-value-sign-acc-delta "${VALIDATION_MIN_VALUE_SIGN_ACC_DELTA}" \
  --validation-min-teacher-best-topk-delta "${VALIDATION_MIN_TEACHER_BEST_TOPK_DELTA}" \
  --eval-games "${EVAL_GAMES}" \
  --eval-sims "${EVAL_SIMS}" \
  --eval-workers "${EVAL_WORKERS}" \
  --eval-max-plies "${EVAL_MAX_PLIES}" \
  --eval-opening-book "${EVAL_OPENING_BOOK}" \
  --eval-movetime-ms "${EVAL_MOVETIME_MS}" \
  --eval-search-type "${EVAL_SEARCH_TYPE}" \
  --eval-c-puct "${EVAL_C_PUCT}" \
  --eval-c-puct-base "${EVAL_C_PUCT_BASE}" \
  --eval-c-puct-factor "${EVAL_C_PUCT_FACTOR}" \
  --eval-fpu-reduction "${EVAL_FPU_REDUCTION}" \
  --eval-uci-depth "${EVAL_UCI_DEPTH}" \
  --eval-uci-multipv "${EVAL_UCI_MULTIPV}" \
  --eval-min-net-wins "${EVAL_MIN_NET_WINS}" \
  --eval-min-acpl-improvement "${EVAL_MIN_ACPL_IMPROVEMENT}" \
  --eval-min-accuracy-improvement "${EVAL_MIN_ACCURACY_IMPROVEMENT}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}"
