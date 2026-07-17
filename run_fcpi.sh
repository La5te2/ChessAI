#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--foreground" ]]; then
  RUN_ID="fcpi_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
  DATA_RUN_DIR="data/runs/${RUN_ID}"
  mkdir -p "${DATA_RUN_DIR}"
  nohup bash "$0" --foreground "${RUN_ID}" > "${DATA_RUN_DIR}/info.log" 2>&1 < /dev/null &
  PID="$!"
  echo "${PID}" > "${DATA_RUN_DIR}/pid"
  echo "fcpi launched"
  echo "run_id=${RUN_ID}"
  echo "pid=${PID}"
  echo "log=${DATA_RUN_DIR}/info.log"
  echo "tail -f ${DATA_RUN_DIR}/info.log"
  exit 0
fi

RUN_ID="${2:?missing run id}"
export FCPI_RUN_ID="${RUN_ID}"

MODEL="models/chessnet.pth"
DEVICE="cuda"

ITERATIONS=5
GAMES_PER_ITER=1000
GAMES_IN_FLIGHT=100
MAX_PLIES=240
POSITIONS_PER_GAME=100
OPENING_BOOK="data/openings.gen.bin"
BOOK_PLIES=8
MAX_BOOK_POSITIONS=50000
INFERENCE_BATCH_SIZE=64
TARGET_RECORDS_PER_BATCH=256
VALIDATION_FRACTION=0.10

EPOCHS=30
TRAIN_MAX_STEPS=2500
BATCH_SIZE=256
TRAIN_WORKERS=4
LR=0.00001
WEIGHT_DECAY=0.0001
GRAD_CLIP=1.0

EVAL_GAMES=200
EVAL_SIMS=0
EVAL_WORKERS=10
EVAL_MAX_PLIES=240
EVAL_OPENING_BOOK="data/openings.gen.bin"
EVAL_BOOK_PLIES=8
EVAL_MAX_BOOK_POSITIONS=50000
EVAL_MCTS_BATCH_SIZE=64
EVAL_MOVETIME_MS=0
EVAL_SEARCH_TYPE=closed
EVAL_MIN_NET_WINS=4

LOG_EVERY=50
SEED=2026

ARCH_TYPE="$(python - "${MODEL}" <<'PY'
import sys
sys.path.insert(0, "src")
from model import load_model
model = load_model(sys.argv[1], device="cpu")
print(model.arch()["type"])
PY
)"

ARCH_EXTRA_ARGS=()
case "${ARCH_TYPE}" in
  resnet_pv_linear)
    TD_LAMBDA=0.80
    COUNTERFACTUAL_TOPK=6
    COUNTERFACTUAL_MIN_PLIES=2
    COUNTERFACTUAL_MAX_PLIES=6
    COUNTERFACTUAL_TARGET_AVERAGE_PLIES=4.0
    COUNTERFACTUAL_LAMBDA=0.80
    BEHAVIOR_TEMPERATURE=0.80
    UNIFORM_MIX=0.02
    POLICY_TEMPERATURE=0.25
    PRIOR_POWER=1.0
    PLAYED_RETURN_WEIGHT=0.50
    POLICY_WEIGHT=1.0
    VALUE_WEIGHT=1.0
    KL_WEIGHT=0.05
    ENTROPY_WEIGHT=0.001
    ;;
  resnet_pva_gad)
    TD_LAMBDA=0.85
    COUNTERFACTUAL_TOPK=8
    COUNTERFACTUAL_MIN_PLIES=2
    COUNTERFACTUAL_MAX_PLIES=6
    COUNTERFACTUAL_TARGET_AVERAGE_PLIES=4.0
    COUNTERFACTUAL_LAMBDA=0.85
    BEHAVIOR_TEMPERATURE=0.85
    UNIFORM_MIX=0.02
    POLICY_TEMPERATURE=0.25
    PRIOR_POWER=1.0
    PLAYED_RETURN_WEIGHT=0.50
    POLICY_WEIGHT=1.0
    VALUE_WEIGHT=1.0
    KL_WEIGHT=0.05
    ENTROPY_WEIGHT=0.001
    BEHAVIOR_ADVANTAGE_WEIGHT=0.50
    SUCCESSOR_WEIGHT=0.75
    ADVANTAGE_WEIGHT=0.50
    ARCH_EXTRA_ARGS=(
      --behavior-advantage-weight "${BEHAVIOR_ADVANTAGE_WEIGHT}"
      --successor-weight "${SUCCESSOR_WEIGHT}"
      --advantage-weight "${ADVANTAGE_WEIGHT}"
    )
    ;;
  *)
    echo "unsupported FCPI architecture: ${ARCH_TYPE}" >&2
    exit 1
    ;;
esac

ARCH_ARGS=(
  --td-lambda "${TD_LAMBDA}"
  --counterfactual-topk "${COUNTERFACTUAL_TOPK}"
  --counterfactual-min-plies "${COUNTERFACTUAL_MIN_PLIES}"
  --counterfactual-max-plies "${COUNTERFACTUAL_MAX_PLIES}"
  --counterfactual-target-average-plies "${COUNTERFACTUAL_TARGET_AVERAGE_PLIES}"
  --counterfactual-lambda "${COUNTERFACTUAL_LAMBDA}"
  --behavior-temperature "${BEHAVIOR_TEMPERATURE}"
  --uniform-mix "${UNIFORM_MIX}"
  --policy-temperature "${POLICY_TEMPERATURE}"
  --prior-power "${PRIOR_POWER}"
  --played-return-weight "${PLAYED_RETURN_WEIGHT}"
  --policy-weight "${POLICY_WEIGHT}"
  --value-weight "${VALUE_WEIGHT}"
  --kl-weight "${KL_WEIGHT}"
  --entropy-weight "${ENTROPY_WEIGHT}"
  "${ARCH_EXTRA_ARGS[@]}"
)

echo "fcpi foreground start"
echo "run_id=${RUN_ID}"
echo "model=${MODEL}"
echo "arch_type=${ARCH_TYPE}"
echo "device=${DEVICE}"
echo "self-play: games=${GAMES_PER_ITER} in_flight=${GAMES_IN_FLIGHT} max_plies=${MAX_PLIES} positions_per_game=${POSITIONS_PER_GAME} opening_book=${OPENING_BOOK}"
echo "counterfactual: topk=${COUNTERFACTUAL_TOPK} plies=${COUNTERFACTUAL_MIN_PLIES}-${COUNTERFACTUAL_MAX_PLIES} target_average_plies=${COUNTERFACTUAL_TARGET_AVERAGE_PLIES} lambda=${COUNTERFACTUAL_LAMBDA}"
echo "train: epochs=${EPOCHS} max_steps=${TRAIN_MAX_STEPS} batch_size=${BATCH_SIZE} lr=${LR}"
echo "eval: games=${EVAL_GAMES} search_type=${EVAL_SEARCH_TYPE} sims=${EVAL_SIMS} min_net_wins=${EVAL_MIN_NET_WINS}"

exec python src/fcpi.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --iterations "${ITERATIONS}" \
  --games-per-iter "${GAMES_PER_ITER}" \
  --games-in-flight "${GAMES_IN_FLIGHT}" \
  --max-plies "${MAX_PLIES}" \
  --positions-per-game "${POSITIONS_PER_GAME}" \
  --opening-book "${OPENING_BOOK}" \
  --book-plies "${BOOK_PLIES}" \
  --max-book-positions "${MAX_BOOK_POSITIONS}" \
  --inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --target-records-per-batch "${TARGET_RECORDS_PER_BATCH}" \
  --validation-fraction "${VALIDATION_FRACTION}" \
  --epochs "${EPOCHS}" \
  --train-max-steps "${TRAIN_MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-workers "${TRAIN_WORKERS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip "${GRAD_CLIP}" \
  --eval-games "${EVAL_GAMES}" \
  --eval-sims "${EVAL_SIMS}" \
  --eval-workers "${EVAL_WORKERS}" \
  --eval-max-plies "${EVAL_MAX_PLIES}" \
  --eval-opening-book "${EVAL_OPENING_BOOK}" \
  --eval-book-plies "${EVAL_BOOK_PLIES}" \
  --eval-max-book-positions "${EVAL_MAX_BOOK_POSITIONS}" \
  --eval-mcts-batch-size "${EVAL_MCTS_BATCH_SIZE}" \
  --eval-movetime-ms "${EVAL_MOVETIME_MS}" \
  --eval-search-type "${EVAL_SEARCH_TYPE}" \
  --eval-min-net-wins "${EVAL_MIN_NET_WINS}" \
  --log-every "${LOG_EVERY}" \
  --seed "${SEED}" \
  "${ARCH_ARGS[@]}"
