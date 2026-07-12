#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-}"
if [[ "${MODE}" == "--foreground" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "usage: bash run_selflearn.sh --foreground <run-id>" >&2
    exit 2
  fi
  SELFLEARN_RUN_ID="$2"
elif [[ -n "${MODE}" ]]; then
  echo "usage: bash run_selflearn.sh" >&2
  echo "       bash run_selflearn.sh --foreground <run-id>" >&2
  exit 2
else
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    RUN_SUFFIX="$(cut -c1-8 /proc/sys/kernel/random/uuid)"
  else
    RUN_SUFFIX="${RANDOM}${RANDOM}"
  fi
  SELFLEARN_RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)_$$_${RUN_SUFFIX}}"
  LOG_DIR="data/runs/${SELFLEARN_RUN_ID}"
  LOG_PATH="${LOG_DIR}/info.log"
  PID_PATH="${LOG_DIR}/pid"
  mkdir -p "${LOG_DIR}"
  echo "selflearn background start"
  echo "selflearn run id: ${SELFLEARN_RUN_ID}"
  echo "selflearn log: ${LOG_PATH}"
  nohup bash "$0" --foreground "${SELFLEARN_RUN_ID}" > "${LOG_PATH}" 2>&1 < /dev/null &
  SELFLEARN_PID=$!
  echo "${SELFLEARN_PID}" > "${PID_PATH}"
  disown "${SELFLEARN_PID}" 2>/dev/null || true
  echo "selflearn pid: ${SELFLEARN_PID}"
  echo "watch log: tail -f ${LOG_PATH}"
  echo "stop run: kill \$(cat ${PID_PATH})"
  exit 0
fi

MODEL="models/chessnet.pth"
SUPERVISED_DATA="data/games.h5"
UCI="models/stockfish/stockfish"
DEVICE="cuda"

ITERATIONS=5
GAMES_PER_ITER=200
PARALLEL=10
MAX_PLIES=150
OPENING_BOOK="data/openings.gen.bin"
BOOK_PLIES=8
MAX_BOOK_POSITIONS="${GAMES_PER_ITER}"

SIMS=64
MCTS_BATCH_SIZE=64
MOVETIME_MS=1000
C_PUCT=0.5
ALPHA_BETA_DEPTH=3
ALPHA_BETA_TOPK=4
ALPHA_BETA_NODES=20000
ALPHA_BETA_QUIESCENCE=2
ALPHA_BETA_MARGIN=0.02
ALPHA_BETA_TIME_FRACTION=0.20
MATE_GUARD_PLIES=3
Q_TIEBREAK_MIN_VISITS=32
Q_TIEBREAK_P_RATIO=0.9
Q_TIEBREAK_VISIT_RATIO=0.9
Q_TIEBREAK_MARGIN=0.03

UCI_DEPTH=12
UCI_MULTIPV=4
UCI_THREADS=1
UCI_HASH_MB=512
TEACHER_EVERY=1
TEACHER_SAMPLE_RATE=1
TEACHER_LABEL_TOPK=4
TEACHER_LABEL_MIN_WEIGHT=0.20
TEACHER_VETO_REGRET_CP=100
TEACHER_VETO_MIN_WEIGHT=0.05

EPOCHS_PER_ITER=60
TRAIN_MAX_STEPS=2500
BATCH_SIZE=256
TRAIN_WORKERS=4
REPLAY_WINDOW=5
LR=2e-5
SUPERVISED_WEIGHT=0.50
KL_WEIGHT=0.20
MAX_SUPERVISED_LOSS_INCREASE=0.25
MAX_TARGET_CE_INCREASE=0.02

REGRESSION_SIMS=200
REGRESSION_MOVETIME_MS=1000
MIN_REGRESSION_ACCURACY=0.0
MAX_REGRESSION_DROP=0

EVAL_GAMES=100
EVAL_SIMS=64
EVAL_MAX_PLIES=150
EVAL_MCTS_BATCH_SIZE=64
EVAL_MOVETIME_MS=1000
EVAL_C_PUCT=0.5
EVAL_OPENING_BOOK="data/openings.gen.bin"
EVAL_BOOK_PLIES=8
EVAL_MAX_BOOK_POSITIONS=500
EVAL_MIN_NET_WINS=0
EVAL_MIN_ACPL_IMPROVEMENT=0.0
EVAL_MIN_ACCURACY_IMPROVEMENT=0.0
EVAL_UCI_DEPTH=16
EVAL_UCI_MULTIPV=4
LOG_EVERY=50

echo "selflearn start"
echo "selflearn model: ${MODEL}"
echo "selflearn data: games_per_iter=${GAMES_PER_ITER} iterations=${ITERATIONS} opening_book=${OPENING_BOOK} max_book_positions=${MAX_BOOK_POSITIONS}"
echo "selflearn workers: parallel=${PARALLEL} train_workers=${TRAIN_WORKERS} device=${DEVICE}"
echo "selflearn search: sims=${SIMS} movetime_ms=${MOVETIME_MS} c_puct=${C_PUCT} alpha_beta=${ALPHA_BETA_DEPTH}/${ALPHA_BETA_TOPK}/${ALPHA_BETA_NODES}/${ALPHA_BETA_QUIESCENCE} ab_time_fraction=${ALPHA_BETA_TIME_FRACTION} mate_guard_plies=${MATE_GUARD_PLIES}"
echo "selflearn move selection: deterministic top1"
echo "selflearn teacher: uci_depth=${UCI_DEPTH} uci_multipv=${UCI_MULTIPV} uci_threads=${UCI_THREADS} label_topk=${TEACHER_LABEL_TOPK} label_min_weight=${TEACHER_LABEL_MIN_WEIGHT} veto_regret_cp=${TEACHER_VETO_REGRET_CP} veto_min_weight=${TEACHER_VETO_MIN_WEIGHT}"
echo "selflearn train: epochs_per_iter=${EPOCHS_PER_ITER} train_max_steps=${TRAIN_MAX_STEPS} batch_size=${BATCH_SIZE} replay_window=${REPLAY_WINDOW} lr=${LR} supervised_weight=${SUPERVISED_WEIGHT} kl_weight=${KL_WEIGHT}"
echo "selflearn validation: max_supervised_loss_increase=${MAX_SUPERVISED_LOSS_INCREASE} max_target_ce_increase=${MAX_TARGET_CE_INCREASE}"
echo "selflearn eval: games=${EVAL_GAMES} sims=${EVAL_SIMS} movetime_ms=${EVAL_MOVETIME_MS} min_net_wins=${EVAL_MIN_NET_WINS} min_acpl_improvement=${EVAL_MIN_ACPL_IMPROVEMENT} min_accuracy_improvement=${EVAL_MIN_ACCURACY_IMPROVEMENT}"

exec python src/selflearn.py \
  --model "${MODEL}" \
  --supervised-data "${SUPERVISED_DATA}" \
  --uci "${UCI}" \
  --device "${DEVICE}" \
  --run-id "${SELFLEARN_RUN_ID}" \
  --iterations "${ITERATIONS}" \
  --games-per-iter "${GAMES_PER_ITER}" \
  --parallel "${PARALLEL}" \
  --max-plies "${MAX_PLIES}" \
  --opening-book "${OPENING_BOOK}" \
  --book-plies "${BOOK_PLIES}" \
  --max-book-positions "${MAX_BOOK_POSITIONS}" \
  --sims "${SIMS}" \
  --mcts-batch-size "${MCTS_BATCH_SIZE}" \
  --movetime-ms "${MOVETIME_MS}" \
  --c-puct "${C_PUCT}" \
  --alpha-beta-depth "${ALPHA_BETA_DEPTH}" \
  --alpha-beta-topk "${ALPHA_BETA_TOPK}" \
  --alpha-beta-nodes "${ALPHA_BETA_NODES}" \
  --alpha-beta-quiescence "${ALPHA_BETA_QUIESCENCE}" \
  --alpha-beta-margin "${ALPHA_BETA_MARGIN}" \
  --alpha-beta-time-fraction "${ALPHA_BETA_TIME_FRACTION}" \
  --mate-guard-plies "${MATE_GUARD_PLIES}" \
  --q-tiebreak-min-visits "${Q_TIEBREAK_MIN_VISITS}" \
  --q-tiebreak-p-ratio "${Q_TIEBREAK_P_RATIO}" \
  --q-tiebreak-visit-ratio "${Q_TIEBREAK_VISIT_RATIO}" \
  --q-tiebreak-margin "${Q_TIEBREAK_MARGIN}" \
  --uci-depth "${UCI_DEPTH}" \
  --uci-multipv "${UCI_MULTIPV}" \
  --uci-threads "${UCI_THREADS}" \
  --uci-hash-mb "${UCI_HASH_MB}" \
  --teacher-start-ply 0 \
  --teacher-every "${TEACHER_EVERY}" \
  --teacher-sample-rate "${TEACHER_SAMPLE_RATE}" \
  --teacher-label-topk "${TEACHER_LABEL_TOPK}" \
  --teacher-label-min-weight "${TEACHER_LABEL_MIN_WEIGHT}" \
  --teacher-veto-regret-cp "${TEACHER_VETO_REGRET_CP}" \
  --teacher-veto-min-weight "${TEACHER_VETO_MIN_WEIGHT}" \
  --epochs-per-iter "${EPOCHS_PER_ITER}" \
  --train-max-steps "${TRAIN_MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --train-workers "${TRAIN_WORKERS}" \
  --replay-window "${REPLAY_WINDOW}" \
  --lr "${LR}" \
  --supervised-weight "${SUPERVISED_WEIGHT}" \
  --kl-weight "${KL_WEIGHT}" \
  --max-supervised-loss-increase "${MAX_SUPERVISED_LOSS_INCREASE}" \
  --max-target-ce-increase "${MAX_TARGET_CE_INCREASE}" \
  --regression-sims "${REGRESSION_SIMS}" \
  --regression-movetime-ms "${REGRESSION_MOVETIME_MS}" \
  --min-regression-accuracy "${MIN_REGRESSION_ACCURACY}" \
  --max-regression-drop "${MAX_REGRESSION_DROP}" \
  --eval-games "${EVAL_GAMES}" \
  --eval-sims "${EVAL_SIMS}" \
  --eval-max-plies "${EVAL_MAX_PLIES}" \
  --eval-mcts-batch-size "${EVAL_MCTS_BATCH_SIZE}" \
  --eval-movetime-ms "${EVAL_MOVETIME_MS}" \
  --eval-c-puct "${EVAL_C_PUCT}" \
  --eval-alpha-beta-depth "${ALPHA_BETA_DEPTH}" \
  --eval-alpha-beta-topk "${ALPHA_BETA_TOPK}" \
  --eval-alpha-beta-nodes "${ALPHA_BETA_NODES}" \
  --eval-alpha-beta-quiescence "${ALPHA_BETA_QUIESCENCE}" \
  --eval-alpha-beta-margin "${ALPHA_BETA_MARGIN}" \
  --eval-alpha-beta-time-fraction "${ALPHA_BETA_TIME_FRACTION}" \
  --eval-mate-guard-plies "${MATE_GUARD_PLIES}" \
  --eval-q-tiebreak-min-visits "${Q_TIEBREAK_MIN_VISITS}" \
  --eval-q-tiebreak-p-ratio "${Q_TIEBREAK_P_RATIO}" \
  --eval-q-tiebreak-visit-ratio "${Q_TIEBREAK_VISIT_RATIO}" \
  --eval-q-tiebreak-margin "${Q_TIEBREAK_MARGIN}" \
  --eval-opening-book "${EVAL_OPENING_BOOK}" \
  --eval-book-plies "${EVAL_BOOK_PLIES}" \
  --eval-max-book-positions "${EVAL_MAX_BOOK_POSITIONS}" \
  --eval-min-net-wins "${EVAL_MIN_NET_WINS}" \
  --eval-min-acpl-improvement "${EVAL_MIN_ACPL_IMPROVEMENT}" \
  --eval-min-accuracy-improvement "${EVAL_MIN_ACCURACY_IMPROVEMENT}" \
  --eval-uci-depth "${EVAL_UCI_DEPTH}" \
  --eval-uci-multipv "${EVAL_UCI_MULTIPV}" \
  --log-every "${LOG_EVERY}"
