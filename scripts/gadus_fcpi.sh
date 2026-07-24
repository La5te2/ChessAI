#!/usr/bin/env bash
set -euo pipefail

# Launch a production Gadus FCPI run for a 16 GiB RTX 4080 Super.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

FCPI="${FCPI:-build/gadus/fcpi}"
MODEL="${MODEL:-models/gadus/gadus.pth}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"

ITERATIONS="${ITERATIONS:-5}"
GAMES_PER_ITER="${GAMES_PER_ITER:-2000}"
GAMES_IN_FLIGHT="${GAMES_IN_FLIGHT:-512}"
MAX_PLIES="${MAX_PLIES:-240}"
POSITIONS_PER_GAME="${POSITIONS_PER_GAME:-200}"

OPENING_BOOK="${OPENING_BOOK:-data/openings.gen.bin}"
STARTPOS_FRACTION="${STARTPOS_FRACTION:-0.50}"
BOOK_PLIES="${BOOK_PLIES:-8}"
MAX_BOOK_POSITIONS="${MAX_BOOK_POSITIONS:-50000}"

INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-512}"
TARGET_RECORDS_PER_BATCH="${TARGET_RECORDS_PER_BATCH:-512}"
COUNTERFACTUAL_BUDGET="${COUNTERFACTUAL_BUDGET:-24}"
TD_LAMBDA="${TD_LAMBDA:-0.80}"

BEHAVIOR_TEMPERATURE="${BEHAVIOR_TEMPERATURE:-1.0}"
UNIFORM_MIX="${UNIFORM_MIX:-0.03}"

POLICY_WEIGHT="${POLICY_WEIGHT:-1.0}"
VALUE_WEIGHT="${VALUE_WEIGHT:-1.0}"
EPOCHS="${EPOCHS:-15}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

EVAL_GAMES="${EVAL_GAMES:-400}"
EVAL_GAMES_IN_FLIGHT="${EVAL_GAMES_IN_FLIGHT:-256}"
EVAL_MAX_PLIES="${EVAL_MAX_PLIES:-240}"
EVAL_OPENING_BOOK="${EVAL_OPENING_BOOK:-data/openings.gen.bin}"
EVAL_BOOK_PLIES="${EVAL_BOOK_PLIES:-8}"
EVAL_MAX_BOOK_POSITIONS="${EVAL_MAX_BOOK_POSITIONS:-50000}"
EVAL_SEARCH_TYPE="${EVAL_SEARCH_TYPE:-closed}"
EVAL_SIMS="${EVAL_SIMS:-0}"
EVAL_MCTS_BATCH_SIZE="${EVAL_MCTS_BATCH_SIZE:-512}"
EVAL_MOVETIME_MS="${EVAL_MOVETIME_MS:-0}"
EVAL_C_PUCT="${EVAL_C_PUCT:-0.5}"
EVAL_C_PUCT_BASE="${EVAL_C_PUCT_BASE:-19652}"
EVAL_C_PUCT_FACTOR="${EVAL_C_PUCT_FACTOR:-1.0}"
EVAL_FPU_REDUCTION="${EVAL_FPU_REDUCTION:-0.15}"
EVAL_REPETITION_POLICY_PENALTY="${EVAL_REPETITION_POLICY_PENALTY:-1.0}"
EVAL_INSTANT_MATE_FIRST="${EVAL_INSTANT_MATE_FIRST:-1}"
EVAL_MIN_NET_WINS="${EVAL_MIN_NET_WINS:-4}"

LOG_EVERY="${LOG_EVERY:-50}"
SEED="${SEED:-2026}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	printf '%s\n' \
		"usage: bash scripts/gadus_fcpi.sh" \
		"" \
		"Environment overrides:" \
		"  MODEL=models/gadus/gadus.pth ITERATIONS=5 GAMES_PER_ITER=2000" \
		"  PRECISION=bf16 BATCH_SIZE=1024 INFERENCE_BATCH_SIZE=512 EVAL_GAMES=400" \
		"" \
		"The process runs in the background. The launcher prints its run id, pid," \
		"log path, tail command, and stop command."
	exit 0
fi

if [[ ! -x "${FCPI}" ]]; then
	echo "Gadus FCPI executable is missing: ${FCPI}" >&2
	echo "Run: bash scripts/build.sh" >&2
	exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
	echo "Gadus model is missing: ${MODEL}" >&2
	exit 1
fi
if [[ -n "${OPENING_BOOK}" && ! -f "${OPENING_BOOK}" ]]; then
	echo "Opening book is missing: ${OPENING_BOOK}" >&2
	exit 1
fi
if [[ -n "${EVAL_OPENING_BOOK}" && ! -f "${EVAL_OPENING_BOOK}" ]]; then
	echo "Evaluation opening book is missing: ${EVAL_OPENING_BOOK}" >&2
	exit 1
fi
if [[ "${DEVICE}" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
	echo "CUDA was requested but nvidia-smi is unavailable." >&2
	exit 1
fi

COMMAND=(
	"${FCPI}"
	--model "${MODEL}"
	--device "${DEVICE}"
	--precision "${PRECISION}"
	--iterations "${ITERATIONS}"
	--games-per-iter "${GAMES_PER_ITER}"
	--games-in-flight "${GAMES_IN_FLIGHT}"
	--max-plies "${MAX_PLIES}"
	--positions-per-game "${POSITIONS_PER_GAME}"
	--opening-book "${OPENING_BOOK}"
	--startpos-fraction "${STARTPOS_FRACTION}"
	--book-plies "${BOOK_PLIES}"
	--max-book-positions "${MAX_BOOK_POSITIONS}"
	--inference-batch-size "${INFERENCE_BATCH_SIZE}"
	--target-records-per-batch "${TARGET_RECORDS_PER_BATCH}"
	--counterfactual-budget "${COUNTERFACTUAL_BUDGET}"
	--td-lambda "${TD_LAMBDA}"
	--behavior-temperature "${BEHAVIOR_TEMPERATURE}"
	--uniform-mix "${UNIFORM_MIX}"
	--policy-weight "${POLICY_WEIGHT}"
	--value-weight "${VALUE_WEIGHT}"
	--epochs "${EPOCHS}"
	--train-max-steps "${TRAIN_MAX_STEPS}"
	--batch-size "${BATCH_SIZE}"
	--lr "${LEARNING_RATE}"
	--weight-decay "${WEIGHT_DECAY}"
	--grad-clip "${GRAD_CLIP}"
	--eval-games "${EVAL_GAMES}"
	--eval-games-in-flight "${EVAL_GAMES_IN_FLIGHT}"
	--eval-max-plies "${EVAL_MAX_PLIES}"
	--eval-opening-book "${EVAL_OPENING_BOOK}"
	--eval-book-plies "${EVAL_BOOK_PLIES}"
	--eval-max-book-positions "${EVAL_MAX_BOOK_POSITIONS}"
	--eval-search-type "${EVAL_SEARCH_TYPE}"
	--eval-sims "${EVAL_SIMS}"
	--eval-mcts-batch-size "${EVAL_MCTS_BATCH_SIZE}"
	--eval-movetime-ms "${EVAL_MOVETIME_MS}"
	--eval-c-puct "${EVAL_C_PUCT}"
	--eval-c-puct-base "${EVAL_C_PUCT_BASE}"
	--eval-c-puct-factor "${EVAL_C_PUCT_FACTOR}"
	--eval-fpu-reduction "${EVAL_FPU_REDUCTION}"
	--eval-repetition-policy-penalty "${EVAL_REPETITION_POLICY_PENALTY}"
	--eval-instant-mate-first "${EVAL_INSTANT_MATE_FIRST}"
	--eval-min-net-wins "${EVAL_MIN_NET_WINS}"
	--log-every "${LOG_EVERY}"
	--seed "${SEED}"
)

mkdir -p data/runs
LAUNCH_LOG="data/runs/.gadus_fcpi_$(date +%Y%m%d_%H%M%S)_$$.log"

echo "Gadus FCPI launch"
echo "model=${MODEL} device=${DEVICE} precision=${PRECISION} iterations=${ITERATIONS}"
echo "self-play: games=${GAMES_PER_ITER} games_in_flight=${GAMES_IN_FLIGHT} max_plies=${MAX_PLIES}"
echo "counterfactual: budget_per_root=${COUNTERFACTUAL_BUDGET}"
echo "training: batch_size=${BATCH_SIZE} epochs=${EPOCHS} max_steps=${TRAIN_MAX_STEPS} lr=${LEARNING_RATE}"
echo "arena: games=${EVAL_GAMES} search_type=${EVAL_SEARCH_TYPE} sims=${EVAL_SIMS} min_net_wins=${EVAL_MIN_NET_WINS}"

nohup "${COMMAND[@]}" >"${LAUNCH_LOG}" 2>&1 < /dev/null &
PID=$!

RUN_ID=""
for _ in {1..100}; do
	if [[ -f "${LAUNCH_LOG}" ]]; then
		RUN_ID="$(sed -n 's/^fcpi run id: //p' "${LAUNCH_LOG}" | head -n 1)"
	fi
	if [[ -n "${RUN_ID}" ]]; then
		break
	fi
	if ! kill -0 "${PID}" 2>/dev/null; then
		echo "Gadus FCPI exited before creating a run." >&2
		cat "${LAUNCH_LOG}" >&2
		exit 1
	fi
	sleep 0.1
done

if [[ -z "${RUN_ID}" ]]; then
	echo "Gadus FCPI started, but its run id was not observed within 10 seconds." >&2
	echo "pid=${PID}" >&2
	echo "log=${LAUNCH_LOG}" >&2
	exit 1
fi

RUN_DIR="data/runs/${RUN_ID}"
LOG="${RUN_DIR}/info.log"
mv "${LAUNCH_LOG}" "${LOG}"
printf '%s\n' "${PID}" >"${RUN_DIR}/pid"

echo "Gadus FCPI launched"
echo "run_id=${RUN_ID}"
echo "pid=${PID}"
echo "log=${LOG}"
echo "tail -n 100 -f ${LOG}"
echo "kill ${PID}"
