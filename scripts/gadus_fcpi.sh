#!/usr/bin/env bash
set -euo pipefail

# Launch a production Gadus FCPI run for a 16 GiB RTX 4080 Super.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

FCPI="${FCPI:-build/gadus/fcpi}"
ARENA="${ARENA:-build/gadus/arena}"
MODEL="${MODEL:-models/gadus/gadus.pth}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"

ITERATIONS="${ITERATIONS:-10}"
GAMES_PER_ITER="${GAMES_PER_ITER:-2000}"
GAMES_IN_FLIGHT="${GAMES_IN_FLIGHT:-512}"
MAX_PLIES="${MAX_PLIES:-240}"

OPENING_BOOK="${OPENING_BOOK:-data/openings.gen.bin}"
STARTPOS_FRACTION="${STARTPOS_FRACTION:-0.2}"
BOOK_PLIES="${BOOK_PLIES:-8}"
MAX_BOOK_POSITIONS="${MAX_BOOK_POSITIONS:-50000}"

INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-512}"
TARGET_RECORDS_PER_BATCH="${TARGET_RECORDS_PER_BATCH:-512}"
COUNTERFACTUAL_BUDGET="${COUNTERFACTUAL_BUDGET:-24}"

BEHAVIOR_TEMPERATURE="${BEHAVIOR_TEMPERATURE:-1.0}"

EPOCHS="${EPOCHS:-30}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"

EVAL_GAMES="${EVAL_GAMES:-2000}"
EVAL_GAMES_IN_FLIGHT="${EVAL_GAMES_IN_FLIGHT:-256}"
EVAL_MAX_PLIES="${EVAL_MAX_PLIES:-240}"
EVAL_OPENING_BOOK="${EVAL_OPENING_BOOK:-data/openings.gen.bin}"
EVAL_BOOK_PLIES="${EVAL_BOOK_PLIES:-8}"
EVAL_MAX_BOOK_POSITIONS="${EVAL_MAX_BOOK_POSITIONS:-50000}"
EVAL_SEARCH_TYPE="${EVAL_SEARCH_TYPE:-closed}"
EVAL_SIMS="${EVAL_SIMS:-0}"
EVAL_MCTS_BATCH_SIZE="${EVAL_MCTS_BATCH_SIZE:-512}"
EVAL_MOVETIME_MS="${EVAL_MOVETIME_MS:-0}"
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
if [[ ! -x "${ARENA}" ]]; then
	echo "Gadus Arena executable is missing: ${ARENA}" >&2
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
if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 is required to merge the final Arena report into summary.json." >&2
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
	--opening-book "${OPENING_BOOK}"
	--startpos-fraction "${STARTPOS_FRACTION}"
	--book-plies "${BOOK_PLIES}"
	--max-book-positions "${MAX_BOOK_POSITIONS}"
	--inference-batch-size "${INFERENCE_BATCH_SIZE}"
	--target-records-per-batch "${TARGET_RECORDS_PER_BATCH}"
	--counterfactual-budget "${COUNTERFACTUAL_BUDGET}"
	--behavior-temperature "${BEHAVIOR_TEMPERATURE}"
	--epochs "${EPOCHS}"
	--train-max-steps "${TRAIN_MAX_STEPS}"
	--batch-size "${BATCH_SIZE}"
	--lr "${LEARNING_RATE}"
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
	--eval-repetition-policy-penalty "${EVAL_REPETITION_POLICY_PENALTY}"
	--eval-instant-mate-first "${EVAL_INSTANT_MATE_FIRST}"
	--eval-min-net-wins "${EVAL_MIN_NET_WINS}"
	--log-every "${LOG_EVERY}"
	--seed "${SEED}"
)

run_pipeline() {
	local fcpi_capture
	local arena_capture
	local run_id
	local run_dir
	local summary
	local pgn
	local -a children
	mkdir -p data/runs
	fcpi_capture="$(mktemp "data/runs/.gadus_fcpi_capture_XXXXXX.log")"
	arena_capture=""
	cleanup_pipeline() {
		mapfile -t children < <(jobs -pr)
		if (( ${#children[@]} > 0 )); then
			kill "${children[@]}" 2>/dev/null || true
		fi
		rm -f -- "${fcpi_capture}"
		if [[ -n "${arena_capture}" ]]; then
			rm -f -- "${arena_capture}"
		fi
	}
	abort_pipeline() {
		trap - EXIT TERM INT
		cleanup_pipeline
		exit 143
	}
	trap cleanup_pipeline EXIT
	trap abort_pipeline TERM INT

	"${COMMAND[@]}" 2>&1 | tee "${fcpi_capture}"
	run_id="$(sed -n 's/^fcpi run id: //p' "${fcpi_capture}" | head -n 1)"
	if [[ -z "${run_id}" ]]; then
		echo "FCPI completed without reporting a run id." >&2
		exit 1
	fi
	run_dir="data/runs/${run_id}"
	summary="${run_dir}/summary.json"
	pgn="${run_dir}/current_vs_initial.pgn"
	arena_capture="$(mktemp "${run_dir}/.final_arena_XXXXXX.log")"

	echo "fcpi final arena: current=models/runs/${run_id}/current.pth initial=models/runs/${run_id}/initial.pth games=${EVAL_GAMES}"
	"${ARENA}" \
		--candidate "models/runs/${run_id}/current.pth" \
		--baseline "models/runs/${run_id}/initial.pth" \
		--device "${DEVICE}" \
		--precision "${PRECISION}" \
		--games "${EVAL_GAMES}" \
		--games-in-flight "${EVAL_GAMES_IN_FLIGHT}" \
		--max-plies "${EVAL_MAX_PLIES}" \
		--opening-book "${EVAL_OPENING_BOOK}" \
		--book-plies "${EVAL_BOOK_PLIES}" \
		--max-book-positions "${EVAL_MAX_BOOK_POSITIONS}" \
		--search-type closed \
		--sims 0 \
		--mcts-batch-size "${EVAL_MCTS_BATCH_SIZE}" \
		--movetime-ms 0 \
		--repetition-policy-penalty "${EVAL_REPETITION_POLICY_PENALTY}" \
		--instant-mate-first "${EVAL_INSTANT_MATE_FIRST}" \
		--min-net-wins 0 \
		--pgn-output "${pgn}" \
		--seed "$((SEED + ITERATIONS + 1))" \
		--log-every "${LOG_EVERY}" \
		2>&1 | tee "${arena_capture}"

	python3 - "${summary}" "${arena_capture}" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
arena_log = Path(sys.argv[2]).read_text(encoding="utf-8")
marker = "arena summary:\n"
if marker not in arena_log:
	raise RuntimeError("final Arena output does not contain a JSON summary")
arena = json.loads(arena_log.rsplit(marker, 1)[1])
for key in ("accepted", "result_ok", "min_net_wins"):
	arena.pop(key, None)
arena["informational"] = True
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["final_arena"] = arena
temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, summary_path)
PY
	echo "fcpi final arena merged: ${summary}"
	cleanup_pipeline
	trap - EXIT TERM INT
}

if [[ "${1:-}" == "--pipeline" ]]; then
	run_pipeline
	exit 0
fi

mkdir -p data/runs
LAUNCH_LOG="data/runs/.gadus_fcpi_$(date +%Y%m%d_%H%M%S)_$$.log"

echo "Gadus FCPI launch"
echo "model=${MODEL} device=${DEVICE} precision=${PRECISION} iterations=${ITERATIONS}"
echo "self-play: games=${GAMES_PER_ITER} games_in_flight=${GAMES_IN_FLIGHT} max_plies=${MAX_PLIES}"
echo "counterfactual: deep_budget_per_root=${COUNTERFACTUAL_BUDGET}"
echo "training: batch_size=${BATCH_SIZE} epochs=${EPOCHS} max_steps=${TRAIN_MAX_STEPS} lr=${LEARNING_RATE}"
echo "arena: games=${EVAL_GAMES} search_type=${EVAL_SEARCH_TYPE} sims=${EVAL_SIMS} min_net_wins=${EVAL_MIN_NET_WINS}"
echo "final arena: current vs initial games=${EVAL_GAMES} search_type=closed informational=true"

nohup bash "${BASH_SOURCE[0]}" --pipeline >"${LAUNCH_LOG}" 2>&1 < /dev/null &
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
