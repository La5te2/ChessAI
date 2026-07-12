"""Dynamic teacher-generated regression set for self-learning."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List


def load_cases(path: str) -> List[Dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError(f"invalid regression data: {path}")

    cleaned = []
    for item in cases:
        if not isinstance(item, dict):
            continue
        fen = str(item.get("fen", "")).strip()
        answers = sorted({str(move).strip() for move in item.get("answers", []) if str(move).strip()})
        if fen and answers:
            cleaned.append({
                "fen": fen,
                "answers": answers,
                "best_score_cp": int(item.get("best_score_cp", 0)),
                "regret_cp": int(item.get("regret_cp", 0)),
                "teacher_weight": float(item.get("teacher_weight", 0.0)),
                "seen": int(item.get("seen", 1)),
            })
    return cleaned


def _atomic_write(path: str, cases: List[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp_{os.getpid()}"
    payload = {
        "source": "teacher_annotated_selflearning",
        "cases": cases,
    }
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def merge_cases(path: str, new_cases: Iterable[Dict], max_cases: int = 2000) -> List[Dict]:
    merged = {case["fen"]: case for case in load_cases(path)}

    for raw in new_cases:
        fen = str(raw.get("fen", "")).strip()
        answers = sorted({str(move).strip() for move in raw.get("answers", []) if str(move).strip()})
        if not fen or not answers:
            continue
        incoming = {
            "fen": fen,
            "answers": answers,
            "best_score_cp": int(raw.get("best_score_cp", 0)),
            "regret_cp": int(raw.get("regret_cp", 0)),
            "teacher_weight": float(raw.get("teacher_weight", 0.0)),
            "seen": int(raw.get("seen", 1)),
        }
        previous = merged.get(fen)
        if previous is None:
            merged[fen] = incoming
        else:
            previous["answers"] = sorted(set(previous["answers"]) | set(incoming["answers"]))
            previous["best_score_cp"] = max(previous["best_score_cp"], incoming["best_score_cp"])
            previous["regret_cp"] = max(previous["regret_cp"], incoming["regret_cp"])
            previous["teacher_weight"] = max(previous["teacher_weight"], incoming["teacher_weight"])
            previous["seen"] = int(previous.get("seen", 1)) + int(incoming.get("seen", 1))

    cases = list(merged.values())
    cases.sort(
        key=lambda item: (
            float(item.get("teacher_weight", 0.0)),
            int(item.get("regret_cp", 0)),
            int(item.get("seen", 1)),
        ),
        reverse=True,
    )
    cases = cases[: max(1, int(max_cases))]
    _atomic_write(path, cases)
    return cases
