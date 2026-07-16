"""Arena-result acceptance helpers."""

from __future__ import annotations


def arena_quality_available(metrics):
    quality = metrics.get("quality") or {}
    candidate = quality.get("candidate") or {}
    baseline = quality.get("baseline") or {}
    return (
        candidate.get("moves", 0) > 0
        and baseline.get("moves", 0) > 0
        and candidate.get("acpl") is not None
        and baseline.get("acpl") is not None
        and candidate.get("accuracy") is not None
        and baseline.get("accuracy") is not None
    )


def evaluate_arena_acceptance(
    metrics,
    min_net_wins,
    min_acpl_improvement,
    min_accuracy_improvement,
):
    quality = metrics.get("quality") or {}
    candidate = quality.get("candidate") or {}
    baseline = quality.get("baseline") or {}

    result_ok = int(metrics.get("net_wins", 0)) >= int(min_net_wins)
    quality_available = arena_quality_available(metrics)
    quality_ok = bool(
        quality_available
        and float(candidate["acpl"])
        < float(baseline["acpl"]) - float(min_acpl_improvement)
        and float(candidate["accuracy"])
        > float(baseline["accuracy"]) + float(min_accuracy_improvement)
    )

    return {
        "result_ok": bool(result_ok),
        "quality_ok": bool(quality_ok),
        "accepted": bool(result_ok and quality_ok),
        "quality_available": bool(quality_available),
        "min_net_wins": int(min_net_wins),
        "min_acpl_improvement": float(min_acpl_improvement),
        "min_accuracy_improvement": float(min_accuracy_improvement),
    }


def attach_arena_acceptance(
    metrics,
    min_net_wins,
    min_acpl_improvement,
    min_accuracy_improvement,
):
    metrics = dict(metrics)
    metrics.update(
        evaluate_arena_acceptance(
            metrics,
            min_net_wins=min_net_wins,
            min_acpl_improvement=min_acpl_improvement,
            min_accuracy_improvement=min_accuracy_improvement,
        )
    )
    return metrics
