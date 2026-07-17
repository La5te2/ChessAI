"""Arena game-result acceptance helpers."""

from __future__ import annotations


def evaluate_arena_acceptance(metrics, min_net_wins):
    result_ok = int(metrics.get("net_wins", 0)) >= int(min_net_wins)
    return {
        "result_ok": bool(result_ok),
        "accepted": bool(result_ok),
        "min_net_wins": int(min_net_wins),
    }


def attach_arena_acceptance(metrics, min_net_wins):
    metrics = dict(metrics)
    metrics.update(
        evaluate_arena_acceptance(
            metrics,
            min_net_wins=min_net_wins,
        )
    )
    return metrics
