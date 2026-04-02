"""
poker/decision_log.py — Structured per-decision JSON-lines logger.

Writes one JSON object per line to logs/{bot_name}_decisions.log.
Only active when bot.LOG_DECISIONS is True.

Enable via:
    python bot.py --name Alice --log-decisions
    POKER_DECISION_LOG=1 python bot.py --name Alice
"""

import json
import os
from typing import Any

_LOG_DIR = "logs"

# Module-level file handle — opened lazily on first write.
_log_fh = None
_hand_counter: int = 0


def _ensure_open() -> bool:
    """Open the log file if logging is enabled and not yet open. Return True if ready."""
    import bot
    global _log_fh
    if not bot.LOG_DECISIONS:
        return False
    if _log_fh is None:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fname = os.path.join(_LOG_DIR, f"{bot.BOT_NAME}_decisions.log")
        _log_fh = open(fname, "a", encoding="utf-8")
    return True


def increment_hand() -> None:
    """Call on each hand_start to bump the hand counter."""
    global _hand_counter
    _hand_counter += 1


def reset() -> None:
    """Reset logger state. Called by conftest between tests."""
    global _log_fh, _hand_counter
    if _log_fh is not None:
        _log_fh.close()
    _log_fh = None
    _hand_counter = 0


def log_decision(
    state: Any,
    pos: int,
    bb: int,
    action: Any,
    sim_result: Any | None = None,
) -> None:
    """
    Write one JSON line capturing the full decision context.

    Called from decide() after _validate(). sim_result is None for preflop.
    """
    if not _ensure_open():
        return

    import bot

    stack_bb = state.chips / bb

    # Opponent archetypes for non-folded opponents only
    archetypes: dict[str, str] = {}
    for pid_str, folded in state.player_folded.items():
        pid_int = int(pid_str)
        if not folded and pid_int != state.my_pid:
            archetypes[pid_str] = bot._OPP.archetype(pid_int)

    entry: dict[str, Any] = {
        "hand":       _hand_counter,
        "street":     state.street,
        "position":   pos,
        "hole_cards": state.hole_cards,
        "community":  state.community,
        "stack_bb":   round(stack_bb, 2),
        "pot":        state.pot,
        "to_call":    state.to_call,
        "pot_odds":   round(state.pot_odds, 4),
        "archetypes": archetypes,
    }

    if sim_result is not None:
        entry["equity"]             = round(sim_result.equity, 4)
        entry["risk_adj_equity"]    = round(sim_result.risk_adj_equity, 4)
        entry["draw_premium"]       = round(sim_result.draw_premium, 4)
        entry["cvar"]               = round(sim_result.cvar, 4)
        entry["variance"]           = round(sim_result.variance, 4)
        entry["continuation_value"] = round(sim_result.continuation_value, 4)
        entry["ev_by_action"]       = {k: round(v, 2) for k, v in sim_result.ev_by_action.items()}
        entry["lambda"]             = sim_result.lambda_
        entry["n_sims"]             = sim_result.n_sims

    # Normalize tuple to list for JSON serialisation
    entry["action"] = list(action) if isinstance(action, tuple) else action

    _log_fh.write(json.dumps(entry) + "\n")
    _log_fh.flush()
