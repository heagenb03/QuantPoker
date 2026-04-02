"""
poker/decide.py — Main decision entry point.

decide(state) is the only function called by the server (via BotClient).
Routes to preflop or postflop modules and validates the result.
"""

from poker.position import get_position
from poker.preflop import _preflop
from poker.postflop import _postflop, _validate
from poker.simulation import run_unified_simulation
from poker.decision_log import log_decision


def decide(state):
    """
    Given the current game state, return your action.

    Returns one of:
        "fold"
        "check"              — only valid when state.can_check is True
        "call"
        "allin"
        ("raise", amount)    — amount = total bet level (>= state.min_raise)
    """
    import bot  # access bot._HLEN, bot._OPP, bot.BIG_BLIND, bot.DEALER_PID

    # ── Step 1: Update opponent model with any new history ─────────
    # Only processes messages added since the last call — O(new msgs).
    new_msgs = state.history[bot._HLEN:]
    bot._OPP.update(new_msgs)
    bot._HLEN = len(state.history)

    # ── Step 2: Derive table parameters ───────────────────────────
    bb       = bot.BIG_BLIND
    stack_bb = state.chips / bb

    # ── Step 3: Position (0=BTN, acts last postflop = best) ────────
    pos = get_position(state.my_pid, bot.DEALER_PID, state.num_players)

    # ── Step 4: Street-specific decision ──────────────────────────
    sim_result = None
    if state.street == 'preflop':
        action = _preflop(state, pos, bb)
    else:
        # Unified simulation: Steps 1-3 in one pass.
        # Preflop skips this — preflop_strength() is fast and sufficient.
        sim_result = run_unified_simulation(state, bb, n_sims=600)
        action     = _postflop(state, sim_result, pos, bb)

    # ── Step 5: Validate and return ───────────────────────────────
    action = _validate(action, state)

    # ── Step 6: Log decision (no-op unless --log-decisions enabled) ─
    log_decision(state, pos, bb, action, sim_result)

    return action
