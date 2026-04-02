"""
poker/postflop.py — Postflop decision module and action validator.

Takes SimResult — never computes anything itself.
This is the isolation boundary: test _postflop by passing mock SimResult.
"""

import random


def _postflop(state, sim_result, pos: int, bb: int) -> object:
    """
    Selects the best postflop action given a SimResult.

    Decision flow:
        1. Start with ev_by_action from Step 3 (EV-ranked candidates)
        2. Apply archetype override rules (trump EV in specific cases)
        3. Filter to legal actions only
        4. Pick highest EV, translate key to concrete action + amount

    To debug in isolation: construct a SimResult manually and call this
    function directly — no simulation needed.
    """
    import bot

    ev = dict(sim_result.ev_by_action)   # mutable copy

    opp_types = {
        int(p): bot._OPP.archetype(int(p))
        for p, folded in state.player_folded.items()
        if not folded and int(p) != state.my_pid
    }
    station_present = 'station' in opp_types.values()
    maniac_present  = 'maniac'  in opp_types.values()
    n_active        = state.active_opponents
    pot             = state.pot

    # ── Override Rule 1: never bluff a station ────────────────────
    # Stations call everything — bluffing is strictly -EV.
    # Remove bet options when station is present AND we're not value-betting.
    if station_present and sim_result.risk_adj_equity < 0.52:
        ev.pop('bet_half', None)
        ev.pop('bet_full', None)

    # ── Override Rule 2: trap maniacs with strong hands ───────────
    # Maniacs bet into us — check and let the pot grow naturally.
    # Equivalent to not showing our full order book, letting price discover.
    if (maniac_present
            and sim_result.risk_adj_equity >= 0.65
            and state.can_check
            and sim_result.street != 'river'):
        # Force 'check' to win by boosting its EV above any bet option
        best_bet = max(ev.get('bet_half', 0), ev.get('bet_full', 0), 0)
        ev['check'] = max(ev.get('check', 0), best_bet * 1.15)

    # ── Override Rule 3: bluff gate ───────────────────────────────
    # All conditions must hold to allow a bluff bet.
    # Quant parallel: only deploy a low-Sharpe alpha signal when the
    # statistical edge justifies the variance — not by default.
    bluff_ok = (
        n_active == 1                      # heads-up: one target
        and pos == 0                       # BTN: in position, max info
        and not station_present            # station won't fold
        and sim_result.draw_premium < 0.02 # representing strength, not a draw
        and random.random() < 0.25         # frequency cap: avoid being readable
    )
    if not bluff_ok and sim_result.risk_adj_equity < 0.42:
        ev.pop('bet_half', None)
        ev.pop('bet_full', None)

    # ── Override Rule 4: pot commitment ───────────────────────────
    # If folding concedes > 60% of chips, call regardless of EV.
    if state.to_call >= state.chips * 0.60:
        ev['call'] = max(ev.get('call', 0),
                         sim_result.continuation_value * (pot + state.to_call))

    # ── Filter to legal actions ────────────────────────────────────
    legal_keys = {'fold', 'shove'}
    if state.can_check:
        legal_keys |= {'check', 'bet_half', 'bet_full'}
    if state.to_call > 0:
        legal_keys |= {'call'}

    available = {k: v for k, v in ev.items() if k in legal_keys}
    if not available:
        return 'fold'

    best_key = max(available, key=available.__getitem__)

    # ── Translate key to concrete action ──────────────────────────
    if best_key == 'fold':     return 'fold'
    if best_key == 'check':    return 'check'
    if best_key == 'call':     return 'call'
    if best_key == 'shove':    return 'allin'

    if best_key == 'bet_half':
        bet = max(state.min_raise, int(pot * 0.50))
        bet = min(bet, state.chips + state.current_bet)
        return ('raise', bet)

    if best_key == 'bet_full':
        bet = max(state.min_raise, int(pot * 1.00))
        bet = min(bet, state.chips + state.current_bet)
        return ('raise', bet)

    return 'fold'   # safety fallback


def _validate(action, state) -> object:
    """Ensure the returned action is legal before sending."""
    if action == "check" and not state.can_check:
        return "fold"
    if isinstance(action, tuple):
        verb, amount = action
        amount = int(amount)
        amount = max(amount, state.min_raise)
        amount = min(amount, state.chips + state.current_bet)
        if amount <= state.current_bet:        # can't actually raise
            return "call"
        return (verb, amount)
    return action
