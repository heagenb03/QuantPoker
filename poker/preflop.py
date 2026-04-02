"""
poker/preflop.py — Preflop hand strength and decision logic.

Preflop modules:
- preflop_strength()    — Chen formula, [0,1]. Fast, no simulation.
- _build_strength_lookup() — precomputes 169-entry hand strength table.
- open_strength_threshold() — converts hand-fraction percentile to Chen score.
- _push_fold()          — binary push/fold for <15BB stacks.
- _preflop()            — main preflop decision dispatcher.
"""

from poker.constants import RANKS, RANK_IDX
from poker.position import open_range_pct, players_behind_preflop
from poker.config import (
    PUSH_FOLD_STACK_BB, PUSH_RANGE_FLOOR, PUSH_RANGE_BASE,
    PUSH_RANGE_POS_DECAY, PUSH_RANGE_TABLE_DECAY, PUSH_RANGE_TABLE_START,
    PUSH_FACING_SHOVE_MULT, OPEN_SIZE_BB, MANIAC_TIGHTEN_MULT,
    VS_3BET_SHOVE_SHORT, VS_3BET_SHOVE_DEEP, VS_3BET_SHORT_BB, VS_3BET_RAISE_MULT,
    SPECULATIVE_STRENGTH, SPECULATIVE_POT_ODDS,
    CALL_THRESH_DEFAULT, CALL_THRESH_VS_NIT, CALL_THRESH_VS_MANIAC, CALL_THRESH_VS_STATION,
    SQUEEZE_STRENGTH_MULT, SQUEEZE_RAISE_MULT, MIN_STACK_TO_SQUEEZE,
    POT_ODDS_CALL_MAX,
)


def preflop_strength(hole_cards: list) -> float:
    """
    Chen formula score normalized to [0, 1].
    AA ≈ 1.0   72o ≈ 0.0
    Used for open/fold thresholds — fast, no simulation needed.
    """
    r1 = RANK_IDX[hole_cards[0][0]]
    r2 = RANK_IDX[hole_cards[1][0]]
    s1 = hole_cards[0][1]
    s2 = hole_cards[1][1]
    if r1 < r2:
        r1, r2 = r2, r1                        # r1 = higher rank

    _pts = {12:10, 11:8, 10:7, 9:6, 8:5, 7:4.5,
            6:4,   5:3.5, 4:3, 3:2.5, 2:2, 1:1.5, 0:1}
    score = _pts[r1]

    if r1 == r2:                               # pair: double, min 5
        score = max(score * 2, 5)
    else:
        if s1 == s2:                           # suited: +2
            score += 2
        gap = r1 - r2 - 1                      # 0=connected, 1=one gap …
        if   gap == 1: score -= 1
        elif gap == 2: score -= 2
        elif gap == 3: score -= 4
        elif gap >= 4: score -= 5
        if gap <= 0 and r1 < 11:               # connected below Q: +1
            score += 1

    # Raw range ≈ [-4, 20]  →  normalize to [0, 1]
    return max(0.0, min(1.0, (score + 4) / 24))


def _build_strength_lookup() -> list:
    """Sorted Chen scores for all 169 canonical starting hand types.
    Index 0 = strongest (AA ≈ 1.0), index 168 = weakest (72o ≈ 0.10).
    Used by open_strength_threshold() to convert a hand-fraction percentile
    into a minimum Chen score — fixing the direct `strength >= open_pct`
    comparison which breaks at large tables where open_pct drops below the
    minimum possible Chen score (~0.10), causing every hand to pass.
    """
    scores = []
    for i, r1 in enumerate(RANKS):
        for r2 in RANKS[i:]:
            if r1 != r2:
                scores.append(preflop_strength([r1 + 'h', r2 + 'h']))  # suited
            scores.append(preflop_strength([r1 + 'h', r2 + 'c']))      # offsuit / pair
    return sorted(scores, reverse=True)   # descending: strongest first


_HAND_STRENGTHS: list = _build_strength_lookup()   # 169 entries, computed once at import


def open_strength_threshold(open_pct: float) -> float:
    """Convert a hand-fraction open range to a minimum Chen score threshold.

    open_pct=0.20 → return the Chen score of the weakest hand still in the
    top 20% of all 169 starting hand types.

    Fixes the direct comparison `strength >= open_pct` which silently breaks
    at large tables: the minimum Chen score (~0.10 for 72o) exceeds the UTG
    open range at 13+ player tables (~0.08), making every hand appear openable.
    """
    idx = min(int(len(_HAND_STRENGTHS) * open_pct), len(_HAND_STRENGTHS) - 1)
    return _HAND_STRENGTHS[idx]


def _push_fold(state, strength: float,
               pos: int, n: int, bb: int, stack_bb: float) -> object:
    """
    Binary push/fold strategy for <15 BB stacks.
    Tightens with more players left to act (more chance of being called).
    Quant parallel: at critical drawdown threshold, reduce to binary
    all-in/out positions — no half-measures.
    """
    # Push threshold: looser on BTN, tighter UTG, tighter at big tables
    pos_pct   = pos / max(n - 1, 1)           # 0=BTN(loose) → 1=UTG(tight)
    push_pct  = max(PUSH_RANGE_FLOOR, PUSH_RANGE_BASE
                    - pos_pct * PUSH_RANGE_POS_DECAY
                    - max(n - PUSH_RANGE_TABLE_START, 0) * PUSH_RANGE_TABLE_DECAY)

    if state.to_call <= bb:                    # no raise or just BB
        if strength >= push_pct:
            return "allin"
        return "check" if state.can_check else "fold"
    else:                                      # facing a shove or big bet
        if strength >= push_pct * PUSH_FACING_SHOVE_MULT:
            return "allin"
        return "fold"


def _preflop(state, pos: int, bb: int) -> object:
    import bot

    strength = preflop_strength(state.hole_cards)
    n        = state.num_players
    stack_bb = state.chips / bb

    # Opponent archetypes among players still in the hand
    opp_types = [
        bot._OPP.archetype(int(p))
        for p, folded in state.player_folded.items()
        if not folded and int(p) != state.my_pid
    ]
    stations = opp_types.count('station')
    maniacs  = opp_types.count('maniac')
    nits     = opp_types.count('nit')

    # ── Short stack: push / fold only (<15 BB) ─────────────────────
    # Quant parallel: drawdown control — when stack is small,
    # variance is existential; switch to binary bet sizing.
    if stack_bb < PUSH_FOLD_STACK_BB:
        return _push_fold(state, strength, pos, n, bb, stack_bb)

    facing_raise = state.to_call > bb
    facing_3bet  = state.to_call > 3 * bb
    open_pct     = open_range_pct(pos, n)
    open_thresh  = open_strength_threshold(open_pct)

    # ── No raise to face: open opportunity ─────────────────────────
    if not facing_raise:
        if strength >= open_thresh:
            # Tighten slightly if maniacs will 3-bet us light
            if maniacs > 0 and strength < open_thresh * MANIAC_TIGHTEN_MULT:
                return "check" if state.can_check else "fold"
            # Sizing: OPEN_SIZE_BB + 1BB per limper already in the pot
            limpers  = max(0, (state.pot - bb - bb // 2) // bb)
            size     = int((OPEN_SIZE_BB + limpers) * bb)
            size     = max(size, state.min_raise)
            size     = min(size, state.chips + state.current_bet)
            return ("raise", size)
        return "check" if state.can_check else "fold"

    # ── Facing a 3-bet ─────────────────────────────────────────────
    if facing_3bet:
        shove_thresh = VS_3BET_SHOVE_SHORT if stack_bb < VS_3BET_SHORT_BB else VS_3BET_SHOVE_DEEP
        if strength >= shove_thresh:
            if stack_bb < VS_3BET_SHORT_BB:
                return "allin"
            size = min(int(state.to_call * VS_3BET_RAISE_MULT),
                    state.chips + state.current_bet)
            return ("raise", max(size, state.min_raise))
        # Speculative set-mine / suited connector in position
        if (pos in (0, n - 1) and strength >= SPECULATIVE_STRENGTH
                and state.pot_odds < SPECULATIVE_POT_ODDS):
            return "call"
        return "fold"

    # ── Facing a single raise ──────────────────────────────────────
    # Quant parallel: facing a raise = adverse selection signal.
    # Glosten-Milgrom: widen your required equity vs informed flow.
    call_thresh = CALL_THRESH_DEFAULT
    if nits    > 0: call_thresh = CALL_THRESH_VS_NIT
    if maniacs > 0: call_thresh = CALL_THRESH_VS_MANIAC
    if stations > 0: call_thresh = CALL_THRESH_VS_STATION

    if strength >= call_thresh * SQUEEZE_STRENGTH_MULT and stack_bb > MIN_STACK_TO_SQUEEZE:
        # 3-bet squeeze
        size = max(state.min_raise,
                   min(int(state.to_call * SQUEEZE_RAISE_MULT),
                    state.chips + state.current_bet))
        return ("raise", size)
    if strength >= call_thresh and state.pot_odds < POT_ODDS_CALL_MAX:
        return "call"
    return "fold"
