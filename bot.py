"""
bot.py — Tournament Poker Bot
==============================
Architecture:
1. Parse history → _OpponentModel (VPIP, PFR, AF, showdown calibration)
2. Detect position from hand_start messages stored in history
3. Preflop: situation classifier → polarized ranges (position × table_size)
4. Postflop: run_unified_simulation() → SimResult → _postflop()

Unified simulation (postflop only) — three steps in one loop:
Step 1: full distribution (equity + variance + CVaR + risk-adjusted equity)
Step 2: continuation value (draw premium per street — Longstaff-Schwartz intuition)
Step 3: per-action EV (opponent fold/bet probabilities from archetype model)

Isolation pattern:
SimResult is the only interface between simulation and decision.
_postflop(state, sim_result, pos, bb) takes SimResult, never raw floats.
Tests construct SimResult directly — no simulation needed to test decisions.

Quant finance parallels:
- Step 1 variance + CVaR     ↔  risk-adjusted return, Sharpe penalization
- Step 1 lambda scaling      ↔  fractional Kelly, risk aversion by drawdown
- Step 2 draw premium        ↔  option value / theta decay by street
- Step 3 fold equity         ↔  market impact, Kyle lambda
- Opponent archetypes        ↔  counterparty classification, alpha signals

Usage:
    pip install eval7
    python bot.py [--host HOST] [--port PORT] [--name NAME]
"""

import socket
import json
import argparse
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, field

# ── Module-level globals set by BotClient on welcome / hand_start ─
BIG_BLIND  = 20       # updated from "welcome"
DEALER_PID = None     # updated from "hand_start"
NUM_DECKS  = 1        # updated from "welcome" via num_players

# ── Card constants ─────────────────────────────────────────────────
RANKS     = '23456789TJQKA'
SUITS     = 'cdhs'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
RANK_IDX  = {r: i for i, r in enumerate(RANKS)}   # '2'→0 … 'A'→12

# ── Hand evaluation wrapper ────────────────────────────────────────
# Multi-deck aware: eval7 fast path for single-deck (no duplicates possible);
# custom rank-counting evaluator for multi-deck (where duplicates can occur).

_RANK_ORDER = {r: i+2 for i, r in enumerate('23456789TJQKA')}  # '2'→2 … 'A'→14

from itertools import combinations as _combinations

def _eval_5card_multideck(cards: list) -> tuple:
    """Evaluate a 5-card hand with possible duplicates.
    Returns (category, tiebreakers) — higher tuple = stronger hand."""
    ranks = [_RANK_ORDER[c[0]] for c in cards]
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    counts_desc = sorted(rank_counts.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    unique_sorted = sorted(set(ranks), reverse=True)

    is_straight, straight_high = False, 0
    if len(unique_sorted) >= 5:
        for i in range(len(unique_sorted) - 4):
            w = unique_sorted[i:i+5]
            if w[0] - w[4] == 4:
                is_straight, straight_high = True, w[0]
                break
        if not is_straight and 14 in unique_sorted:
            lows = sorted([r for r in unique_sorted if r <= 5] + [1], reverse=True)
            if len(lows) >= 5 and lows[0] - lows[4] == 4:
                is_straight, straight_high = True, 5

    if counts_desc[0] >= 5:
        fivek = max(r for r, c in rank_counts.items() if c >= 5)
        return (9, (fivek,))
    if is_straight and is_flush:
        return (8, (straight_high,))
    if counts_desc[0] >= 4:
        quad = max(r for r, c in rank_counts.items() if c >= 4)
        kicker = max(r for r in ranks if r != quad)
        return (7, (quad, kicker))
    if counts_desc[0] >= 3 and len(counts_desc) > 1 and counts_desc[1] >= 2:
        trip = max(r for r, c in rank_counts.items() if c >= 3)
        pair = max(r for r, c in rank_counts.items() if c >= 2 and r != trip)
        return (6, (trip, pair))
    if is_flush:
        return (5, tuple(sorted(ranks, reverse=True)))
    if is_straight:
        return (4, (straight_high,))
    if counts_desc[0] >= 3:
        trip = max(r for r, c in rank_counts.items() if c >= 3)
        kickers = tuple(sorted([r for r in set(ranks) if r != trip], reverse=True)[:2])
        return (3, (trip,) + kickers)
    if len(counts_desc) > 1 and counts_desc[0] >= 2 and counts_desc[1] >= 2:
        pairs = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)[:2]
        kicker = max(r for r in set(ranks) if r not in pairs)
        return (2, tuple(pairs) + (kicker,))
    if counts_desc[0] >= 2:
        pair_r = max(r for r, c in rank_counts.items() if c >= 2)
        kickers = tuple(sorted([r for r in set(ranks) if r != pair_r], reverse=True)[:3])
        return (1, (pair_r,) + kickers)
    return (0, tuple(sorted(ranks, reverse=True)[:5]))


def _eval_multideck(card_strings: list) -> int:
    """Evaluate a 5–7 card hand with possible duplicates (multi-deck).
    Returns an integer where higher = stronger (matches eval7 convention)."""
    cards = [(s[0], s[1]) for s in card_strings]
    best = (-1, ())
    for combo in _combinations(range(len(cards)), 5):
        five = [cards[i] for i in combo]
        score = _eval_5card_multideck(five)
        if score > best:
            best = score
    cat, tb = best
    result = cat << 20
    for i, v in enumerate(tb[:5]):
        result |= v << (4 * (4 - i))
    return result


try:
    import eval7 as _ev7
    def _eval(card_strings: list) -> int:
        """Evaluate a hand. Higher score = stronger (eval7 convention).
        Single-deck: delegates to eval7 (C speed).
        Multi-deck: uses custom rank-counting evaluator (handles duplicates)."""
        if NUM_DECKS == 1:
            return _ev7.evaluate([_ev7.Card(c) for c in card_strings])
        return _eval_multideck(card_strings)
    print("[BOT] eval7 loaded ✓")
except ImportError:
    print("[BOT] WARNING: eval7 not found.  Run:  pip install eval7")
    def _eval(card_strings: list) -> int:
        if NUM_DECKS == 1:
            # Naive rank-sum fallback — functional but weak. Install eval7!
            return sum(RANK_IDX[c[0]] for c in card_strings)
        return _eval_multideck(card_strings)


# ═══════════════════════════════════════════════════════════════════
# GAME STATE  (read-only snapshot given to decide())
# ═══════════════════════════════════════════════════════════════════

class GameState:
    """Every fact about the current decision point."""

    def __init__(self, raw: dict, history: list, my_pid: int):
        self.my_pid        = my_pid
        self.hole_cards    = raw["hole_cards"]      # e.g. ["Ah","Kd"]
        self.community     = raw["community"]        # 0–5 cards
        self.street        = raw["street"]           # preflop|flop|turn|river
        self.chips         = raw["chips"]
        self.pot           = raw["pot"]
        self.to_call       = raw["to_call"]
        self.current_bet   = raw["current_bet"]
        self.min_raise     = raw["min_raise"]
        self.num_players   = raw["num_players"]      # players IN THIS HAND
        self.player_bets   = raw["player_bets"]
        self.player_chips  = raw["player_chips"]
        self.player_folded = raw["player_folded"]
        self.player_allin  = raw["player_allin"]
        self.history       = history

    @property
    def can_check(self):
        return self.to_call == 0

    @property
    def active_opponents(self):
        """Non-folded opponents still in the hand (excluding me)."""
        return sum(1 for pid, folded in self.player_folded.items()
                   if not folded and str(pid) != str(self.my_pid))

    @property
    def pot_odds(self):
        """Fraction of (pot + call) I must invest to call. 0 if free."""
        if self.to_call == 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    def __repr__(self):
        return (f"<GameState {self.street} hole={self.hole_cards} "
                f"board={self.community} pot={self.pot} "
                f"chips={self.chips} to_call={self.to_call}>")


# ═══════════════════════════════════════════════════════════════════
# 1. PREFLOP HAND STRENGTH  (Chen formula, normalized 0→1)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# 2. SIMRESULT — isolation boundary between simulation and decision
#    _postflop() takes SimResult, never raw floats.
#    Tests construct SimResult directly without running simulation.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    # Step 1 — full distribution (not just mean)
    equity:           float   # P(win at showdown) — mean across simulations
    variance:         float   # spread of outcomes — how bimodal is the distribution?
    cvar:             float   # mean of worst 20% outcomes (CVaR / tail risk)
    risk_adj_equity:  float   # equity - λ*variance (Sharpe-penalized value)

    # Step 2 — continuation value (draw-aware, street-specific)
    continuation_value: float  # equity adjusted for implied odds by street
    draw_premium:       float  # continuation_value - equity
                               # positive = drawing hand (option value present)
                               # negative = made hand vulnerable to draws

    # Step 3 — per-action EV from strategy simulation
    ev_by_action: dict         # {'check': chips, 'call': chips,
                               #  'bet_half': chips, 'bet_full': chips,
                               #  'shove': chips, 'fold': 0}

    # Metadata (for debugging and test assertion)
    street:      str
    n_opponents: int
    n_sims:      int
    lambda_:     float         # risk aversion used (scales with stack depth)


# ── Archetype response tables (used in Steps 1 and 3) ─────────────
_FOLD_TO_BET = {
    'nit':     0.65,
    'tag':     0.45,
    'station': 0.08,   # stations call almost everything
    'maniac':  0.20,
    'unknown': 0.40,
}
_BET_WHEN_CHECKED = {
    'nit':     0.20,
    'tag':     0.45,
    'station': 0.35,
    'maniac':  0.75,   # maniacs bet into you constantly
    'unknown': 0.35,
}


def run_unified_simulation(state: 'GameState', bb: int,
                           n_sims: int = 600) -> SimResult:
    """
    Three-step unified Monte Carlo producing a SimResult.
    All three steps share one loop over the same simulations — no repeated work.

    Step 1 — full distribution:
        Keeps all outcomes, computes equity + variance + CVaR + risk_adj_equity.
        lambda_ (risk aversion) scales with stack depth:
            deep (≥60BB) → λ=0.05  (nearly risk-neutral)
            medium (30BB) → λ=0.15
            short (≤15BB) → λ=0.28  (variance is existential)
        Quant parallel: Sharpe-penalized return. CVaR = tail risk measure.

    Step 2 — continuation value per street:
        Within each simulation, compares our current board hand strength
        to terminal outcome. Detects draw paths (we were weak but improved).
        draw_premium: positive = drawing hand (option value), negative = made
        hand vulnerable to being drawn out on.
        Street discount: flop 0.70 / turn 0.90 / river 1.0 (theta decay).
        Quant parallel: American option vs European. Draws are options on
        future board cards. Option value decays as expiry approaches.

    Step 3 — per-action EV with opponent response:
        For each candidate action, computes E[chips] given opponents respond
        according to their archetype (not assuming everyone checks to showdown).
        p_all_fold compounds across all active opponents independently.
        Quant parallel: market impact model. Your bet triggers opponent
        response — sizing interacts with their fold probability.
    """
    street      = state.street
    hole_cards  = state.hole_cards
    board       = state.community
    n_opponents = state.active_opponents
    stack_bb    = state.chips / bb

    # ── Risk aversion (lambda): scales with stack depth ────────────
    # Quant parallel: fractional Kelly. Deep stacks tolerate variance;
    # short stacks must minimize it — ruin is permanent in a freezeout.
    lambda_ = max(0.05, min(0.28, 0.30 - (stack_bb - 15) * 0.008))

    # ── Opponent archetypes for Steps 2 and 3 ─────────────────────
    opp_pids = [
        int(p) for p, folded in state.player_folded.items()
        if not folded and int(p) != state.my_pid
    ]
    archetypes = [_OPP.archetype(pid) for pid in opp_pids]

    # ── Early exit: no opponents (won uncontested) ─────────────────
    if n_opponents <= 0:
        ev = {'fold': 0, 'check': state.pot, 'shove': state.pot}
        return SimResult(
            equity=1.0, variance=0.0, cvar=1.0, risk_adj_equity=1.0,
            continuation_value=1.0, draw_premium=0.0,
            ev_by_action=ev, street=street, n_opponents=0,
            n_sims=n_sims, lambda_=lambda_
        )

    # ── Set up card pool ───────────────────────────────────────────
    full_deck = ALL_CARDS * NUM_DECKS
    available = list((Counter(full_deck) - Counter(hole_cards + board)).elements())
    needed    = n_opponents * 2 + (5 - len(board))

    if len(available) < needed:
        # Degenerate state (too few cards) — return neutral SimResult
        ev = {'fold': 0, 'check': state.pot * 0.5, 'call': -state.to_call * 0.5}
        return SimResult(
            equity=0.5, variance=0.25, cvar=0.0, risk_adj_equity=0.5 - lambda_ * 0.25,
            continuation_value=0.5, draw_premium=0.0,
            ev_by_action=ev, street=street, n_opponents=n_opponents,
            n_sims=n_sims, lambda_=lambda_
        )

    # ── Step 2 setup: current board hand strength (one-time eval) ──
    # Compare current partial-board score to terminal score per simulation.
    # If terminal is significantly stronger, we "improved" — a draw path.
    # eval7 is consistent across 5/6/7 card evals (best-5-of-N logic).
    if board:
        current_board_score = _eval(hole_cards + board)
    else:
        current_board_score = 0   # preflop — no board yet

    # ── Main simulation loop (Steps 1 and 2 data collected here) ───
    outcomes        = []   # 0 / 0.5 / 1 per simulation (Step 1)
    terminal_scores = []   # our eval7 hand score per simulation (Step 2)

    for _ in range(n_sims):
        deck = available[:]
        random.shuffle(deck)

        opp_hands  = [deck[i*2 : i*2+2] for i in range(n_opponents)]
        cursor     = n_opponents * 2
        sim_board  = board + deck[cursor : cursor + (5 - len(board))]

        my_score   = _eval(hole_cards + sim_board)
        opp_best   = max(_eval(h + sim_board) for h in opp_hands)

        if   my_score > opp_best: outcomes.append(1.0)
        elif my_score == opp_best: outcomes.append(0.5)
        else:                      outcomes.append(0.0)

        terminal_scores.append(my_score)

    # ── Step 1: distribution computations ─────────────────────────
    equity   = sum(outcomes) / n_sims
    variance = sum((x - equity) ** 2 for x in outcomes) / n_sims

    # CVaR: mean of worst 20% of outcomes (tail risk, not just spread)
    # Quant parallel: Expected Shortfall — what we lose in bad scenarios
    cvar_n   = max(1, int(n_sims * 0.20))
    cvar     = sum(sorted(outcomes)[:cvar_n]) / cvar_n

    # Risk-adjusted equity: penalize variance by stack-depth-scaled lambda
    risk_adj_equity = equity - lambda_ * variance

    # ── Step 2: draw premium and continuation value ────────────────
    # Draw path: terminal hand is meaningfully stronger than current board.
    # eval7: higher score = stronger hand (opposite of server convention).
    IMPROVE_THRESH = 1.08   # terminal must be 8% stronger to count as improvement
    if board and street != 'river':
        improved = [s > current_board_score * IMPROVE_THRESH for s in terminal_scores]
        n_imp    = sum(improved)
        n_steady = n_sims - n_imp

        imp_wins    = sum(o for o, imp in zip(outcomes, improved) if imp)
        steady_wins = sum(o for o, imp in zip(outcomes, improved) if not imp)

        imp_rate    = imp_wins    / n_imp    if n_imp    > 0 else equity
        steady_rate = steady_wins / n_steady if n_steady > 0 else equity

        # draw_premium: positive if draw paths contribute above-average wins
        # negative if we're a made hand being drawn out on
        improve_freq = n_imp / n_sims
        raw_premium  = improve_freq * (imp_rate - steady_rate)
        # Cap to prevent outlier sims from distorting the number
        draw_premium = max(-0.12, min(0.18, raw_premium))

        # Street discount: option value decays as fewer cards remain
        # Quant parallel: theta decay — options lose value as expiry approaches
        discount = {'flop': 0.70, 'turn': 0.90}.get(street, 1.0)
        continuation_value = max(0.0, min(1.0, equity + draw_premium * discount))
    else:
        # River or no board: no future streets, no option value
        draw_premium       = 0.0
        continuation_value = equity

    # ── Step 3: per-action EV with opponent response ───────────────
    # Models opponents as active agents, not passive card-holders.
    # Quant parallel: execution algorithm accounting for market impact —
    # your action triggers opponent response, changing the outcome distribution.
    pot     = state.pot
    to_call = state.to_call

    # Fold probability compounds independently across all opponents
    # (all must fold for a bet to win the pot uncontested)
    p_all_fold = 1.0
    for pid in opp_pids:
        p_all_fold *= _OPP.fold_to_bet_est(pid)

    # Probability at least one opponent bets when checked to
    p_no_bet = 1.0
    for pid in opp_pids:
        p_no_bet *= (1.0 - _OPP.bet_when_checked_est(pid))
    p_someone_bets = 1.0 - p_no_bet

    bet_half  = max(state.min_raise, int(pot * 0.50))
    bet_full  = max(state.min_raise, int(pot * 1.00))
    shove_amt = state.chips + state.current_bet

    # Helper: EV of betting a fixed size (fold equity + call equity)
    def _ev_bet(size: int) -> float:
        return (p_all_fold * pot
                + (1.0 - p_all_fold)
                * (continuation_value * (pot + size) - size))

    ev_by_action: dict = {'fold': 0.0}

    if state.can_check:
        # Assume opponent bets ~half pot when they bet (conservative)
        implied_bet = max(1, int(pot * 0.50))
        new_pot_if_bet = pot + implied_bet * 2
        if continuation_value > 0.25:   # worth calling their bet
            ev_if_bet = continuation_value * new_pot_if_bet - implied_bet
        else:
            ev_if_bet = 0.0             # we fold to their bet
        ev_by_action['check']    = (p_no_bet * continuation_value * pot
                                    + p_someone_bets * ev_if_bet)
        ev_by_action['bet_half'] = _ev_bet(bet_half)
        ev_by_action['bet_full'] = _ev_bet(bet_full)

    if to_call > 0:
        ev_by_action['call'] = continuation_value * (pot + to_call) - to_call

    ev_by_action['shove'] = _ev_bet(shove_amt)

    return SimResult(
        equity=equity,
        variance=variance,
        cvar=cvar,
        risk_adj_equity=risk_adj_equity,
        continuation_value=continuation_value,
        draw_premium=draw_premium,
        ev_by_action=ev_by_action,
        street=street,
        n_opponents=n_opponents,
        n_sims=n_sims,
        lambda_=lambda_,
    )


def monte_carlo_equity(hole_cards: list, board: list,
                       n_opponents: int, n_sims: int = 200) -> float:
    """
    Thin wrapper returning raw equity only.
    Kept for backward compat with tests and BotClient logging.
    Production code uses run_unified_simulation() instead.
    """
    if n_opponents <= 0:
        return 1.0
    full_deck = ALL_CARDS * NUM_DECKS
    available = list((Counter(full_deck) - Counter(hole_cards + board)).elements())
    needed    = n_opponents * 2 + (5 - len(board))
    if len(available) < needed:
        return 0.5
    wins = ties = 0
    for _ in range(n_sims):
        deck = available[:]
        random.shuffle(deck)
        opp_hands = [deck[i*2 : i*2+2] for i in range(n_opponents)]
        cursor    = n_opponents * 2
        sim_board = board + deck[cursor : cursor + (5 - len(board))]
        my_score  = _eval(hole_cards + sim_board)
        opp_best  = max(_eval(h + sim_board) for h in opp_hands)
        if   my_score > opp_best: wins += 1
        elif my_score == opp_best: ties += 1
    return (wins + ties * 0.5) / n_sims


# ═══════════════════════════════════════════════════════════════════
# 3. POSITION
#    Position = information asymmetry.
#    Late position (acting last) ↔ latency advantage in e-trading.
# ═══════════════════════════════════════════════════════════════════

def get_position(my_pid: int, dealer_pid, n_players: int) -> int:
    """
    Clockwise distance from the dealer button.
    0 = BTN (acts last postflop — best)
    1 = SB,  2 = BB,  3 = UTG, …,  n-1 = CO
    """
    if dealer_pid is None:
        return n_players // 2                  # unknown → assume middle
    return (my_pid - dealer_pid) % n_players


def players_behind_preflop(pos: int, n: int) -> int:
    """
    Players who act after me preflop (the real variable underlying ranges).
    Named position labels (BTN/CO/EP) are shortcuts for this value —
    they break at indefinite table sizes and are not used.
    """
    if pos == 2: return 0   # BB closes action
    if pos == 1: return 1   # SB: only BB behind
    if pos == 0: return 2   # BTN: SB and BB behind
    return (n - pos) + 2    # UTG through CO


def open_range_pct(pos: int, n: int) -> float:
    """
    Fraction of hands to open-raise. Continuous exponential decay —
    no named buckets, works at any table size.
    Each extra player behind reduces range by ~15%.
    CO always has 3 players behind regardless of n → always ~41%.
    Ranges loosen as players bust (n shrinks) — no manual tuning.
    """
    behind = players_behind_preflop(pos, n)
    if behind <= 1:
        return 0.38 if behind == 1 else 0.30   # SB / BB
    pct = 0.48 * (0.85 ** (behind - 2))
    return max(0.06, pct)                        # floor: always open aces


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


# ═══════════════════════════════════════════════════════════════════
# 4. OPPONENT MODEL
#    Tracks VPIP, PFR, aggression, and showdown hole cards per pid.
#    Classifies into archetypes → exploit table.
#    Quant parallel: factor model + alpha signal construction.
#    Showdown reveals = labeled training data (unique to this server).
# ═══════════════════════════════════════════════════════════════════

class _OpponentModel:
    """
    Incremental opponent model built from state.history messages.
    Key signals (minimum 10 hands before trusting a read):
      VPIP   — how often they voluntarily invest preflop
      PFR    — how often they raise preflop
      AF     — aggression factor (raises / calls)
      showdown cards — ground-truth range calibration
    """

    def __init__(self):
        self._s = defaultdict(lambda: {
            'hands': 0,
            'vpip':  0,
            'pfr':   0,
            'raises': 0,
            'calls':  0,
            'showdown': [],        # list of hole-card pairs seen at showdown
            # Bayesian frequency tracking (Beta distribution parameters)
            'ftb_alpha': 0.0,      # fold-to-bet successes
            'ftb_beta':  0.0,      # fold-to-bet failures (call/raise)
            'bwc_alpha': 0.0,      # bet-when-checked successes
            'bwc_beta':  0.0,      # bet-when-checked failures (check)
            'ftb_seeded': False,   # whether prior has been seeded
            'bwc_seeded': False,
        })
        self._vpip_this_hand = set()
        self._pfr_this_hand  = set()
        self._had_bet_this_street: bool = False
        self._current_street: str = ''

    def update(self, messages: list) -> None:
        """Process only the NEW messages appended since last call."""
        for msg in messages:
            t = msg.get('type')

            if t == 'hand_start':
                # Commit VPIP / PFR counts from the hand that just ended
                for pid in self._vpip_this_hand:
                    self._s[pid]['vpip'] += 1
                for pid in self._pfr_this_hand:
                    self._s[pid]['pfr']  += 1
                # Count one hand seen for every player still at the table
                for pid_str in msg.get('stacks', {}):
                    self._s[int(pid_str)]['hands'] += 1
                self._vpip_this_hand = set()
                self._pfr_this_hand  = set()
                # Reset street tracking for new hand
                self._had_bet_this_street = False
                self._current_street = ''

            elif t == 'player_action':
                pid    = msg['pid']
                action = msg['action']
                street = msg.get('street', '')

                # Detect street transition → reset bet tracking
                if street != self._current_street:
                    self._had_bet_this_street = False
                    self._current_street = street

                if street == 'preflop':
                    if action in ('call', 'raise', 'allin'):
                        self._vpip_this_hand.add(pid)
                    if action in ('raise', 'allin'):
                        self._pfr_this_hand.add(pid)

                # Bayesian updates — postflop only
                if street in ('flop', 'turn', 'river'):
                    if self._had_bet_this_street:
                        # Opponent faces a bet
                        if action == 'fold':
                            self._reseed_priors(pid, 'ftb')
                            self._s[pid]['ftb_alpha'] += 1
                        elif action in ('call', 'raise', 'allin'):
                            self._reseed_priors(pid, 'ftb')
                            self._s[pid]['ftb_beta'] += 1
                    else:
                        # No bet yet this street
                        if action in ('raise', 'bet', 'allin'):
                            self._reseed_priors(pid, 'bwc')
                            self._s[pid]['bwc_alpha'] += 1
                        elif action == 'check':
                            self._reseed_priors(pid, 'bwc')
                            self._s[pid]['bwc_beta'] += 1

                    # After processing, mark if a bet occurred
                    if action in ('raise', 'bet', 'allin'):
                        self._had_bet_this_street = True

                # Global aggression tracking (all streets)
                # Only count voluntary aggressive/passive actions — fold and check
                # are neutral and must not inflate the call count (which deflates AF).
                if action in ('raise', 'allin', 'bet'):
                    self._s[pid]['raises'] += 1
                elif action == 'call':
                    self._s[pid]['calls']  += 1

            elif t == 'showdown':
                # Ground-truth hole cards — the richest signal available.
                # Tells us exactly what they called a raise with, etc.
                for pid_str, info in msg.get('hands', {}).items():
                    cards = info.get('cards', [])
                    if cards:
                        self._s[int(pid_str)]['showdown'].append(cards)

    # ── Bayesian frequency accessors ─────────────────────────────

    def _reseed_priors(self, pid: int, param: str) -> None:
        """Seed Beta prior from archetype lookup on first observation.

        Deferred until archetype is not 'unknown' — seeding from the
        unknown prior (0.40/0.35) before hand 10 would lock in a stale
        prior that persists even after the player reclassifies. Holding
        off means the early observations accumulate without a prior, and
        the prior seeds from the correct archetype the moment it stabilises.

        Uses pseudocount of 5 so the prior is worth ~5 observations.
        *param* is 'ftb' or 'bwc'.

        NOTE: seeding (=) must happen before the caller increments (+=1).
        Do not reorder those two lines.
        """
        seeded_key = f'{param}_seeded'
        if self._s[pid][seeded_key]:
            return
        arch = self.archetype(pid)
        if arch == 'unknown':
            return   # defer until archetype stabilises (≥10 hands)
        if param == 'ftb':
            prior_mean = _FOLD_TO_BET.get(arch, 0.40)
        else:
            prior_mean = _BET_WHEN_CHECKED.get(arch, 0.35)
        # Add pseudocount on top of any raw observations already accumulated
        # while archetype was 'unknown'. This preserves early data rather
        # than overwriting it.
        pseudocount = 5.0
        self._s[pid][f'{param}_alpha'] += prior_mean * pseudocount
        self._s[pid][f'{param}_beta'] += (1.0 - prior_mean) * pseudocount
        self._s[pid][seeded_key] = True

    def fold_to_bet_est(self, pid: int) -> float:
        """Bayesian estimate of how often *pid* folds to a postflop bet.

        Pre-archetype-stabilisation (< 10 hands): raw observation counts
        are used directly (no pseudocount prior yet).
        Post-stabilisation: pseudocount prior has been added via _reseed_priors,
        so the posterior mean is (prior + observations) / total.
        Falls back to archetype lookup only when zero observations exist.
        """
        s = self._s[pid]
        total = s['ftb_alpha'] + s['ftb_beta']
        if total <= 0:
            return _FOLD_TO_BET.get(self.archetype(pid), 0.40)
        return s['ftb_alpha'] / total

    def bet_when_checked_est(self, pid: int) -> float:
        """Bayesian estimate of how often *pid* bets when checked to.

        See fold_to_bet_est for the seeding/prior semantics.
        """
        s = self._s[pid]
        total = s['bwc_alpha'] + s['bwc_beta']
        if total <= 0:
            return _BET_WHEN_CHECKED.get(self.archetype(pid), 0.35)
        return s['bwc_alpha'] / total

    # ── Derived stats ──────────────────────────────────────────────

    def vpip(self, pid: int) -> float:
        s = self._s[pid]
        return s['vpip'] / s['hands'] if s['hands'] >= 5 else 0.25

    def pfr(self, pid: int) -> float:
        s = self._s[pid]
        return s['pfr'] / s['hands'] if s['hands'] >= 5 else 0.12

    def af(self, pid: int) -> float:
        """Aggression factor: raises / calls. >2 = aggressive."""
        s = self._s[pid]
        total = s['raises'] + s['calls']
        return s['raises'] / max(s['calls'], 1) if total >= 5 else 1.0

    def archetype(self, pid: int) -> str:
        """
        Classify into one of four exploitable archetypes.
        Requires ≥10 hands; returns 'unknown' before that.

        Archetype → counter-strategy:
          nit     → steal often, fold to their aggression
          station → value bet relentlessly, never bluff
          maniac  → tighten preflop, trap with strong hands
          tag     → GTO-approximate play
        """
        if self._s[pid]['hands'] < 10:
            return 'unknown'
        v  = self.vpip(pid)
        af = self.af(pid)
        if v < 0.18:                  return 'nit'
        if v >= 0.35 and af < 1.5:   return 'station'
        if v >= 0.30 and af >= 2.5:  return 'maniac'
        return 'tag'

    def range_width(self, pid: int) -> float:
        """Estimated fraction of hands this opponent plays."""
        return {'nit':0.12, 'tag':0.22, 'station':0.45,
                'maniac':0.38, 'unknown':0.25}[self.archetype(pid)]


# ── Singleton — persists across all decide() calls ─────────────────
_OPP  = _OpponentModel()
_HLEN = 0    # index of last processed history message


# ═══════════════════════════════════════════════════════════════════
# 5. PREFLOP DECISION MODULE
# ═══════════════════════════════════════════════════════════════════

def _preflop(state: GameState, pos: int, bb: int) -> object:
    strength = preflop_strength(state.hole_cards)
    n        = state.num_players
    stack_bb = state.chips / bb

    # Opponent archetypes among players still in the hand
    opp_types = [
        _OPP.archetype(int(p))
        for p, folded in state.player_folded.items()
        if not folded and int(p) != state.my_pid
    ]
    stations = opp_types.count('station')
    maniacs  = opp_types.count('maniac')
    nits     = opp_types.count('nit')

    # ── Short stack: push / fold only (<15 BB) ─────────────────────
    # Quant parallel: drawdown control — when stack is small,
    # variance is existential; switch to binary bet sizing.
    if stack_bb < 15:
        return _push_fold(state, strength, pos, n, bb, stack_bb)

    facing_raise = state.to_call > bb
    facing_3bet  = state.to_call > 3 * bb
    open_pct     = open_range_pct(pos, n)
    open_thresh  = open_strength_threshold(open_pct)

    # ── No raise to face: open opportunity ─────────────────────────
    if not facing_raise:
        if strength >= open_thresh:
            # Tighten slightly if maniacs will 3-bet us light
            if maniacs > 0 and strength < open_thresh * 1.35:
                return "check" if state.can_check else "fold"
            # Sizing: 3BB + 1BB per limper already in the pot
            limpers  = max(0, (state.pot - bb - bb // 2) // bb)
            size     = int((3 + limpers) * bb)
            size     = max(size, state.min_raise)
            size     = min(size, state.chips + state.current_bet)
            return ("raise", size)
        return "check" if state.can_check else "fold"

    # ── Facing a 3-bet ─────────────────────────────────────────────
    if facing_3bet:
        # Short stacks use a lower shove threshold (AKo = 0.583, covers AK/AQ/TT+)
        shove_thresh = 0.55 if stack_bb < 25 else 0.65
        if strength >= shove_thresh:
            if stack_bb < 25:
                return "allin"
            size = min(int(state.to_call * 3),
                    state.chips + state.current_bet)
            return ("raise", max(size, state.min_raise))
        # Speculative set-mine / suited connector in position
        if (pos in (0, n - 1) and strength >= 0.12
                and state.pot_odds < 0.28):
            return "call"
        return "fold"

    # ── Facing a single raise ──────────────────────────────────────
    # Quant parallel: facing a raise = adverse selection signal.
    # Glosten-Milgrom: widen your required equity vs informed flow.
    call_thresh = 0.25
    if nits    > 0: call_thresh = 0.30    # nit's raise = tight range → need more
    if maniacs > 0: call_thresh = 0.18   # maniac raises wide → call lighter
    if stations > 0: call_thresh = 0.22

    if strength >= call_thresh * 1.5 and stack_bb > 20:
        # 3-bet squeeze
        size = max(state.min_raise,
                   min(int(state.to_call * 3),
                    state.chips + state.current_bet))
        return ("raise", size)
    if strength >= call_thresh and state.pot_odds < 0.35:
        return "call"
    return "fold"


def _push_fold(state: GameState, strength: float,
            pos: int, n: int, bb: int, stack_bb: float) -> object:
    """
    Binary push/fold strategy for <15 BB stacks.
    Tightens with more players left to act (more chance of being called).
    Quant parallel: at critical drawdown threshold, reduce to binary
    all-in/out positions — no half-measures.
    """
    # Push threshold: looser on BTN, tighter UTG, tighter at big tables
    pos_pct   = pos / max(n - 1, 1)           # 0=BTN(loose) → 1=UTG(tight)
    push_pct  = max(0.12, 0.52
                    - pos_pct * 0.28
                    - max(n - 6, 0) * 0.02)

    if state.to_call <= bb:                    # no raise or just BB
        if strength >= push_pct:
            return "allin"
        return "check" if state.can_check else "fold"
    else:                                      # facing a shove or big bet
        if strength >= push_pct * 1.25:
            return "allin"
        return "fold"


# ═══════════════════════════════════════════════════════════════════
# 6. POSTFLOP DECISION MODULE
#    Takes SimResult — never computes anything itself.
#    This is the isolation boundary: test _postflop by passing mock SimResult.
# ═══════════════════════════════════════════════════════════════════

def _postflop(state: GameState, sim_result: SimResult,
            pos: int, bb: int) -> object:
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
    ev = dict(sim_result.ev_by_action)   # mutable copy

    opp_types = {
        int(p): _OPP.archetype(int(p))
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


# ═══════════════════════════════════════════════════════════════════
# 7. ACTION VALIDATOR
# ═══════════════════════════════════════════════════════════════════

def _validate(action, state: GameState):
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


# ═══════════════════════════════════════════════════════════════════
#   decide()  ←  ONLY FUNCTION YOU MUST SUBMIT
# ═══════════════════════════════════════════════════════════════════

def decide(state: GameState):
    """
    Given the current game state, return your action.

    Returns one of:
        "fold"
        "check"              — only valid when state.can_check is True
        "call"
        "allin"
        ("raise", amount)    — amount = total bet level (>= state.min_raise)
    """
    global _HLEN, DEALER_PID, BIG_BLIND

    # ── Step 1: Update opponent model with any new history ─────────
    # Only processes messages added since the last call — O(new msgs).
    new_msgs = state.history[_HLEN:]
    _OPP.update(new_msgs)
    _HLEN = len(state.history)

    # ── Step 2: Derive table parameters ───────────────────────────
    bb       = BIG_BLIND
    stack_bb = state.chips / bb

    # ── Step 3: Position (0=BTN, acts last postflop = best) ────────
    pos = get_position(state.my_pid, DEALER_PID, state.num_players)

    # ── Step 4: Street-specific decision ──────────────────────────
    if state.street == 'preflop':
        action = _preflop(state, pos, bb)
    else:
        # Unified simulation: Steps 1-3 in one pass.
        # Preflop skips this — preflop_strength() is fast and sufficient.
        sim_result = run_unified_simulation(state, bb, n_sims=600)
        action     = _postflop(state, sim_result, pos, bb)

    # ── Step 6: Validate and return ───────────────────────────────
    return _validate(action, state)


# ═══════════════════════════════════════════════════════════════════
# BOT CLIENT  (networking — two small edits from original)
# ═══════════════════════════════════════════════════════════════════

class BotClient:
    def __init__(self, host, port, name="Bot"):
        self.host    = host
        self.port    = port
        self.name    = name
        self.pid     = None
        self.history = []
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._buf    = b''

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"[{self.name}] Connected to {self.host}:{self.port}")

    def send(self, msg: dict):
        self.sock.sendall((json.dumps(msg) + '\n').encode())

    def recv(self):
        while b'\n' not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        return json.loads(line.decode())

    def run(self):
        # ↓ Two module-level globals set here so decide() stays clean
        global BIG_BLIND, DEALER_PID
        self.connect()

        while True:
            msg = self.recv()
            if msg is None:
                print(f"[{self.name}] Server disconnected.")
                break

            t = msg.get("type")

            # ── EDIT 1: capture big_blind from welcome ─────────────
            if t == "welcome":
                global BIG_BLIND, NUM_DECKS
                self.pid  = msg["pid"]
                BIG_BLIND = msg["big_blind"]
                n_p = msg["num_players"]
                NUM_DECKS = max(1, min(4, (n_p // 5) + 1))
                print(f"[{self.name}] PID={self.pid}  chips={msg['chips']}  "
                    f"bb={BIG_BLIND}  players={n_p}  decks={NUM_DECKS}")

            # ── EDIT 2: capture dealer from hand_start + append it ─
            elif t == "hand_start":
                DEALER_PID = msg["dealer"]
                self.history.append(msg)       # opponent model needs this
                print(f"[{self.name}] ── New hand ── "
                    f"dealer={msg['dealer']}  sb={msg['sb']}  bb={msg['bb']}")

            elif t == "hole_cards":
                print(f"[{self.name}] Dealt: {msg['cards']}  "
                    f"chips={msg['chips']}  pot={msg['pot']}")
                self.history.append(msg)

            elif t == "community_cards":
                print(f"[{self.name}] Board ({msg['street']}): "
                    f"{msg['cards']}  pot={msg['pot']}")
                self.history.append(msg)

            elif t == "action_request":
                state  = GameState(msg, self.history, self.pid)
                action = decide(state)
                resp   = self._build_response(action, state)
                print(f"[{self.name}] Action → {resp}  "
                    f"[pos={get_position(self.pid, DEALER_PID, state.num_players)} "
                    f"bb={state.chips // BIG_BLIND}BB "
                    f"street={state.street}]")
                self.send(resp)
                self.history.append({"type": "my_action", **resp})

            elif t == "player_action":
                pid = msg["pid"]
                arch = _OPP.archetype(pid)
                print(f"[{self.name}] Player {pid} ({arch}) → "
                    f"{msg['action']} {msg.get('amount','')}  "
                    f"chips={msg.get('chips','?')}")
                self.history.append(msg)

            elif t == "showdown":
                print(f"[{self.name}] SHOWDOWN — winners: {msg['winners']}  "
                    f"pot={msg['pot']}")
                for pid, info in msg["hands"].items():
                    arch = _OPP.archetype(int(pid))
                    print(f"           Player {pid} ({arch}): "
                        f"{info['cards']} ({info['hand']})")
                print(f"           Stacks: {msg['stacks']}")
                self.history.append(msg)

            elif t == "winner":
                print(f"[{self.name}] Player {msg['pid']} wins "
                    f"pot={msg['pot']} ({msg['reason']})")
                print(f"           Stacks: {msg['stacks']}")
                self.history.append(msg)       # opponent model reads this

            elif t == "game_over":
                print(f"[{self.name}] GAME OVER — winner: player {msg['winner']}")
                break

            else:
                print(f"[{self.name}] MSG: {msg}")

    def _build_response(self, action, state: GameState) -> dict:
        if isinstance(action, tuple):
            verb, amount = action
            amount = max(int(amount), state.min_raise)
            amount = min(amount, state.chips + state.current_bet)
            return {"action": verb, "amount": amount}

        action = action.lower()

        if action == "check":
            if not state.can_check:
                print(f"[{self.name}] WARNING: tried check, must call — folding")
                return {"action": "fold"}
            return {"action": "check"}

        if action == "allin":
            return {"action": "allin",
                    "amount": state.chips + state.current_bet}

        if action in ("fold", "call"):
            return {"action": action}

        print(f"[{self.name}] WARNING: unknown action '{action}' — folding")
        return {"action": "fold"}


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Poker Bot")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--name", default="Bot")
    args = ap.parse_args()

    bot = BotClient(args.host, args.port, args.name)
    bot.run()
