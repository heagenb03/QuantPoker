"""
poker/simulation.py — Monte Carlo simulation and SimResult.

Three-step unified Monte Carlo:
Step 1: full distribution (equity + variance + CVaR + risk_adj_equity)
Step 2: continuation value (draw premium per street — Longstaff-Schwartz intuition)
Step 3: per-action EV using opponent archetype fold/bet probabilities
"""

import random
from collections import Counter
from dataclasses import dataclass

from poker.constants import ALL_CARDS
from poker.eval import _eval
from poker.config import (
    N_SIMS, CVAR_TAIL, LAMBDA_MIN, LAMBDA_MAX, LAMBDA_BASE, LAMBDA_DECAY,
    LAMBDA_REF_BB, IMPROVE_THRESH, DRAW_PREMIUM_MIN, DRAW_PREMIUM_MAX,
    DRAW_DISCOUNT_FLOP, DRAW_DISCOUNT_TURN, BET_HALF_FRAC, BET_FULL_FRAC,
    IMPLIED_OPP_BET_FRAC, MIN_CV_TO_CALL, FOLD_TO_BET, BET_WHEN_CHECKED,
    AGGRESSION_DISCOUNT_FACTOR,
)


# ── Archetype response tables (used in Steps 1 and 3) ─────────────
# These are imported from config.py — edit there to adjust bot behavior.
_FOLD_TO_BET     = FOLD_TO_BET
_BET_WHEN_CHECKED = BET_WHEN_CHECKED


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


def run_unified_simulation(state: 'GameState', bb: int,
                           n_sims: int = N_SIMS) -> SimResult:
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
    import bot  # reads bot.NUM_DECKS, bot._OPP

    street      = state.street
    hole_cards  = state.hole_cards
    board       = state.community
    n_opponents = state.active_opponents
    stack_bb    = state.chips / bb

    # ── Risk aversion (lambda): scales with stack depth ────────────
    # Quant parallel: fractional Kelly. Deep stacks tolerate variance;
    # short stacks must minimize it — ruin is permanent in a freezeout.
    lambda_ = max(LAMBDA_MIN, min(LAMBDA_MAX, LAMBDA_BASE - (stack_bb - LAMBDA_REF_BB) * LAMBDA_DECAY))

    # ── Opponent archetypes for Steps 2 and 3 ─────────────────────
    opp_pids = [
        int(p) for p, folded in state.player_folded.items()
        if not folded and int(p) != state.my_pid
    ]
    archetypes = [bot._OPP.archetype(pid) for pid in opp_pids]

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
    full_deck = ALL_CARDS * bot.NUM_DECKS
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

    # CVaR: mean of worst CVAR_TAIL of outcomes (tail risk, not just spread)
    # Quant parallel: Expected Shortfall — what we lose in bad scenarios
    cvar_n   = max(1, int(n_sims * CVAR_TAIL))
    cvar     = sum(sorted(outcomes)[:cvar_n]) / cvar_n

    # Risk-adjusted equity: penalize variance by stack-depth-scaled lambda
    risk_adj_equity = equity - lambda_ * variance

    # ── Step 2: draw premium and continuation value ────────────────
    # Draw path: terminal hand is meaningfully stronger than current board.
    # eval7: higher score = stronger hand (opposite of server convention).
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
        draw_premium = max(DRAW_PREMIUM_MIN, min(DRAW_PREMIUM_MAX, raw_premium))

        # Street discount: option value decays as fewer cards remain
        # Quant parallel: theta decay — options lose value as expiry approaches
        discount = {'flop': DRAW_DISCOUNT_FLOP, 'turn': DRAW_DISCOUNT_TURN}.get(street, 1.0)
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
    # (all must fold for a bet to win the pot uncontested).
    # Discount each opponent's fold estimate by their in-hand aggression:
    # each raise/bet/allin they have shown this hand multiplies by
    # AGGRESSION_DISCOUNT_FACTOR, reflecting that demonstrated strength
    # reduces their likelihood of folding to our bet.
    p_all_fold = 1.0
    for pid in opp_pids:
        base_fold = bot._OPP.fold_to_bet_est(pid)
        agg = bot._OPP.in_hand_aggression(pid)
        p_all_fold *= base_fold * (AGGRESSION_DISCOUNT_FACTOR ** agg)

    # Probability at least one opponent bets when checked to
    p_no_bet = 1.0
    for pid in opp_pids:
        p_no_bet *= (1.0 - bot._OPP.bet_when_checked_est(pid))
    p_someone_bets = 1.0 - p_no_bet

    bet_half  = max(state.min_raise, int(pot * BET_HALF_FRAC))
    bet_full  = max(state.min_raise, int(pot * BET_FULL_FRAC))
    shove_amt = state.chips + state.current_bet

    # Helper: EV of betting a fixed size (fold equity + call equity)
    def _ev_bet(size: int) -> float:
        return (p_all_fold * pot
                + (1.0 - p_all_fold)
                * (continuation_value * (pot + size) - size))

    ev_by_action: dict = {'fold': 0.0}

    if state.can_check:
        # Assume opponent bets ~half pot when they bet (conservative)
        implied_bet = max(1, int(pot * IMPLIED_OPP_BET_FRAC))
        new_pot_if_bet = pot + implied_bet * 2
        if continuation_value > MIN_CV_TO_CALL:   # worth calling their bet
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
    import bot
    if n_opponents <= 0:
        return 1.0
    full_deck = ALL_CARDS * bot.NUM_DECKS
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
