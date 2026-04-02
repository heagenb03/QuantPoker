"""
poker/config.py — All tunable bot parameters in one place.

Edit this file to adjust bot behavior without touching logic code.
Parameters are grouped by the module that uses them.
"""

# ── Simulation ─────────────────────────────────────────────────────
N_SIMS = 100000    # Monte Carlo rollouts per decision

# CVaR tail fraction — fraction of worst outcomes averaged for tail risk
CVAR_TAIL = 0.20              # 0.20 = worst 20%

# Risk aversion (lambda) — scales with stack depth
# Deep (≥ LAMBDA_REF_BB + enough) → LAMBDA_MIN (nearly risk-neutral)
# Short (≤ LAMBDA_REF_BB)         → LAMBDA_MAX (variance is existential)
LAMBDA_MIN      = 0.05
LAMBDA_MAX      = 0.28
LAMBDA_BASE     = 0.30        # intercept of the linear formula
LAMBDA_DECAY    = 0.008       # chips-per-BB slope
LAMBDA_REF_BB   = 15          # stack_bb anchor for decay

# Draw premium — continuation value adjustment for drawing hands
IMPROVE_THRESH      = 1.08    # terminal score must be 8% stronger to count as improvement
DRAW_PREMIUM_MIN    = -0.12   # cap: made hand being drawn out on
DRAW_PREMIUM_MAX    =  0.18   # cap: strong drawing hand
DRAW_DISCOUNT_FLOP  = 0.70    # theta decay: flop has most option value
DRAW_DISCOUNT_TURN  = 0.90    # turn: one card left, less value
# River discount = 1.0 (no decay needed, implied by absence)

# Bet sizing fractions (× pot)
BET_HALF_FRAC = 0.50
BET_FULL_FRAC = 1.00

# Assumed opponent bet size (× pot) when modelling check EV — conservative estimate
IMPLIED_OPP_BET_FRAC = 0.50

# Minimum continuation value to call an opponent's bet when checking
MIN_CV_TO_CALL = 0.25

# Archetype fold/bet tables — seed Bayesian Beta priors
# These are also the fallback when zero observations exist.
FOLD_TO_BET = {
    'nit':     0.65,
    'tag':     0.45,
    'station': 0.08,
    'maniac':  0.20,
    'unknown': 0.40,
}
BET_WHEN_CHECKED = {
    'nit':     0.20,
    'tag':     0.45,
    'station': 0.35,
    'maniac':  0.75,
    'unknown': 0.35,
}

# ── Postflop decisions ─────────────────────────────────────────────
# Override Rule 1: never bluff a station unless value threshold met
STATION_BET_EQUITY      = 0.52   # min risk_adj_equity to bet vs station

# Override Rule 2: trap maniacs — check with strong hands, let them bet
MANIAC_TRAP_EQUITY      = 0.65   # min risk_adj_equity to trap
MANIAC_TRAP_EV_BOOST    = 1.15   # multiply best-bet EV to force check choice

# Override Rule 3: bluff gate — all must hold to allow a bluff
BLUFF_FREQ              = 0.25   # random frequency cap (0.25 = bluff 25% of spots)
BLUFF_DRAW_PREMIUM_MAX  = 0.02   # must not be on a draw when bluffing
BET_GATE_EQUITY         = 0.42   # min risk_adj_equity to bet at all (no bluff ok)

# Override Rule 4: pot commitment — call regardless of EV
POT_COMMITMENT_RATIO    = 0.60   # if to_call ≥ chips × this, always call

# ── Preflop decisions ──────────────────────────────────────────────
PUSH_FOLD_STACK_BB      = 15     # stack_bb below this → push/fold only

# Push/fold range formula: max(PUSH_RANGE_FLOOR, BASE - pos_decay - table_decay)
PUSH_RANGE_FLOOR        = 0.12
PUSH_RANGE_BASE         = 0.52
PUSH_RANGE_POS_DECAY    = 0.28   # BTN loosens, UTG tightens
PUSH_RANGE_TABLE_DECAY  = 0.02   # extra players beyond TABLE_DECAY_START tighten range
PUSH_RANGE_TABLE_START  = 6      # table size at which table_decay kicks in
PUSH_FACING_SHOVE_MULT  = 1.25   # tighten threshold when facing a shove

OPEN_SIZE_BB            = 3      # open raise sizing in BBs
MANIAC_TIGHTEN_MULT     = 1.35   # require strength × this to open vs maniacs

# Facing a 3-bet
VS_3BET_SHOVE_SHORT     = 0.55   # shove threshold when stack < VS_3BET_SHORT_BB
VS_3BET_SHOVE_DEEP      = 0.65   # 4-bet threshold vs TAG/unknown: JJ+, AKs
VS_3BET_SHORT_BB        = 25     # stack_bb cutoff for short vs deep 3-bet response
VS_3BET_RAISE_MULT      = 3      # 4-bet sizing = to_call × this

# Archetype-adjusted 3-bet defense (deep stack only — short stack always shoves/folds)
# Maniac 3-bets wide → defend more aggressively
VS_3BET_SHOVE_DEEP_VS_MANIAC = 0.52  # 4-bet vs maniac: TT+, AK, AQs
VS_3BET_CALL_DEEP_VS_MANIAC  = 0.44  # call vs maniac: 88-99, AJo, KQs
VS_3BET_CALL_DEEP            = 0.58  # call vs TAG/unknown: AKo, TT, AQs (below 4-bet tier)

# Speculative call vs 3-bet (set-mine / suited connector in position)
SPECULATIVE_STRENGTH    = 0.12
SPECULATIVE_POT_ODDS    = 0.28   # only call if pot_odds < this

# Facing a single raise — call thresholds by opponent archetype
CALL_THRESH_DEFAULT     = 0.25
CALL_THRESH_VS_NIT      = 0.30
CALL_THRESH_VS_MANIAC   = 0.18
CALL_THRESH_VS_STATION  = 0.22

# 3-bet squeeze vs single raise
SQUEEZE_STRENGTH_MULT   = 1.5    # strength ≥ call_thresh × this to squeeze
SQUEEZE_RAISE_MULT      = 3      # squeeze sizing = to_call × this
MIN_STACK_TO_SQUEEZE    = 20     # stack_bb must exceed this to squeeze

# Max pot odds to call a raise (above this → fold)
POT_ODDS_CALL_MAX       = 0.35

# ── Position / open ranges ─────────────────────────────────────────
OPEN_RANGE_BASE         = 0.48   # UTG open range at 2-player-behind baseline
OPEN_RANGE_DECAY        = 0.85   # per-extra-player-behind multiplier (~15% per player)
OPEN_RANGE_FLOOR        = 0.06   # always open at least the top 6% of hands
SB_OPEN_PCT             = 0.38   # SB: 1 player behind
BB_OPEN_PCT             = 0.30   # BB: closing action (usually check or defend)

# ── Opponent model ─────────────────────────────────────────────────
MIN_HANDS_FOR_ARCHETYPE = 10     # hands seen before trusting archetype read
MIN_HANDS_FOR_STATS     = 5      # hands seen before trusting VPIP/PFR/AF numbers

# Default stats before enough data is collected
DEFAULT_VPIP            = 0.25
DEFAULT_PFR             = 0.12
DEFAULT_AF              = 1.0

# Archetype VPIP / AF classification thresholds
NIT_VPIP_MAX            = 0.18
STATION_VPIP_MIN        = 0.35
STATION_AF_MAX          = 1.5
MANIAC_VPIP_MIN         = 0.30
MANIAC_AF_MIN           = 2.5

# Estimated hand ranges per archetype (used by range_width())
ARCHETYPE_RANGE_WIDTH = {
    'nit':     0.12,
    'tag':     0.22,
    'station': 0.45,
    'maniac':  0.38,
    'unknown': 0.25,
}

# Bayesian Beta prior pseudocount (worth this many observations)
BETA_PSEUDOCOUNT        = 5.0
# Weak prior applied on first postflop observation before archetype stabilises
BETA_PSEUDOCOUNT_INITIAL = 2.0
# Per in-hand aggressive action: fold_est *= this (0.65^2 ≈ 0.42 after 3-bet + c-bet)
AGGRESSION_DISCOUNT_FACTOR = 0.65
