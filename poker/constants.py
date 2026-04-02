"""
poker/constants.py — Immutable card constants shared across all modules.
"""

RANKS     = '23456789TJQKA'
SUITS     = 'cdhs'
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
RANK_IDX  = {r: i for i, r in enumerate(RANKS)}    # '2'→0 … 'A'→12
