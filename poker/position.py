"""
poker/position.py — Position calculation helpers.

Position = information asymmetry.
Late position (acting last) ↔ latency advantage in e-trading.
"""


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
