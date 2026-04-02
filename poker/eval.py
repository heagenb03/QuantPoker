"""
poker/eval.py — Hand evaluation wrapper.

Multi-deck aware: eval7 fast path for single-deck (no duplicates possible);
custom rank-counting evaluator for multi-deck (where duplicates can occur).
"""

from collections import Counter
from itertools import combinations as _combinations

_RANK_ORDER = {r: i + 2 for i, r in enumerate('23456789TJQKA')}  # '2'→2 … 'A'→14


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
        import bot
        if bot.NUM_DECKS == 1:
            return _ev7.evaluate([_ev7.Card(c) for c in card_strings])
        return _eval_multideck(card_strings)
    print("[BOT] eval7 loaded ✓")
except ImportError:
    print("[BOT] WARNING: eval7 not found.  Run:  pip install eval7")
    def _eval(card_strings: list) -> int:
        import bot
        from poker.constants import RANK_IDX
        if bot.NUM_DECKS == 1:
            # Naive rank-sum fallback — functional but weak. Install eval7!
            return sum(RANK_IDX[c[0]] for c in card_strings)
        return _eval_multideck(card_strings)
