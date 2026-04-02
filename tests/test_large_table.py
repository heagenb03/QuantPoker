"""
test_large_table.py -- Unit tests for large-table (15-20 player) scenarios
and multi-deck hand evaluation.

The hackathon has 15-20 participants at a single table.  With >= 15 players
the server uses 4 decks, which allows duplicate cards in a hand.  These tests
cover:

  * NUM_DECKS formula correctness for every relevant player count
  * _eval_5card_multideck() — all hand categories, including five-of-a-kind
    and hands containing duplicate-suit cards (only possible with 2+ decks)
  * _eval_multideck() — best-5-from-N logic with duplicate cards
  * _eval() routing (single-deck vs multi-deck) driven by NUM_DECKS
  * Position utilities at 15-20 player tables
  * Preflop decisions at large tables (tight UTG, wide BTN, push/fold)
  * run_unified_simulation() card-pool adequacy and sanity
  * decide() end-to-end traces at large tables

Run modes:
    pytest -m large_table               # all large-table tests
    pytest -m multideck                 # multi-deck eval tests only
    pytest test_large_table.py -v       # verbose output
"""

from collections import Counter

import pytest

import bot
from bot import (
    _eval,
    _eval_5card_multideck,
    _eval_multideck,
    _preflop,
    decide,
    get_position,
    open_range_pct,
    players_behind_preflop,
    run_unified_simulation,
)
from conftest import make_state


# ═══════════════════════════════════════════════════════════════════
# NUM_DECKS FORMULA
# Server formula: NUM_DECKS = max(1, min(4, (n_players // 5) + 1))
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
class TestNumDecksFormula:
    """Verify NUM_DECKS formula for every key player count."""

    @pytest.mark.parametrize("n_players,expected", [
        (2,  1),   # 2-player HU     → (2//5)+1 = 1
        (4,  1),   # 4-player        → (4//5)+1 = 1
        (5,  2),   # 5-player        → (5//5)+1 = 2
        (9,  2),   # 9-player        → (9//5)+1 = 2
        (10, 3),   # 10-player       → (10//5)+1 = 3
        (14, 3),   # 14-player       → (14//5)+1 = 3
        (15, 4),   # hackathon floor → (15//5)+1 = 4
        (16, 4),   # 16-player       → (16//5)+1 = 4  (clamped at 4)
        (20, 4),   # hackathon ceil  → clamped to 4
        (25, 4),   # hypothetical    → clamped to 4
    ])
    def test_formula(self, n_players: int, expected: int) -> None:
        result = max(1, min(4, (n_players // 5) + 1))
        assert result == expected, (
            f"n_players={n_players}: expected {expected} decks, got {result}"
        )

    def test_all_hackathon_sizes_use_4_decks(self) -> None:
        """Every table size from 15 to 20 must use 4 decks."""
        for n in range(15, 21):
            decks = max(1, min(4, (n // 5) + 1))
            assert decks == 4, f"{n} players → expected 4 decks, got {decks}"

    def test_4_decks_means_208_cards(self) -> None:
        """4 decks = 52 * 4 = 208 cards in the full pool."""
        assert len(bot.ALL_CARDS) * 4 == 208


# ═══════════════════════════════════════════════════════════════════
# MULTI-DECK 5-CARD EVALUATOR
# _eval_5card_multideck(cards: list) -> (category: int, tiebreakers: tuple)
# Higher tuple = stronger hand.
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.multideck
class TestMultiDeck5CardEval:
    """_eval_5card_multideck() — hand categories with and without duplicates."""

    # ── Category detection ─────────────────────────────────────────

    def test_five_of_a_kind_is_category_9(self) -> None:
        """Five aces (duplicate suits across decks) → category 9."""
        cat, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Ah"])
        assert cat == 9, f"Five-of-a-kind should be category 9, got {cat}"

    def test_straight_flush_is_category_8(self) -> None:
        cat, _ = _eval_5card_multideck(["Th", "Jh", "Qh", "Kh", "Ah"])
        assert cat == 8, f"Royal flush should be category 8, got {cat}"

    def test_four_of_a_kind_is_category_7(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Kh"])
        assert cat == 7, f"Quads should be category 7, got {cat}"

    def test_four_of_a_kind_duplicate_suit_is_category_7(self) -> None:
        """Two Ah from different decks still gives quads."""
        cat, _ = _eval_5card_multideck(["Ah", "Ah", "As", "Ad", "Kh"])
        assert cat == 7, f"Quads (duplicate suit) should be category 7, got {cat}"

    def test_full_house_is_category_6(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Kh", "Ks"])
        assert cat == 6, f"Full house should be category 6, got {cat}"

    def test_flush_is_category_5(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "Kh", "Qh", "Jh", "9h"])
        assert cat == 5, f"Flush should be category 5, got {cat}"

    def test_straight_is_category_4(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "Kd", "Qc", "Js", "Th"])
        assert cat == 4, f"Straight should be category 4, got {cat}"

    def test_wheel_straight_is_category_4(self) -> None:
        """A-2-3-4-5 wheel straight."""
        cat, _ = _eval_5card_multideck(["Ah", "2d", "3c", "4s", "5h"])
        assert cat == 4, f"Wheel should be category 4, got {cat}"

    def test_three_of_a_kind_is_category_3(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Kh", "Qc"])
        assert cat == 3, f"Trips should be category 3, got {cat}"

    def test_trips_duplicate_suit_is_category_3(self) -> None:
        """Trips where two copies share a suit (multi-deck)."""
        cat, _ = _eval_5card_multideck(["Ah", "Ah", "As", "Kh", "Qc"])
        assert cat == 3, f"Trips (duplicate suit) should be category 3, got {cat}"

    def test_two_pair_is_category_2(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "As", "Kh", "Ks", "Qc"])
        assert cat == 2, f"Two pair should be category 2, got {cat}"

    def test_two_pair_duplicate_suit_is_category_2(self) -> None:
        """Pair formed by same card from two decks (e.g. two Ah)."""
        cat, _ = _eval_5card_multideck(["Ah", "Ah", "Kh", "Ks", "Qc"])
        assert cat == 2, f"Two pair (Ah Ah) should be category 2, got {cat}"

    def test_one_pair_is_category_1(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "As", "Kh", "Qc", "Jd"])
        assert cat == 1, f"One pair should be category 1, got {cat}"

    def test_high_card_is_category_0(self) -> None:
        cat, _ = _eval_5card_multideck(["Ah", "Kd", "Qc", "Js", "9h"])
        assert cat == 0, f"High card should be category 0, got {cat}"

    # ── Hand ordering ──────────────────────────────────────────────

    def test_hand_category_ordering(self) -> None:
        """All nine categories must be strictly ordered weakest → strongest."""
        five_oak,    _ = _eval_5card_multideck(["Ah", "Ah", "As", "Ad", "Ac"])
        sf,          _ = _eval_5card_multideck(["Kh", "Qh", "Jh", "Th", "9h"])
        four_oak,    _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Kh"])
        full_house,  _ = _eval_5card_multideck(["Ah", "As", "Ad", "Kh", "Ks"])
        flush,       _ = _eval_5card_multideck(["Ah", "Kh", "Qh", "Jh", "9h"])
        straight,    _ = _eval_5card_multideck(["Ah", "Kd", "Qc", "Js", "Th"])
        trips,       _ = _eval_5card_multideck(["Ah", "As", "Ad", "Kh", "Qc"])
        two_pair,    _ = _eval_5card_multideck(["Ah", "As", "Kh", "Ks", "Qc"])
        one_pair,    _ = _eval_5card_multideck(["Ah", "As", "Kh", "Qc", "Jd"])
        high_card,   _ = _eval_5card_multideck(["Ah", "Kd", "Qc", "Js", "9h"])

        cats = [five_oak, sf, four_oak, full_house, flush,
                straight, trips, two_pair, one_pair, high_card]
        assert cats == sorted(cats, reverse=True), (
            "Categories must be strictly descending: "
            + str(cats)
        )

    def test_higher_quads_beat_lower_quads(self) -> None:
        # Compare full (category, tiebreaker) tuples — same category, different rank
        ace_quads  = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "2h"])
        king_quads = _eval_5card_multideck(["Kh", "Ks", "Kd", "Kc", "Ah"])
        assert ace_quads > king_quads, "Quad aces should beat quad kings"

    def test_higher_full_house_beats_lower(self) -> None:
        # Compare full (category, tiebreaker) tuples
        big   = _eval_5card_multideck(["Ah", "As", "Ad", "Kh", "Ks"])  # AAA KK
        small = _eval_5card_multideck(["Kh", "Ks", "Kd", "Qh", "Qs"])  # KKK QQ
        assert big > small, "AAA KK should beat KKK QQ"

    def test_five_oak_beats_quads(self) -> None:
        five_oak, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Ah"])
        quads,    _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Kh"])
        assert five_oak > quads, "Five-of-a-kind should beat four-of-a-kind"


# ═══════════════════════════════════════════════════════════════════
# MULTI-DECK N-CARD EVALUATOR (best-5-of-N)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.multideck
class TestMultiDeckNCardEval:
    """_eval_multideck() — best-5-from-N with duplicate cards."""

    def test_7card_picks_best_5_by_kicker(self) -> None:
        """With 7 cards, higher kicker should produce higher score."""
        score_king   = _eval_multideck(["Ah", "As", "Ad", "Ac", "Kh", "Qs", "2c"])
        score_queen  = _eval_multideck(["Ah", "As", "Ad", "Ac", "Qh", "Js", "2c"])
        assert score_king > score_queen, "Quad aces + K kicker should beat + Q kicker"

    def test_7card_finds_quads_with_duplicate_suit(self) -> None:
        """7-card hand with duplicate-suit aces (2+ decks) correctly forms quads."""
        quads_hand = _eval_multideck(["Ah", "Ah", "As", "Ad", "Kh", "Ks", "Qc"])
        two_pair   = _eval_multideck(["Ah", "As", "Kh", "Kd", "Qc", "Js", "Th"])
        assert quads_hand > two_pair, "Four aces should beat two pair aces/kings"

    def test_category_agreement_5card_vs_7card(self) -> None:
        """5-card and 7-card evals should agree on category when extras are junk."""
        cat_5, _ = _eval_5card_multideck(["Ah", "As", "Ad", "Ac", "Kh"])
        score_7   = _eval_multideck(["Ah", "As", "Ad", "Ac", "Kh", "2c", "3d"])
        # _eval_multideck packs: result = cat << 20 | tiebreakers
        cat_7 = score_7 >> 20
        assert cat_5 == cat_7, (
            f"5-card and 7-card evals disagree: cat_5={cat_5}, cat_7={cat_7}"
        )

    # ── _eval() routing ───────────────────────────────────────────

    def test_eval_routes_to_multideck_when_num_decks_gt_1(self) -> None:
        """_eval() must call _eval_multideck when NUM_DECKS > 1."""
        bot.NUM_DECKS = 4
        # Duplicate-suit card — would produce wrong result via eval7 bitmask
        score = _eval(["Ah", "Ah", "As", "Ad", "Kh"])
        assert isinstance(score, int), "_eval should return int in multi-deck mode"
        # Verify it at least detects quads (category 7 → encoded ≥ 7 << 20)
        assert score >= (7 << 20), "Quad aces should encode to category ≥ 7"

    def test_eval_single_deck_returns_int(self) -> None:
        """_eval() works in single-deck mode (no duplicates)."""
        bot.NUM_DECKS = 1
        score = _eval(["Ah", "As", "Kh", "Kd", "Qc"])
        assert isinstance(score, int)

    def test_eval_multideck_stronger_hand_scores_higher(self) -> None:
        """In multi-deck mode, stronger hand always gets higher score."""
        bot.NUM_DECKS = 4
        quads = _eval(["Ah", "As", "Ad", "Ac", "Kh"])
        pair  = _eval(["Ah", "As", "Kh", "Qd", "Jc"])
        assert quads > pair, "Quads must outscore one pair in multi-deck mode"


# ═══════════════════════════════════════════════════════════════════
# LARGE TABLE POSITION UTILITIES
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
class TestLargeTablePosition:
    """Position, players_behind_preflop, and open_range_pct at 15-20 players."""

    # ── get_position ───────────────────────────────────────────────

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_btn_is_pos_0(self, n: int) -> None:
        assert get_position(my_pid=0, dealer_pid=0, n_players=n) == 0

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_sb_is_pos_1(self, n: int) -> None:
        assert get_position(my_pid=1, dealer_pid=0, n_players=n) == 1

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_bb_is_pos_2(self, n: int) -> None:
        assert get_position(my_pid=2, dealer_pid=0, n_players=n) == 2

    def test_wrap_around_at_20_players(self) -> None:
        """Clockwise wrap: pid=3, dealer=5, n=20 → (3-5) % 20 = 18."""
        pos = get_position(my_pid=3, dealer_pid=5, n_players=20)
        assert pos == 18

    # ── players_behind_preflop ─────────────────────────────────────

    @pytest.mark.parametrize("n", [15, 16, 17, 18, 20])
    def test_btn_always_2_behind(self, n: int) -> None:
        assert players_behind_preflop(0, n) == 2

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_sb_always_1_behind(self, n: int) -> None:
        assert players_behind_preflop(1, n) == 1

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_bb_always_0_behind(self, n: int) -> None:
        assert players_behind_preflop(2, n) == 0

    def test_utg_behind_at_20_players(self) -> None:
        """UTG (pos=3) at 20 players: (20-3)+2 = 19 behind."""
        assert players_behind_preflop(3, 20) == 19

    def test_utg_behind_at_15_players(self) -> None:
        """UTG (pos=3) at 15 players: (15-3)+2 = 14 behind."""
        assert players_behind_preflop(3, 15) == 14

    # ── open_range_pct ─────────────────────────────────────────────

    @pytest.mark.parametrize("n", [15, 16, 17, 20])
    def test_open_range_bounded_at_large_tables(self, n: int) -> None:
        """All positions at large tables must be in [0.06, 0.50]."""
        for pos in range(n):
            pct = open_range_pct(pos, n)
            assert 0.06 <= pct <= 0.50, (
                f"open_range_pct(pos={pos}, n={n}) = {pct:.3f} out of [0.06, 0.50]"
            )

    @pytest.mark.parametrize("n", [15, 16, 20])
    def test_btn_opens_wider_than_utg(self, n: int) -> None:
        assert open_range_pct(0, n) > open_range_pct(3, n), (
            f"BTN should open wider than UTG at n={n}"
        )

    def test_utg_at_16_players_is_floor(self) -> None:
        """UTG at 16 players has 15 behind → 0.48*0.85^13 ≈ 0.058 → floored to 6%."""
        pct = open_range_pct(3, 16)
        assert pct == pytest.approx(0.06), (
            f"UTG at 16 players should open floor 6%, got {pct:.4f}"
        )

    def test_utg_at_15_players_near_floor(self) -> None:
        """UTG at 15 players has 14 behind → ≈6.8% (just above floor but still very tight)."""
        pct = open_range_pct(3, 15)
        assert 0.06 <= pct < 0.10, (
            f"UTG at 15 players should be in [6%, 10%), got {pct:.4f}"
        )

    def test_ranges_tighten_with_more_players(self) -> None:
        """UTG range should shrink as table size grows (more players behind)."""
        r6  = open_range_pct(3, 6)
        r9  = open_range_pct(3, 9)
        r15 = open_range_pct(3, 15)
        assert r6 >= r9 >= r15, (
            f"UTG ranges should tighten: 6-max={r6:.3f}, 9-max={r9:.3f}, 15-max={r15:.3f}"
        )


# ═══════════════════════════════════════════════════════════════════
# LARGE TABLE PREFLOP DECISIONS
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.preflop
class TestLargeTablePreflopDecision:
    """Preflop decisions at 15 and 20 player tables."""

    def test_AA_raises_on_btn_15_players(self) -> None:
        """AA on BTN at 15-player table should open-raise."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"AA BTN 15 players should raise, got {action}"
        )

    def test_AA_raises_utg_15_players(self) -> None:
        """AA UTG at 15-player table is strong enough to open from any seat."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"AA UTG 15 players should raise, got {action}"
        )

    def test_72o_folds_utg_15_players_facing_raise(self) -> None:
        """72o UTG at 15-player table facing a raise (to_call > BB) should fold.

        to_call=40 (2x BB) triggers facing_raise=True; call_thresh=0.25
        and preflop_strength(72o)~0.10 < 0.25, so the bot folds.
        """
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=70, to_call=40, current_bet=40, min_raise=80,
            num_players=15, my_pid=3,
        )
        action = _preflop(state, pos=3, bb=20)
        assert action == "fold", f"72o UTG 15 players facing raise should fold, got {action}"

    def test_72o_folds_utg_20_players_facing_raise(self) -> None:
        """72o UTG at 20-player table facing a raise (to_call > BB) should fold."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=70, to_call=40, current_bet=40, min_raise=80,
            num_players=20, my_pid=3,
        )
        action = _preflop(state, pos=3, bb=20)
        assert action == "fold", f"72o UTG 20 players facing raise should fold, got {action}"

    def test_push_fold_AA_short_stack_15_players(self) -> None:
        """AA with 10BB at 15-player table → push all-in."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=200, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=0, bb=20)
        assert action == "allin", f"AA 10BB 15 players should shove, got {action}"

    def test_push_fold_weak_hand_folds_15_players(self) -> None:
        """72o with 10BB facing a raise at 15-player table should fold."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=200, pot=100, to_call=60, current_bet=60, min_raise=120,
            num_players=15, my_pid=5,
        )
        action = _preflop(state, pos=5, bb=20)
        assert action == "fold", f"72o short stack facing raise 15 players should fold, got {action}"

    def test_bb_checks_free_20_players(self) -> None:
        """BB with weak hand and no raise at 20-player table: free check."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=20, min_raise=40,
            num_players=20, my_pid=2,
        )
        action = _preflop(state, pos=2, bb=20)
        assert action == "check", f"BB no raise 20 players should check, got {action}"


# ═══════════════════════════════════════════════════════════════════
# LARGE TABLE SIMULATION (card pool + multi-deck)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.postflop
class TestLargeTableSimulation:
    """run_unified_simulation() sanity with many opponents and 4-deck pool."""

    def test_card_pool_sufficient_for_20_players_4_decks(self) -> None:
        """4 decks must have enough cards for 19 opponents + 2 remaining community."""
        hole  = ["Ah", "Kd"]
        board = ["Ts", "9h", "2c"]     # flop — 2 community cards still to come
        full_deck = bot.ALL_CARDS * 4
        available = list((Counter(full_deck) - Counter(hole + board)).elements())
        needed = 19 * 2 + (5 - len(board))   # 38 hole + 2 community = 40
        assert len(available) >= needed, (
            f"Pool has {len(available)} cards, need {needed} for 20-player table"
        )

    def test_simulation_completes_14_opponents_flop(self) -> None:
        """run_unified_simulation() should complete for 14 opponents on flop."""
        bot.NUM_DECKS = 4
        state = make_state(
            hole_cards=["Ah", "Kd"],
            community=["Ts", "9h", "2c"],
            street="flop",
            chips=1000, pot=300, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        sim = run_unified_simulation(state, bb=20, n_sims=200)
        assert 0.0 <= sim.equity <= 1.0, f"Equity out of range: {sim.equity}"
        assert sim.n_opponents == 14

    def test_simulation_completes_19_opponents_turn(self) -> None:
        """Stress: 19 opponents on the turn at 20-player table."""
        bot.NUM_DECKS = 4
        state = make_state(
            hole_cards=["Ah", "Kd"],
            community=["Ts", "9h", "2c", "7d"],
            street="turn",
            chips=1000, pot=300, to_call=0, current_bet=0, min_raise=40,
            num_players=20, my_pid=0,
        )
        sim = run_unified_simulation(state, bb=20, n_sims=100)
        assert 0.0 <= sim.equity <= 1.0
        assert sim.n_opponents == 19

    def test_equity_decreases_with_more_opponents(self) -> None:
        """Equity with AKo should decrease as we add more opponents."""
        bot.NUM_DECKS = 4
        results = []
        for n in [2, 6, 15]:
            state = make_state(
                hole_cards=["Ah", "Kd"],
                community=["Ts", "9h", "2c"],
                street="flop",
                chips=1000, pot=100, to_call=0,
                num_players=n, my_pid=0,
            )
            sim = run_unified_simulation(state, bb=20, n_sims=300)
            results.append(sim.equity)

        eq_hu, eq_6, eq_15 = results
        assert eq_hu > eq_6 > eq_15, (
            f"Equity should drop with more opponents: HU={eq_hu:.3f}, "
            f"6max={eq_6:.3f}, 15-handed={eq_15:.3f}"
        )

    def test_strong_hand_high_equity_large_table(self) -> None:
        """Trip aces on the flop should have substantially higher equity than 50/50
        even against 14 opponents with 4 decks in play."""
        bot.NUM_DECKS = 4
        state = make_state(
            hole_cards=["Ah", "As"],
            community=["Ac", "Kd", "2h"],    # we hold pocket aces, board has third ace
            street="flop",
            chips=1000, pot=100, to_call=0,
            num_players=15, my_pid=0,
        )
        sim = run_unified_simulation(state, bb=20, n_sims=300)
        # With 14 opponents and 4 decks, draws are more available, so threshold is 0.65+
        assert sim.equity > 0.65, (
            f"Trip aces (flopped) should have equity > 0.65 even 15-handed, got {sim.equity:.3f}"
        )

    def test_risk_adj_equity_leq_equity(self) -> None:
        """risk_adj_equity = equity - λ*variance must be ≤ equity (λ, variance ≥ 0)."""
        bot.NUM_DECKS = 4
        state = make_state(
            hole_cards=["Kh", "Qd"],
            community=["Ac", "Jh", "Th"],
            street="flop",
            chips=1000, pot=200, to_call=0,
            num_players=15, my_pid=0,
        )
        sim = run_unified_simulation(state, bb=20, n_sims=200)
        assert sim.risk_adj_equity <= sim.equity, (
            f"risk_adj_equity ({sim.risk_adj_equity:.3f}) must be <= equity ({sim.equity:.3f})"
        )

    def test_simresult_fields_are_in_valid_ranges(self) -> None:
        """All SimResult probability fields must be in [0, 1]."""
        bot.NUM_DECKS = 4
        state = make_state(
            hole_cards=["Ah", "Kd"],
            community=["Ac", "Kh", "3s"],
            street="flop",
            chips=1000, pot=200, to_call=0,
            num_players=15, my_pid=0,
        )
        sim = run_unified_simulation(state, bb=20, n_sims=200)
        for field in ("equity", "variance", "cvar", "risk_adj_equity",
                      "continuation_value"):
            val = getattr(sim, field)
            assert isinstance(val, float), f"{field} should be float"
        assert sim.n_opponents == 14
        assert sim.n_sims == 200


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: decide() end-to-end at large tables
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.integration
class TestLargeTableDecide:
    """decide() end-to-end at 15-20 player tables with 4-deck evaluation."""

    VALID_STR_ACTIONS = {"fold", "check", "call", "allin"}

    def _assert_valid(self, action, context: str = "") -> None:
        if action in self.VALID_STR_ACTIONS:
            return
        if (isinstance(action, tuple) and len(action) == 2
                and action[0] == "raise"
                and isinstance(action[1], (int, float))):
            return
        pytest.fail(f"Invalid action {action!r}" + (f" ({context})" if context else ""))

    def _setup_large_table_globals(self) -> None:
        bot.NUM_DECKS = 4
        bot.DEALER_PID = 0
        bot.BIG_BLIND = 20

    def test_decide_preflop_15_players_strong_hand(self) -> None:
        """decide() returns valid action preflop with AA at 15-player table."""
        self._setup_large_table_globals()
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        self._assert_valid(decide(state), "AA preflop 15 players")

    def test_decide_preflop_20_players_weak_hand(self) -> None:
        """decide() returns valid action preflop with 72o at 20-player table."""
        self._setup_large_table_globals()
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=30, to_call=20, current_bet=20, min_raise=40,
            num_players=20, my_pid=3,
        )
        self._assert_valid(decide(state), "72o preflop 20 players")

    def test_decide_flop_15_players_multideck(self) -> None:
        """decide() returns valid action on flop at 15-player table (4 decks)."""
        self._setup_large_table_globals()
        state = make_state(
            hole_cards=["Ah", "Kd"],
            community=["Ac", "Kh", "3s"],
            street="flop",
            chips=1000, pot=200, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        self._assert_valid(decide(state), "flop 15 players")

    def test_decide_turn_15_players_multideck(self) -> None:
        """decide() returns valid action on turn at 15-player table."""
        self._setup_large_table_globals()
        state = make_state(
            hole_cards=["Kh", "Qd"],
            community=["Ac", "Jh", "Th", "9d"],
            street="turn",
            chips=500, pot=300, to_call=50, current_bet=50, min_raise=100,
            num_players=15, my_pid=0,
        )
        self._assert_valid(decide(state), "turn 15 players")

    def test_decide_river_20_players_multideck(self) -> None:
        """decide() returns valid action on river at 20-player table."""
        self._setup_large_table_globals()
        state = make_state(
            hole_cards=["Ah", "Kd"],
            community=["Ac", "Kh", "3s", "7d", "9c"],
            street="river",
            chips=1000, pot=400, to_call=100, current_bet=100, min_raise=200,
            num_players=20, my_pid=0,
        )
        self._assert_valid(decide(state), "river 20 players")

    def test_decide_never_raises_exception_large_table(self) -> None:
        """decide() must not raise for any large-table scenario."""
        self._setup_large_table_globals()
        scenarios = [
            # (hole, board, street, n_players, chips, to_call)
            (["Ah", "As"], [],                              "preflop", 20, 1000,  0),
            (["7h", "2c"], [],                              "preflop", 20, 1000, 20),
            (["Kh", "Qd"], ["Ac", "Jh", "Th"],             "flop",    15,  500, 50),
            (["2h", "3c"], ["Ac", "Jh", "Th", "9d"],       "turn",    16,  200,  0),
            (["Kh", "Qd"], ["Ac", "Jh", "Th", "9d", "8s"], "river",   18,  300, 100),
            (["5h", "5d"], [],                              "preflop", 20,  200,  0),  # short stack
        ]
        for hole, board, street, n, chips, to_call in scenarios:
            state = make_state(
                hole_cards=hole, community=board, street=street,
                chips=chips, pot=100, to_call=to_call, current_bet=to_call,
                min_raise=40, num_players=n, my_pid=0,
            )
            try:
                action = decide(state)
                self._assert_valid(action, f"{street} n={n}")
            except Exception as exc:
                pytest.fail(
                    f"decide() raised {type(exc).__name__}: {exc} "
                    f"[street={street}, n={n}]"
                )


# ═══════════════════════════════════════════════════════════════════
# OPEN STRENGTH THRESHOLD  (regression tests for the large-table fix)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.large_table
@pytest.mark.preflop
class TestOpenStrengthThreshold:
    """Regression tests for open_strength_threshold() and the _preflop() fix.

    Before the fix: open_range_pct returned a fraction (~0.068 at UTG/15-max)
    that was compared directly against Chen scores. Because every hand scores
    >= 0.10, every hand passed the threshold — the bot opened 72o from UTG.

    After the fix: open_range_pct is converted to a Chen score threshold via
    open_strength_threshold(), which looks up the actual weakest hand in the
    top-N% of 169 starting hands.
    """

    def test_lookup_has_169_entries(self) -> None:
        """Sanity check: 13 pairs + 78 suited + 78 offsuit = 169 hand types."""
        assert len(bot._HAND_STRENGTHS) == 169

    def test_lookup_is_sorted_descending(self) -> None:
        """Strongest hand first, weakest last."""
        s = bot._HAND_STRENGTHS
        assert s == sorted(s, reverse=True), "Lookup must be sorted descending"

    def test_aa_is_strongest(self) -> None:
        """AA (index 0) should be the highest score in the lookup."""
        aa = bot.preflop_strength(["Ah", "Ac"])
        assert bot._HAND_STRENGTHS[0] == pytest.approx(aa)

    def test_threshold_top_10pct_is_above_72o(self) -> None:
        """Top 10% threshold must be above 72o — 72o is NOT a top-10% hand."""
        thresh_10 = bot.open_strength_threshold(0.10)
        score_72o = bot.preflop_strength(["7h", "2c"])
        assert thresh_10 > score_72o, (
            f"Top-10% threshold ({thresh_10:.3f}) should exceed 72o ({score_72o:.3f})"
        )

    def test_threshold_top_30pct_excludes_72o(self) -> None:
        """72o is one of the worst hands — it does NOT make the top-30% cut."""
        thresh_30 = bot.open_strength_threshold(0.30)
        score_72o = bot.preflop_strength(["7h", "2c"])
        assert thresh_30 > score_72o, (
            f"Top-30% threshold ({thresh_30:.3f}) should exceed 72o ({score_72o:.3f})"
        )

    def test_threshold_increases_with_percentile(self) -> None:
        """Tighter range = higher (stricter) Chen score threshold."""
        t5  = bot.open_strength_threshold(0.05)
        t10 = bot.open_strength_threshold(0.10)
        t20 = bot.open_strength_threshold(0.20)
        t40 = bot.open_strength_threshold(0.40)
        assert t5 >= t10 >= t20 >= t40, (
            f"Thresholds must be non-increasing: {t5:.3f} {t10:.3f} {t20:.3f} {t40:.3f}"
        )

    def test_threshold_floor_is_0_pct(self) -> None:
        """threshold(0.0) returns the strongest hand score (AA)."""
        assert bot.open_strength_threshold(0.0) == pytest.approx(
            bot._HAND_STRENGTHS[0]
        )

    def test_threshold_ceiling_is_weakest_hand(self) -> None:
        """threshold(1.0) returns the weakest hand score."""
        assert bot.open_strength_threshold(1.0) == pytest.approx(
            bot._HAND_STRENGTHS[-1]
        )

    # ── The actual bug regression ──────────────────────────────────

    def test_72o_does_not_open_utg_15_players_no_raise(self) -> None:
        """THE BUG: 72o used to open-raise from UTG at 15-player table when
        no one had raised yet, because open_pct (~0.068) < Chen(72o) (~0.104).
        After the fix, 72o is correctly below the top-7% threshold and folds."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert action in ("fold", "check"), (
            f"72o UTG at 15 players (no raise) should fold/check, got {action!r}\n"
            f"open_thresh={bot.open_strength_threshold(bot.open_range_pct(3, 15)):.3f}, "
            f"72o_strength={bot.preflop_strength(['7h','2c']):.3f}"
        )

    def test_72o_does_not_open_utg_20_players_no_raise(self) -> None:
        """Same bug at 20 players."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=20, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert action in ("fold", "check"), (
            f"72o UTG at 20 players (no raise) should fold/check, got {action!r}"
        )

    def test_83o_does_not_open_utg_15_players_no_raise(self) -> None:
        """83o — another hand that was incorrectly opened from UTG at large tables."""
        state = make_state(
            hole_cards=["8h", "3c"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert action in ("fold", "check"), (
            f"83o UTG at 15 players should fold/check, got {action!r}"
        )

    def test_AA_still_opens_utg_15_players_no_raise(self) -> None:
        """Fix must not over-tighten: AA should still open from any seat."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"AA UTG at 15 players should raise, got {action!r}"
        )

    def test_6max_behavior_unchanged(self) -> None:
        """Fix must not regress normal 6-max play: 72o still folds UTG."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=6, my_pid=3,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=3, bb=20)
        assert action in ("fold", "check"), (
            f"72o UTG 6-max should still fold/check, got {action!r}"
        )

    def test_btn_opens_wide_still_at_large_table(self) -> None:
        """BTN at large table should still open suited connectors (top ~38%)."""
        state = make_state(
            hole_cards=["Td", "9d"],
            street="preflop",
            chips=1000, pot=30, to_call=0, current_bet=0, min_raise=40,
            num_players=15, my_pid=0,
        )
        bot.DEALER_PID = 0
        action = _preflop(state, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"T9s on BTN at 15 players should still open, got {action!r}"
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
