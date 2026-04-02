"""
tests/test_preflop.py -- Preflop strength, position, and decision tests.

Run modes:
    pytest -m preflop
    pytest tests/test_preflop.py -v
"""

import pytest

import bot
from bot import (
    _preflop,
    get_position,
    open_range_pct,
    players_behind_preflop,
    preflop_strength,
)
from conftest import make_state, setup_opponent


# ═══════════════════════════════════════════════════════════════════
# PREFLOP STRENGTH (Chen formula)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.preflop
class TestPreflopStrength:
    """Unit tests for preflop_strength() -- Chen formula normalized to [0,1]."""

    def test_AA_is_strongest(self) -> None:
        score = preflop_strength(["Ah", "As"])
        assert score >= 0.80, f"AA should be strong, got {score}"

    def test_72o_is_weakest(self) -> None:
        score = preflop_strength(["7h", "2c"])
        assert score <= 0.25, f"72o should be weak, got {score}"

    def test_suited_bonus(self) -> None:
        suited = preflop_strength(["Ah", "Kh"])
        offsuit = preflop_strength(["Ah", "Kd"])
        assert suited > offsuit, "Suited should score higher than offsuit"

    def test_pair_minimum(self) -> None:
        # Even the lowest pair (22) should be decent
        score = preflop_strength(["2h", "2c"])
        assert score >= 0.20, f"22 should have pair bonus, got {score}"

    def test_connected_bonus(self) -> None:
        connected = preflop_strength(["8h", "7h"])
        gapped = preflop_strength(["8h", "5h"])
        assert connected > gapped, "Connected should beat gapped"

    def test_range_bounded_0_1(self) -> None:
        # Test a spread of hands to ensure all are in [0, 1]
        hands = [
            ["Ah", "As"], ["2h", "3c"], ["Kh", "Qs"],
            ["7h", "2c"], ["Td", "9d"], ["5h", "4h"],
        ]
        for h in hands:
            score = preflop_strength(h)
            assert 0.0 <= score <= 1.0, f"{h} => {score} is out of range"

    def test_order_independent(self) -> None:
        # preflop_strength should normalize card order internally
        a = preflop_strength(["Ah", "Kd"])
        b = preflop_strength(["Kd", "Ah"])
        assert a == b, "Strength should be order-independent"

    def test_gap_penalty_increases(self) -> None:
        # Larger gaps should reduce score
        gap0 = preflop_strength(["8h", "7c"])  # connected
        gap1 = preflop_strength(["8h", "6c"])  # one gap
        gap3 = preflop_strength(["8h", "4c"])  # three gap
        assert gap0 > gap1 >= gap3, "Gap penalty should increase with gap size"


# ═══════════════════════════════════════════════════════════════════
# POSITION UTILITIES
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.preflop
class TestPosition:
    """Unit tests for get_position, players_behind_preflop, open_range_pct."""

    def test_btn_is_position_0(self) -> None:
        pos = get_position(my_pid=3, dealer_pid=3, n_players=6)
        assert pos == 0

    def test_sb_is_position_1(self) -> None:
        pos = get_position(my_pid=4, dealer_pid=3, n_players=6)
        assert pos == 1

    def test_bb_is_position_2(self) -> None:
        pos = get_position(my_pid=5, dealer_pid=3, n_players=6)
        assert pos == 2

    def test_wraps_around(self) -> None:
        pos = get_position(my_pid=0, dealer_pid=3, n_players=6)
        assert pos == 3

    def test_unknown_dealer_gives_middle(self) -> None:
        pos = get_position(my_pid=0, dealer_pid=None, n_players=6)
        assert pos == 3  # n_players // 2

    def test_players_behind_bb_closes(self) -> None:
        assert players_behind_preflop(2, 6) == 0  # BB always closes

    def test_players_behind_sb(self) -> None:
        assert players_behind_preflop(1, 6) == 1

    def test_players_behind_btn(self) -> None:
        assert players_behind_preflop(0, 6) == 2

    def test_open_range_btn_wider_than_utg(self) -> None:
        btn_range = open_range_pct(0, 6)
        utg_range = open_range_pct(3, 6)
        assert btn_range > utg_range, "BTN should open wider than UTG"

    def test_open_range_bounded(self) -> None:
        for n in range(2, 10):
            for pos in range(n):
                pct = open_range_pct(pos, n)
                assert 0.06 <= pct <= 0.50, f"pos={pos} n={n} pct={pct}"


# ═══════════════════════════════════════════════════════════════════
# PREFLOP DECISION
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.preflop
class TestPreflopDecision:
    """Tests for _preflop() routing and sub-handlers."""

    def test_scenario_AA_open_btn(self) -> None:
        """AA on the button with no raise should open-raise."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
            my_pid=0,
        )
        bot.DEALER_PID = 0  # pid 0 is BTN
        action = _preflop(state, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"AA on BTN should raise, got {action}"
        )

    def test_scenario_72o_fold_utg(self) -> None:
        """72o in early position should fold (or check if allowed)."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000,
            pot=30,
            to_call=20,
            current_bet=20,
            min_raise=40,
            num_players=6,
            my_pid=3,
        )
        action = _preflop(state, pos=3, bb=20)
        assert action == "fold", f"72o UTG facing raise should fold, got {action}"

    def test_scenario_push_fold_short_stack(self) -> None:
        """With < 15 BB, strong hands should push all-in."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=200,  # 10 BB
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
            my_pid=0,
        )
        action = _preflop(state, pos=0, bb=20)
        assert action == "allin", f"AA with 10BB should push, got {action}"

    def test_scenario_push_fold_weak_hand_folds(self) -> None:
        """With < 15 BB, weak hands facing a raise should fold."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=200,  # 10 BB
            pot=100,
            to_call=60,
            current_bet=60,
            min_raise=120,
            num_players=4,
            my_pid=3,
        )
        action = _preflop(state, pos=3, bb=20)
        assert action == "fold", f"72o short stack facing raise should fold, got {action}"

    def test_scenario_facing_raise_call_decent_hand(self) -> None:
        """Decent hand facing a single raise should call."""
        state = make_state(
            hole_cards=["Kh", "Qh"],
            street="preflop",
            chips=1000,
            pot=90,
            to_call=40,
            current_bet=40,
            min_raise=80,
            num_players=4,
            my_pid=2,
        )
        action = _preflop(state, pos=2, bb=20)
        assert action in ("call", "fold") or (
            isinstance(action, tuple) and action[0] == "raise"
        ), f"KQs facing raise should call/raise/fold, got {action}"

    def test_scenario_facing_3bet_folds_marginal(self) -> None:
        """87o facing a 3-bet should fold — below all call/shove thresholds."""
        state = make_state(
            hole_cards=["8h", "7c"],
            street="preflop",
            chips=1000,
            pot=200,
            to_call=80,  # > 3*BB=60, so facing_3bet=True
            current_bet=80,
            min_raise=160,
            num_players=4,
            my_pid=1,
        )
        action = _preflop(state, pos=1, bb=20)
        assert action == "fold", f"87o facing 3bet should fold, got {action}"

    def test_3bet_threshold_is_selective(self) -> None:
        """72o should fold vs a 3-bet."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000,
            pot=200,
            to_call=80,
            current_bet=80,
            min_raise=160,
            num_players=4,
            my_pid=1,
        )
        action = _preflop(state, pos=1, bb=20)
        assert action == "fold", f"72o facing 3bet should fold, got {action}"

    def test_scenario_facing_3bet_continues_premium(self) -> None:
        """Premium hand facing a 3-bet should 4-bet or call."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=1000,
            pot=200,
            to_call=80,
            current_bet=80,
            min_raise=160,
            num_players=4,
            my_pid=1,
        )
        action = _preflop(state, pos=1, bb=20)
        assert action != "fold", f"AA facing 3bet should not fold, got {action}"

    def test_AKo_vs_maniac_3bet_does_not_fold(self) -> None:
        """AKo should 4-bet or call a maniac's 3-bet.

        Regression: VS_3BET_SHOVE_DEEP=0.65 was cutting out AKo (strength=0.583),
        folding it even against maniacs who 3-bet wide.
        """
        setup_opponent(pid=1, hands=15, vpip=6, pfr=5, raises=10, calls=2)
        state = make_state(
            hole_cards=["Ah", "Kc"],   # AKo: strength ≈ 0.583
            pot=270, to_call=120, current_bet=120, min_raise=240,
            chips=980, num_players=4, my_pid=0,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _preflop(state, pos=1, bb=20)
        assert action != "fold", f"AKo vs maniac 3-bet should not fold, got {action}"

    def test_AQo_vs_maniac_3bet_does_not_fold(self) -> None:
        """AQo should not fold vs a maniac 3-bet (maniac range is wide)."""
        setup_opponent(pid=1, hands=15, vpip=6, pfr=5, raises=10, calls=2)
        state = make_state(
            hole_cards=["Ah", "Qc"],   # AQo: strength ≈ 0.542
            pot=270, to_call=120, current_bet=120, min_raise=240,
            chips=980, num_players=4, my_pid=0,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _preflop(state, pos=1, bb=20)
        assert action != "fold", f"AQo vs maniac 3-bet should not fold, got {action}"

    def test_88_vs_maniac_3bet_deep_does_not_fold(self) -> None:
        """88 at 97BB should call a maniac's 3-bet, not fold."""
        setup_opponent(pid=1, hands=15, vpip=6, pfr=5, raises=10, calls=2)
        state = make_state(
            hole_cards=["8h", "8d"],   # 88: strength = 0.5
            pot=270, to_call=120, current_bet=120, min_raise=240,
            chips=1940, num_players=4, my_pid=0,  # ~97 BB
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _preflop(state, pos=1, bb=20)
        assert action != "fold", f"88 vs maniac 3-bet at 97BB should not fold, got {action}"

    def test_AKo_vs_tag_3bet_calls(self) -> None:
        """AKo should call (or 4-bet) a TAG's 3-bet — not fold."""
        setup_opponent(pid=1, hands=15, vpip=4, pfr=3, raises=5, calls=3)
        state = make_state(
            hole_cards=["Ah", "Kc"],   # AKo: strength ≈ 0.583
            pot=270, to_call=120, current_bet=120, min_raise=240,
            chips=980, num_players=4, my_pid=0,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _preflop(state, pos=1, bb=20)
        assert action != "fold", f"AKo vs TAG 3-bet should not fold, got {action}"

    def test_no_raise_weak_hand_checks_bb(self) -> None:
        """BB with weak hand and no raise should check."""
        state = make_state(
            hole_cards=["7h", "2c"],
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=20,
            min_raise=40,
            num_players=6,
            my_pid=2,
        )
        action = _preflop(state, pos=2, bb=20)
        assert action == "check", f"72o BB no raise should check, got {action}"

    def test_scenario_facing_3bet_short_stack_shoves(self) -> None:
        """Facing 3-bet with < 25 BB and a hand above threshold => allin."""
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="preflop",
            chips=400,  # 20 BB
            pot=200,
            to_call=80,
            current_bet=80,
            min_raise=160,
            num_players=4,
            my_pid=1,
        )
        action = _preflop(state, pos=1, bb=20)
        assert action == "allin", f"Short stack facing 3bet should shove, got {action}"

    def test_scenario_facing_single_raise_calls(self) -> None:
        """Decent hand facing a single raise (not 3-bet) should call."""
        state = make_state(
            hole_cards=["Kh", "Qh"],  # ~0.50 strength
            street="preflop",
            chips=1000,
            pot=70,
            to_call=40,  # > bb but <= 3*bb, so single raise
            current_bet=40,
            min_raise=80,
            num_players=4,
            my_pid=2,
        )
        action = _preflop(state, pos=2, bb=20)
        assert action in ("call",) or (isinstance(action, tuple) and action[0] == "raise"), (
            f"KQs facing single raise should call or 3bet, got {action}"
        )

    def test_scenario_facing_single_raise_folds_weak(self) -> None:
        """Weak hand facing single raise with bad pot odds should fold."""
        state = make_state(
            hole_cards=["4h", "2c"],  # very weak
            street="preflop",
            chips=1000,
            pot=70,
            to_call=40,
            current_bet=40,
            min_raise=80,
            num_players=4,
            my_pid=3,
        )
        action = _preflop(state, pos=3, bb=20)
        assert action == "fold", f"42o facing raise should fold, got {action}"

    def test_scenario_push_fold_short_weak_check_bb(self) -> None:
        """Short stack BB with weak hand and no raise: check (not fold)."""
        state = make_state(
            hole_cards=["4h", "2c"],
            street="preflop",
            chips=200,  # 10 BB
            pot=30,
            to_call=0,
            current_bet=20,
            min_raise=40,
            num_players=6,
            my_pid=2,
        )
        action = _preflop(state, pos=2, bb=20)
        # BB can check for free; push threshold for pos 2 is tight but can_check=True
        assert action in ("check", "allin"), f"Short BB no raise, got {action}"

    def test_scenario_push_fold_facing_shove_strong(self) -> None:
        """Short stack facing a shove with a strong hand should call all-in."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=200,
            pot=430,
            to_call=200,  # facing all-in
            current_bet=200,
            min_raise=400,
            num_players=4,
            my_pid=1,
        )
        action = _preflop(state, pos=1, bb=20)
        assert action == "allin", f"AA short facing shove should push, got {action}"

    def test_scenario_push_fold_facing_shove_weak(self) -> None:
        """Short stack facing a shove with a weak hand should fold."""
        state = make_state(
            hole_cards=["4h", "2c"],
            street="preflop",
            chips=200,
            pot=430,
            to_call=200,
            current_bet=200,
            min_raise=400,
            num_players=6,
            my_pid=5,
        )
        action = _preflop(state, pos=5, bb=20)
        assert action == "fold", f"42o short facing shove should fold, got {action}"

    def test_maniac_at_table_tightens_open(self) -> None:
        """With a maniac behind, marginal opens should be suppressed.

        We compare against a baseline (no maniac) to confirm that the
        maniac penalty actually flips the decision for a borderline hand.

        Hand choice: 98o (strength ~0.396) is in the BTN marginal zone —
        above the 48%-open threshold (~0.333) but below threshold*1.35 (~0.450).
        T9s (0.500) is no longer marginal after the open_strength_threshold fix;
        it's solidly above the maniac-suppression zone and correctly stays open.
        """
        hand = ["9h", "8c"]  # marginal BTN open: strength 0.396, in (0.333, 0.450)
        base_state = make_state(
            hole_cards=hand,
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
            my_pid=0,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        # Baseline: no maniac at the table
        action_without_maniac = _preflop(base_state, pos=0, bb=20)

        # Now set up pid 1 as a maniac: VPIP >= 0.30, AF >= 2.5
        setup_opponent(pid=1, hands=20, vpip=8, pfr=6, raises=25, calls=5)
        maniac_state = make_state(
            hole_cards=hand,
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
            my_pid=0,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action_with_maniac = _preflop(maniac_state, pos=0, bb=20)

        # Baseline should open-raise (raise tuple), maniac should suppress to check/fold
        baseline_is_raise = isinstance(action_without_maniac, tuple) and action_without_maniac[0] == "raise"
        maniac_is_passive = action_with_maniac in ("check", "fold")
        assert baseline_is_raise, (
            f"Baseline (no maniac) should open-raise 98o, got {action_without_maniac}"
        )
        assert maniac_is_passive, (
            f"With maniac behind, 98o should check/fold, got {action_with_maniac}"
        )
