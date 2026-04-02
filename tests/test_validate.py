"""
tests/test_validate.py -- Action validation, GameState properties, multi-deck eval,
and edge case tests.

Run modes:
    pytest -m validate
    pytest tests/test_validate.py -v
"""

import pytest

import bot
from bot import (
    _postflop,
    _validate,
    preflop_strength,
)
from conftest import make_sim_result, make_state


# ═══════════════════════════════════════════════════════════════════
# VALIDATE
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.validate
class TestValidate:
    """Tests for _validate() -- action clamping and legality checks."""

    def test_illegal_check_becomes_fold(self) -> None:
        state = make_state(to_call=20, current_bet=20)
        result = _validate("check", state)
        assert result == "fold"

    def test_legal_check_passes(self) -> None:
        state = make_state(to_call=0)
        result = _validate("check", state)
        assert result == "check"

    def test_raise_clamped_to_min_raise(self) -> None:
        state = make_state(min_raise=80, current_bet=20, chips=1000)
        result = _validate(("raise", 30), state)
        assert isinstance(result, tuple)
        assert result[1] >= 80, f"Should clamp to min_raise, got {result[1]}"

    def test_raise_clamped_to_max_chips(self) -> None:
        state = make_state(min_raise=40, current_bet=0, chips=100)
        result = _validate(("raise", 500), state)
        assert isinstance(result, tuple)
        assert result[1] <= 100, f"Should clamp to chips, got {result[1]}"

    def test_raise_below_current_bet_becomes_call(self) -> None:
        state = make_state(
            min_raise=40, current_bet=200, chips=100,
        )
        # amount=100, clamped to max(100, min_raise=40) but also min(100, chips+current_bet=300)
        # then amount <= current_bet check: 100 <= 200 => call
        result = _validate(("raise", 50), state)
        assert result == "call", f"Raise <= current_bet should become call, got {result}"

    def test_fold_passes_through(self) -> None:
        state = make_state()
        assert _validate("fold", state) == "fold"

    def test_call_passes_through(self) -> None:
        state = make_state()
        assert _validate("call", state) == "call"

    def test_allin_passes_through(self) -> None:
        state = make_state()
        assert _validate("allin", state) == "allin"

    def test_raise_amount_is_integer(self) -> None:
        state = make_state(min_raise=40, current_bet=0, chips=1000)
        result = _validate(("raise", 55.7), state)
        assert isinstance(result, tuple)
        assert isinstance(result[1], int), "Raise amount should be int"


# ═══════════════════════════════════════════════════════════════════
# GAME STATE PROPERTIES
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.validate
class TestGameStateProperties:
    """Tests for GameState computed properties."""

    def test_can_check_true(self) -> None:
        state = make_state(to_call=0)
        assert state.can_check is True

    def test_can_check_false(self) -> None:
        state = make_state(to_call=20)
        assert state.can_check is False

    def test_active_opponents(self) -> None:
        state = make_state(
            my_pid=0,
            num_players=4,
            player_folded={"0": False, "1": False, "2": True, "3": False},
        )
        assert state.active_opponents == 2  # pid 1 and 3

    def test_active_opponents_all_folded(self) -> None:
        state = make_state(
            my_pid=0,
            num_players=4,
            player_folded={"0": False, "1": True, "2": True, "3": True},
        )
        assert state.active_opponents == 0

    def test_pot_odds_zero_when_free(self) -> None:
        state = make_state(to_call=0, pot=100)
        assert state.pot_odds == 0.0

    def test_pot_odds_calculation(self) -> None:
        state = make_state(to_call=50, pot=150)
        # 50 / (150 + 50) = 0.25
        assert abs(state.pot_odds - 0.25) < 1e-9

    def test_repr_does_not_crash(self) -> None:
        state = make_state()
        r = repr(state)
        assert "GameState" in r


# ═══════════════════════════════════════════════════════════════════
# MULTI-DECK EVALUATION
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.validate
class TestMultiDeckEval:
    """Tests for multi-deck card evaluation."""

    def test_eval_5card_pair_beats_high_card(self) -> None:
        from bot import _eval_5card_multideck
        pair = _eval_5card_multideck(["Ah", "Ad", "Ks", "Qc", "Jh"])
        high = _eval_5card_multideck(["Ah", "Kd", "Qs", "Jc", "9h"])
        assert pair > high

    def test_eval_5card_flush_beats_straight(self) -> None:
        from bot import _eval_5card_multideck
        flush = _eval_5card_multideck(["Ah", "Kh", "Qh", "Jh", "9h"])
        straight = _eval_5card_multideck(["Ah", "Kd", "Qs", "Jc", "Th"])
        assert flush > straight

    def test_eval_5card_full_house_beats_flush(self) -> None:
        from bot import _eval_5card_multideck
        fh = _eval_5card_multideck(["Ah", "Ad", "As", "Kc", "Kh"])
        flush = _eval_5card_multideck(["Ah", "Kh", "Qh", "Jh", "9h"])
        assert fh > flush

    def test_eval_multideck_7_cards(self) -> None:
        from bot import _eval_multideck
        # 7-card hand with AA pair should beat 7-card hand with no pair
        score_pair = _eval_multideck(["Ah", "Ad", "Ks", "Qc", "Jh", "9s", "8c"])
        score_high = _eval_multideck(["Ah", "Kd", "Qs", "Jc", "9h", "7s", "5c"])
        assert score_pair > score_high

    def test_eval_5card_straight_flush(self) -> None:
        from bot import _eval_5card_multideck
        sf = _eval_5card_multideck(["9h", "8h", "7h", "6h", "5h"])
        quads = _eval_5card_multideck(["Ah", "Ad", "As", "Ac", "Kh"])
        assert sf > quads

    def test_eval_5card_quads_beats_full_house(self) -> None:
        from bot import _eval_5card_multideck
        quads = _eval_5card_multideck(["Ah", "Ad", "As", "Ac", "Kh"])
        fh = _eval_5card_multideck(["Ah", "Ad", "As", "Kc", "Kh"])
        assert quads > fh

    def test_eval_5card_straight_no_flush(self) -> None:
        from bot import _eval_5card_multideck
        straight = _eval_5card_multideck(["9h", "8d", "7s", "6c", "5h"])
        three = _eval_5card_multideck(["7h", "7d", "7s", "Ac", "Kh"])
        assert straight > three

    def test_eval_5card_three_of_a_kind(self) -> None:
        from bot import _eval_5card_multideck
        three = _eval_5card_multideck(["7h", "7d", "7s", "Ac", "Kh"])
        two_pair = _eval_5card_multideck(["Ah", "Ad", "Ks", "Kc", "Qh"])
        assert three > two_pair

    def test_eval_5card_two_pair(self) -> None:
        from bot import _eval_5card_multideck
        two_pair = _eval_5card_multideck(["Ah", "Ad", "Ks", "Kc", "Qh"])
        one_pair = _eval_5card_multideck(["Ah", "Ad", "Ks", "Qc", "Jh"])
        assert two_pair > one_pair

    def test_eval_5card_ace_low_straight(self) -> None:
        """A-2-3-4-5 (wheel) should be recognized as a straight."""
        from bot import _eval_5card_multideck
        wheel = _eval_5card_multideck(["Ah", "2d", "3s", "4c", "5h"])
        # Wheel is a straight (category 4) with high=5
        assert wheel[0] == 4  # straight category
        assert wheel[1] == (5,)  # high card is 5

    def test_eval_5card_five_of_a_kind_multideck(self) -> None:
        """Five-of-a-kind (multi-deck only) beats straight flush."""
        from bot import _eval_5card_multideck
        five_k = _eval_5card_multideck(["Ah", "Ad", "As", "Ac", "Ah"])
        sf = _eval_5card_multideck(["Kh", "Qh", "Jh", "Th", "9h"])
        assert five_k > sf


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.validate
class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_community_preflop(self) -> None:
        state = make_state(street="preflop", community=[])
        assert state.community == []
        assert state.street == "preflop"

    def test_full_board_river(self) -> None:
        state = make_state(
            street="river",
            community=["Ts", "9h", "2c", "Kd", "3s"],
        )
        assert len(state.community) == 5

    def test_validate_with_zero_chips(self) -> None:
        state = make_state(chips=0, current_bet=0, min_raise=40)
        result = _validate(("raise", 100), state)
        # chips + current_bet = 0 => amount clamped to 0 <= current_bet => call
        assert result == "call"

    def test_preflop_strength_edge_pair_of_aces(self) -> None:
        score = preflop_strength(["Ah", "As"])
        assert score >= 0.90

    def test_preflop_strength_edge_pair_of_twos(self) -> None:
        score = preflop_strength(["2h", "2c"])
        assert 0.20 <= score <= 0.50

    def test_postflop_no_ev_actions_returns_fold(self) -> None:
        """If ev_by_action results in empty available after filtering, fold."""
        sim = make_sim_result(
            equity=0.10,
            risk_adj_equity=0.05,
            ev_by_action={},  # no actions at all
            street="river",
        )
        state = make_state(
            street="river",
            community=["Ts", "9h", "2c", "Kd", "3s"],
            pot=200,
            to_call=100,
            current_bet=100,
            min_raise=200,
            chips=500,
        )
        action = _postflop(state, sim, pos=2, bb=20)
        assert action == "fold"

    def test_gamestate_repr(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["Ts", "9h", "2c"],
            pot=100,
            chips=500,
            to_call=20,
        )
        r = repr(state)
        assert "flop" in r
        assert "100" in r
