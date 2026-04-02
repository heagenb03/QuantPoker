"""
tests/test_integration.py -- End-to-end decide() call traces with real simulation.

Run modes:
    pytest -m integration
    pytest tests/test_integration.py -v
"""

import pytest

import bot
from bot import decide
from conftest import make_state


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: Full decide() call traces
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestDecideIntegration:
    """End-to-end tests calling decide() with real simulation."""

    def test_preflop_returns_valid_action(self) -> None:
        """decide() on preflop returns a valid action type."""
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
        )
        action = decide(state)
        assert action in ("fold", "check", "call", "allin") or (
            isinstance(action, tuple) and action[0] == "raise"
        ), f"Invalid action: {action}"

    def test_postflop_returns_valid_action(self) -> None:
        """decide() on postflop (runs simulation) returns a valid action type."""
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
        )
        action = decide(state)
        assert action in ("fold", "check", "call", "allin") or (
            isinstance(action, tuple) and action[0] == "raise"
        ), f"Invalid action: {action}"

    def test_decide_updates_hlen(self) -> None:
        """decide() should advance _HLEN to track processed history."""
        history = [
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, "1": 1000, "2": 1000, "3": 1000}},
        ]
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="preflop",
            chips=1000,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
            history=history,
        )
        assert bot._HLEN == 0
        decide(state)
        assert bot._HLEN == 1

    def test_decide_facing_bet_postflop(self) -> None:
        """decide() facing a bet on the flop should return a valid action."""
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=120,
            to_call=60,
            current_bet=60,
            min_raise=120,
            num_players=4,
        )
        action = decide(state)
        assert action in ("fold", "check", "call", "allin") or (
            isinstance(action, tuple) and action[0] == "raise"
        ), f"Invalid action: {action}"

    def test_decide_short_stack_preflop(self) -> None:
        """Short stack (<15BB) preflop should trigger push-fold."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="preflop",
            chips=200,
            pot=30,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=4,
        )
        action = decide(state)
        # AA with 10BB should push
        assert action == "allin", f"AA short stack should push, got {action}"

    def test_decide_river_with_strong_hand(self) -> None:
        """River with a strong hand should produce a bet or call."""
        state = make_state(
            hole_cards=["Ah", "As"],
            street="river",
            community=["Ad", "9h", "2c", "Kd", "3s"],
            chips=1000,
            pot=200,
            to_call=0,
            current_bet=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        action = decide(state)
        # Trip aces on river should bet
        assert action != "fold", f"Trip aces should not fold, got {action}"

    def test_decide_preserves_opponent_model_across_calls(self) -> None:
        """Multiple decide() calls should accumulate opponent model data."""
        history1 = [
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, "1": 1000, "2": 1000, "3": 1000}},
            {"type": "player_action", "pid": 1, "action": "raise", "street": "preflop"},
        ]
        state1 = make_state(
            hole_cards=["Ah", "Kd"],
            street="preflop",
            chips=1000,
            pot=70,
            to_call=40,
            current_bet=40,
            min_raise=80,
            num_players=4,
            history=history1,
        )
        decide(state1)

        # Second call with more history
        history2 = history1 + [
            {"type": "player_action", "pid": 1, "action": "raise", "street": "preflop"},
        ]
        state2 = make_state(
            hole_cards=["Qh", "Qd"],
            street="preflop",
            chips=960,
            pot=70,
            to_call=40,
            current_bet=40,
            min_raise=80,
            num_players=4,
            history=history2,
        )
        decide(state2)
        # pid 1 should have accumulated raise stats
        assert bot._OPP._s[1]["raises"] >= 2
