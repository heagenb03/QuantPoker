"""
tests/test_opponent_model.py -- Opponent model tracking and archetype classification.

Run modes:
    pytest -m opponent_model
    pytest tests/test_opponent_model.py -v
"""

import pytest

import bot
from conftest import setup_opponent


# ═══════════════════════════════════════════════════════════════════
# OPPONENT MODEL
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.opponent_model
class TestOpponentModel:
    """Tests for _OpponentModel archetype classification and stat tracking."""

    def test_unknown_under_10_hands(self) -> None:
        """With fewer than 10 hands, archetype is always 'unknown'."""
        setup_opponent(pid=1, hands=5, vpip=3, pfr=2, raises=5, calls=5)
        assert bot._OPP.archetype(1) == "unknown"

    def test_nit_classification(self) -> None:
        """VPIP < 0.18 => nit."""
        # 3/20 = 0.15 VPIP (< 0.18)
        setup_opponent(pid=1, hands=20, vpip=3, pfr=2, raises=5, calls=10)
        assert bot._OPP.archetype(1) == "nit"

    def test_station_classification(self) -> None:
        """VPIP >= 0.35 and AF < 1.5 => station."""
        # 8/20 = 0.40 VPIP, raises=3 calls=12 => AF = 3/12 = 0.25
        setup_opponent(pid=1, hands=20, vpip=8, pfr=2, raises=3, calls=12)
        assert bot._OPP.archetype(1) == "station"

    def test_maniac_classification(self) -> None:
        """VPIP >= 0.30 and AF >= 2.5 => maniac."""
        # 8/20 = 0.40 VPIP, raises=25 calls=5 => AF = 25/5 = 5.0
        setup_opponent(pid=1, hands=20, vpip=8, pfr=6, raises=25, calls=5)
        assert bot._OPP.archetype(1) == "maniac"

    def test_tag_classification(self) -> None:
        """VPIP >= 0.18, not station, not maniac => tag."""
        # 5/20 = 0.25 VPIP, raises=8 calls=8 => AF = 1.0
        setup_opponent(pid=1, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        assert bot._OPP.archetype(1) == "tag"

    def test_vpip_default_under_5_hands(self) -> None:
        """VPIP defaults to 0.25 when fewer than 5 hands seen."""
        setup_opponent(pid=1, hands=3, vpip=2, pfr=1, raises=1, calls=1)
        assert bot._OPP.vpip(1) == 0.25

    def test_af_default_under_5_actions(self) -> None:
        """AF defaults to 1.0 when fewer than 5 total actions."""
        setup_opponent(pid=1, hands=10, vpip=3, pfr=1, raises=1, calls=1)
        assert bot._OPP.af(1) == 1.0

    def test_range_width_per_archetype(self) -> None:
        expected = {
            "nit": 0.12,
            "tag": 0.22,
            "station": 0.45,
            "maniac": 0.38,
            "unknown": 0.25,
        }
        # Set up each archetype
        # nit
        setup_opponent(pid=10, hands=20, vpip=3, pfr=2, raises=5, calls=10)
        assert bot._OPP.range_width(10) == expected["nit"]
        # station
        setup_opponent(pid=11, hands=20, vpip=8, pfr=2, raises=3, calls=12)
        assert bot._OPP.range_width(11) == expected["station"]
        # maniac
        setup_opponent(pid=12, hands=20, vpip=8, pfr=6, raises=25, calls=5)
        assert bot._OPP.range_width(12) == expected["maniac"]
        # tag
        setup_opponent(pid=13, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        assert bot._OPP.range_width(13) == expected["tag"]
        # unknown
        setup_opponent(pid=14, hands=5, vpip=3, pfr=2, raises=5, calls=5)
        assert bot._OPP.range_width(14) == expected["unknown"]

    def test_update_hand_start_increments_hand_count(self) -> None:
        """hand_start message should trigger hand count increment."""
        messages = [
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, "1": 1000, "2": 1000}},
        ]
        bot._OPP.update(messages)
        # After one hand_start, hands should be 0 for all pids
        # because the first hand_start commits the PREVIOUS hand's VPIP/PFR
        # and increments hand count
        assert bot._OPP._s[0]["hands"] == 1
        assert bot._OPP._s[1]["hands"] == 1
        assert bot._OPP._s[2]["hands"] == 1

    def test_update_player_action_preflop_vpip(self) -> None:
        """Preflop call/raise/allin should register VPIP."""
        messages = [
            {"type": "player_action", "pid": 1, "action": "call", "street": "preflop"},
            {"type": "player_action", "pid": 2, "action": "raise", "street": "preflop"},
        ]
        bot._OPP.update(messages)
        assert 1 in bot._OPP._vpip_this_hand
        assert 2 in bot._OPP._vpip_this_hand

    def test_update_player_action_preflop_pfr(self) -> None:
        """Preflop raise/allin should register PFR."""
        messages = [
            {"type": "player_action", "pid": 1, "action": "call", "street": "preflop"},
            {"type": "player_action", "pid": 2, "action": "raise", "street": "preflop"},
        ]
        bot._OPP.update(messages)
        assert 1 not in bot._OPP._pfr_this_hand  # call is not PFR
        assert 2 in bot._OPP._pfr_this_hand

    def test_update_showdown_records_cards(self) -> None:
        """Showdown messages should store opponent hole cards."""
        messages = [
            {"type": "showdown", "winners": [1], "pot": 200,
             "hands": {
                 "1": {"cards": ["Ah", "Kd"], "hand": "high card"},
                 "2": {"cards": ["Qh", "Jd"], "hand": "high card"},
             },
             "stacks": {"1": 1200, "2": 800}},
        ]
        bot._OPP.update(messages)
        assert ["Ah", "Kd"] in bot._OPP._s[1]["showdown"]
        assert ["Qh", "Jd"] in bot._OPP._s[2]["showdown"]

    def test_vpip_with_sufficient_hands(self) -> None:
        """VPIP is calculated as vpip_count / hands when hands >= 5."""
        setup_opponent(pid=1, hands=20, vpip=10, pfr=5, raises=8, calls=8)
        assert abs(bot._OPP.vpip(1) - 0.50) < 1e-9

    def test_pfr_with_sufficient_hands(self) -> None:
        """PFR is calculated as pfr_count / hands when hands >= 5."""
        setup_opponent(pid=1, hands=20, vpip=10, pfr=6, raises=8, calls=8)
        assert abs(bot._OPP.pfr(1) - 0.30) < 1e-9

    def test_pfr_default_under_5_hands(self) -> None:
        """PFR defaults to 0.12 when fewer than 5 hands."""
        setup_opponent(pid=1, hands=3, vpip=2, pfr=1, raises=1, calls=1)
        assert bot._OPP.pfr(1) == 0.12

    def test_af_with_sufficient_actions(self) -> None:
        """AF = raises / max(calls, 1) when total >= 5."""
        setup_opponent(pid=1, hands=20, vpip=8, pfr=4, raises=10, calls=5)
        assert abs(bot._OPP.af(1) - 2.0) < 1e-9

    def test_af_zero_calls(self) -> None:
        """AF with zero calls uses max(calls, 1) = 1."""
        setup_opponent(pid=1, hands=20, vpip=8, pfr=4, raises=10, calls=0)
        # total = 10 >= 5, so computed; raises/max(0,1) = 10/1 = 10
        assert abs(bot._OPP.af(1) - 10.0) < 1e-9

    def test_raise_increments_raises(self) -> None:
        """Any raise/allin/bet action should increment raises counter."""
        messages = [
            {"type": "player_action", "pid": 1, "action": "raise", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "bet", "street": "turn"},
            {"type": "player_action", "pid": 1, "action": "allin", "street": "river"},
        ]
        bot._OPP.update(messages)
        assert bot._OPP._s[1]["raises"] == 3

    def test_call_and_fold_increment_calls(self) -> None:
        """Call, fold, and check increment the calls counter."""
        messages = [
            {"type": "player_action", "pid": 1, "action": "call", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "fold", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "check", "street": "flop"},
        ]
        bot._OPP.update(messages)
        assert bot._OPP._s[1]["calls"] == 3
