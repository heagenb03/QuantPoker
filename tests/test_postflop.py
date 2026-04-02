"""
tests/test_postflop.py -- Postflop decision, simulation, and equity tests.

Run modes:
    pytest -m postflop
    pytest tests/test_postflop.py -v
"""

import random
from unittest.mock import patch

import pytest

import bot
from bot import (
    SimResult,
    _postflop,
    monte_carlo_equity,
    run_unified_simulation,
)
from conftest import make_sim_result, make_state, setup_opponent


# ═══════════════════════════════════════════════════════════════════
# POSTFLOP DECISION
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.postflop
class TestPostflopDecision:
    """Tests for _postflop() -- takes SimResult directly, no simulation."""

    def test_scenario_high_equity_bets(self) -> None:
        """Strong hand should bet when EV of betting is highest."""
        sim = make_sim_result(
            equity=0.85,
            risk_adj_equity=0.80,
            ev_by_action={
                "fold": 0.0,
                "check": 30.0,
                "bet_half": 55.0,
                "bet_full": 50.0,
                "shove": 40.0,
            },
            street="flop",
        )
        state = make_state(
            street="flop",
            community=["Ts", "9h", "2c"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
        )
        action = _postflop(state, sim, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"High equity should bet, got {action}"
        )

    def test_scenario_low_equity_folds(self) -> None:
        """Weak hand with no equity should fold when facing a bet."""
        sim = make_sim_result(
            equity=0.15,
            risk_adj_equity=0.10,
            continuation_value=0.12,
            draw_premium=-0.03,
            ev_by_action={
                "fold": 0.0,
                "call": -25.0,
                "shove": -80.0,
            },
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
        assert action == "fold", f"Low equity facing big bet should fold, got {action}"

    def test_scenario_station_blocks_bluff(self) -> None:
        """When a station is present and equity is low, bet options are removed."""
        # Set up pid 1 as station: VPIP >= 0.35, AF < 1.5
        setup_opponent(pid=1, hands=20, vpip=9, pfr=2, raises=3, calls=12)

        sim = make_sim_result(
            equity=0.40,
            risk_adj_equity=0.35,
            ev_by_action={
                "fold": 0.0,
                "check": 15.0,
                "bet_half": 25.0,
                "bet_full": 22.0,
                "shove": -10.0,
            },
            street="flop",
        )
        state = make_state(
            street="flop",
            community=["Ts", "9h", "2c"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _postflop(state, sim, pos=0, bb=20)
        # Station present + risk_adj_equity < 0.52 => bet_half and bet_full removed
        # Best remaining should be check
        assert action == "check", f"Should check vs station with low equity, got {action}"

    def test_scenario_station_allows_value_bet(self) -> None:
        """When station is present but equity is HIGH, value betting is allowed."""
        setup_opponent(pid=1, hands=20, vpip=9, pfr=2, raises=3, calls=12)

        sim = make_sim_result(
            equity=0.80,
            risk_adj_equity=0.75,  # >= 0.52, so station override does NOT remove bets
            ev_by_action={
                "fold": 0.0,
                "check": 15.0,
                "bet_half": 60.0,
                "bet_full": 55.0,
                "shove": 30.0,
            },
            street="flop",
        )
        state = make_state(
            street="flop",
            community=["Ts", "9h", "2c"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _postflop(state, sim, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"Should value bet vs station with high equity, got {action}"
        )

    def test_scenario_maniac_trap(self) -> None:
        """Strong hand vs maniac on flop/turn should check to trap."""
        # Set up pid 1 as maniac: VPIP >= 0.30, AF >= 2.5
        setup_opponent(pid=1, hands=20, vpip=8, pfr=6, raises=25, calls=5)

        sim = make_sim_result(
            equity=0.80,
            risk_adj_equity=0.75,  # >= 0.65
            ev_by_action={
                "fold": 0.0,
                "check": 30.0,
                "bet_half": 50.0,
                "bet_full": 45.0,
                "shove": 20.0,
            },
            street="flop",  # not river
        )
        state = make_state(
            street="flop",
            community=["Ts", "9h", "2c"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _postflop(state, sim, pos=0, bb=20)
        # Maniac trap: check EV boosted to best_bet * 1.15
        # best_bet = max(50, 45) = 50, check boosted to 57.5
        assert action == "check", f"Should trap maniac on flop, got {action}"

    def test_scenario_maniac_trap_disabled_on_river(self) -> None:
        """Maniac trap does NOT apply on the river."""
        setup_opponent(pid=1, hands=20, vpip=8, pfr=6, raises=25, calls=5)

        sim = make_sim_result(
            equity=0.80,
            risk_adj_equity=0.75,
            ev_by_action={
                "fold": 0.0,
                "check": 30.0,
                "bet_half": 50.0,
                "bet_full": 45.0,
                "shove": 20.0,
            },
            street="river",
        )
        state = make_state(
            street="river",
            community=["Ts", "9h", "2c", "Kd", "3s"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            player_folded={"0": False, "1": False, "2": True, "3": True},
        )
        action = _postflop(state, sim, pos=0, bb=20)
        # On river, maniac trap is off; bet_half (50) is highest EV
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"Should bet river vs maniac (no trap), got {action}"
        )

    def test_scenario_bluff_gate_allowed(self) -> None:
        """Bluff gate passes when all conditions met: HU, BTN, no station, draw_premium < 0.02, random < 0.25."""
        sim = make_sim_result(
            equity=0.30,
            risk_adj_equity=0.25,  # < 0.42, so bluff_ok matters
            draw_premium=-0.05,
            ev_by_action={
                "fold": 0.0,
                "check": 5.0,
                "bet_half": 20.0,
                "bet_full": 18.0,
                "shove": -5.0,
            },
            street="turn",
            n_opponents=1,
        )
        state = make_state(
            street="turn",
            community=["Ts", "9h", "2c", "Kd"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        # Mock random to return < 0.25 so bluff gate opens
        with patch("bot.random.random", return_value=0.10):
            action = _postflop(state, sim, pos=0, bb=20)
        # Bluff allowed: bet_half (20) should win
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"Bluff should be allowed, got {action}"
        )

    def test_scenario_bluff_gate_blocked_by_random(self) -> None:
        """Bluff gate fails when random >= 0.25 (frequency cap)."""
        sim = make_sim_result(
            equity=0.30,
            risk_adj_equity=0.25,
            draw_premium=-0.05,
            ev_by_action={
                "fold": 0.0,
                "check": 5.0,
                "bet_half": 20.0,
                "bet_full": 18.0,
                "shove": -5.0,
            },
            street="turn",
            n_opponents=1,
        )
        state = make_state(
            street="turn",
            community=["Ts", "9h", "2c", "Kd"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        with patch("bot.random.random", return_value=0.50):
            action = _postflop(state, sim, pos=0, bb=20)
        # Bluff gate closed: bet options removed for risk_adj_equity < 0.42
        assert action != "raise" and not (isinstance(action, tuple) and action[0] == "raise"), (
            f"Bluff should be blocked, got {action}"
        )

    def test_scenario_bluff_gate_blocked_multiway(self) -> None:
        """Bluff gate requires heads-up (n_active==1). Multiway should block."""
        sim = make_sim_result(
            equity=0.30,
            risk_adj_equity=0.25,
            draw_premium=-0.05,
            ev_by_action={
                "fold": 0.0,
                "check": 5.0,
                "bet_half": 20.0,
                "bet_full": 18.0,
                "shove": -5.0,
            },
            street="turn",
            n_opponents=2,
        )
        state = make_state(
            street="turn",
            community=["Ts", "9h", "2c", "Kd"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            num_players=3,
            player_folded={"0": False, "1": False, "2": False},
        )
        with patch("bot.random.random", return_value=0.10):
            action = _postflop(state, sim, pos=0, bb=20)
        # Multiway => bluff gate fails => bets removed
        assert action == "check", f"Bluff blocked multiway, got {action}"

    def test_scenario_bluff_gate_blocked_not_btn(self) -> None:
        """Bluff gate requires BTN (pos==0). Other positions should block."""
        sim = make_sim_result(
            equity=0.30,
            risk_adj_equity=0.25,
            draw_premium=-0.05,
            ev_by_action={
                "fold": 0.0,
                "check": 5.0,
                "bet_half": 20.0,
                "bet_full": 18.0,
                "shove": -5.0,
            },
            street="turn",
            n_opponents=1,
        )
        state = make_state(
            street="turn",
            community=["Ts", "9h", "2c", "Kd"],
            pot=100,
            to_call=0,
            min_raise=40,
            chips=1000,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        with patch("bot.random.random", return_value=0.10):
            action = _postflop(state, sim, pos=2, bb=20)  # pos=2 = BB, not BTN
        assert action == "check", f"Bluff blocked out of position, got {action}"

    def test_scenario_pot_commitment_forces_call(self) -> None:
        """When to_call >= 60% of chips, pot commitment overrides fold."""
        sim = make_sim_result(
            equity=0.30,
            risk_adj_equity=0.25,
            continuation_value=0.30,
            ev_by_action={
                "fold": 0.0,
                "call": -10.0,
                "shove": -50.0,
            },
            street="river",
        )
        state = make_state(
            street="river",
            community=["Ts", "9h", "2c", "Kd", "3s"],
            pot=500,
            to_call=700,  # 70% of 1000 chips
            current_bet=700,
            min_raise=1400,
            chips=1000,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        action = _postflop(state, sim, pos=0, bb=20)
        # Pot commitment: call EV boosted to continuation_value * (pot + to_call) = 0.30 * 1200 = 360
        assert action == "call", f"Pot committed should call, got {action}"

    def test_scenario_bet_full_translation(self) -> None:
        """When bet_full is best EV, action should be ('raise', pot-size)."""
        sim = make_sim_result(
            equity=0.70,
            risk_adj_equity=0.65,
            ev_by_action={
                "fold": 0.0,
                "check": 10.0,
                "bet_half": 30.0,
                "bet_full": 50.0,  # highest
                "shove": 20.0,
            },
            street="turn",
        )
        state = make_state(
            street="turn",
            community=["Ts", "9h", "2c", "Kd"],
            pot=200,
            to_call=0,
            min_raise=40,
            chips=1000,
        )
        action = _postflop(state, sim, pos=0, bb=20)
        assert isinstance(action, tuple) and action[0] == "raise", (
            f"bet_full best EV should raise, got {action}"
        )
        # bet_full = max(min_raise, int(pot * 1.0)) = max(40, 200) = 200
        assert action[1] == 200, f"bet_full should be pot-sized, got {action[1]}"

    def test_scenario_shove_when_best_ev(self) -> None:
        """When shove has highest EV, should go all-in."""
        sim = make_sim_result(
            equity=0.90,
            risk_adj_equity=0.85,
            ev_by_action={
                "fold": 0.0,
                "check": 20.0,
                "bet_half": 40.0,
                "bet_full": 45.0,
                "shove": 80.0,
            },
            street="river",
        )
        state = make_state(
            street="river",
            community=["Ts", "9h", "2c", "Kd", "3s"],
            pot=500,
            to_call=0,
            min_raise=40,
            chips=1000,
        )
        action = _postflop(state, sim, pos=0, bb=20)
        assert action == "allin", f"Shove with highest EV should allin, got {action}"


# ═══════════════════════════════════════════════════════════════════
# SIM RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.postflop
class TestSimResult:
    """Verify SimResult can be constructed and fields are accessible."""

    def test_field_access(self) -> None:
        sim = make_sim_result(equity=0.75, variance=0.05, lambda_=0.10)
        assert sim.equity == 0.75
        assert sim.variance == 0.05
        assert sim.lambda_ == 0.10

    def test_ev_by_action_is_dict(self) -> None:
        sim = make_sim_result()
        assert isinstance(sim.ev_by_action, dict)

    def test_all_fields_present(self) -> None:
        sim = make_sim_result()
        expected_fields = [
            "equity", "variance", "cvar", "risk_adj_equity",
            "continuation_value", "draw_premium", "ev_by_action",
            "street", "n_opponents", "n_sims", "lambda_",
        ]
        for f in expected_fields:
            assert hasattr(sim, f), f"SimResult missing field: {f}"


# ═══════════════════════════════════════════════════════════════════
# MONTE CARLO EQUITY (thin wrapper)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.postflop
class TestMonteCarloEquity:
    """Tests for monte_carlo_equity() thin wrapper."""

    def test_no_opponents_returns_1(self) -> None:
        eq = monte_carlo_equity(["Ah", "Kd"], ["Ts", "9h", "2c"], 0)
        assert eq == 1.0

    def test_equity_bounded_0_1(self) -> None:
        eq = monte_carlo_equity(["Ah", "Kd"], ["Ts", "9h", "2c"], 1, n_sims=100)
        assert 0.0 <= eq <= 1.0

    def test_aces_vs_one_opponent_high_equity(self) -> None:
        # AA preflop vs 1 opponent should have ~80%+ equity
        random.seed(42)
        eq = monte_carlo_equity(["Ah", "As"], [], 1, n_sims=500)
        assert eq >= 0.70, f"AA vs 1 should have high equity, got {eq}"


# ═══════════════════════════════════════════════════════════════════
# RUN UNIFIED SIMULATION (sanity checks)
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.postflop
class TestRunUnifiedSimulation:
    """Sanity checks on the unified simulation output."""

    def test_returns_simresult_type(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        result = run_unified_simulation(state, bb=20, n_sims=50)
        assert isinstance(result, SimResult)

    def test_equity_bounded(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        result = run_unified_simulation(state, bb=20, n_sims=100)
        assert 0.0 <= result.equity <= 1.0
        assert result.variance >= 0.0
        assert 0.0 <= result.cvar <= 1.0

    def test_no_opponents_returns_perfect_equity(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": True},
        )
        result = run_unified_simulation(state, bb=20, n_sims=50)
        assert result.equity == 1.0
        assert result.n_opponents == 0

    def test_ev_by_action_has_fold(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        result = run_unified_simulation(state, bb=20, n_sims=50)
        assert "fold" in result.ev_by_action
        assert result.ev_by_action["fold"] == 0.0

    def test_lambda_scales_with_stack_depth(self) -> None:
        """Deep stacks should have lower lambda (more risk-tolerant)."""
        state_deep = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=2000,  # 100 BB
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        state_short = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=300,  # 15 BB
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        deep_result = run_unified_simulation(state_deep, bb=20, n_sims=50)
        short_result = run_unified_simulation(state_short, bb=20, n_sims=50)
        assert deep_result.lambda_ < short_result.lambda_, (
            f"Deep lambda {deep_result.lambda_} should be < short lambda {short_result.lambda_}"
        )

    def test_degenerate_state_too_few_cards(self) -> None:
        """When available cards < needed, simulation returns neutral SimResult."""
        # With NUM_DECKS=1 and many opponents, we can exhaust the deck
        bot.NUM_DECKS = 1
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="flop",
            community=["As", "9h", "2c"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=30,  # way too many
            player_folded={str(i): False for i in range(30)},
            player_chips={str(i): 1000 for i in range(30)},
            player_bets={str(i): 0 for i in range(30)},
            player_allin={str(i): False for i in range(30)},
        )
        result = run_unified_simulation(state, bb=20, n_sims=50)
        assert result.equity == 0.5

    def test_river_has_no_draw_premium(self) -> None:
        state = make_state(
            hole_cards=["Ah", "Kd"],
            street="river",
            community=["As", "9h", "2c", "Kd", "3s"],
            chips=1000,
            pot=60,
            to_call=0,
            min_raise=40,
            num_players=2,
            player_folded={"0": False, "1": False},
        )
        result = run_unified_simulation(state, bb=20, n_sims=50)
        assert result.draw_premium == 0.0
