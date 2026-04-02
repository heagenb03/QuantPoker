"""
conftest.py -- Shared fixtures and helpers for the poker bot test suite.

Usage:
    pytest                             # run all tests
    pytest -m preflop                  # run only preflop tests
    pytest -m postflop                 # run only postflop tests
    pytest -m opponent_model           # run only opponent model tests
    pytest -m validate                 # run only validate tests
    pytest -m integration              # run only integration tests
    pytest -k "scenario_name"          # run a specific scenario by name

    python test_bot.py                 # run all tests via script entry
    # --group and --scenario handled by test_bot.py __main__ block (not conftest hooks)
"""

import pytest
import bot


# ── Autouse fixture: reset all module-level globals between tests ──

@pytest.fixture(autouse=True)
def _reset_bot_globals() -> None:
    """Reset module globals and singleton state before every test."""
    bot.BIG_BLIND = 20
    bot.DEALER_PID = 0
    bot.NUM_DECKS = 1
    bot._HLEN = 0
    bot._OPP = bot._OpponentModel()
    yield
    # Teardown: reset again to avoid pollution
    bot.BIG_BLIND = 20
    bot.DEALER_PID = 0
    bot.NUM_DECKS = 1
    bot._HLEN = 0
    bot._OPP = bot._OpponentModel()


# ── Helper: Build a GameState from keyword overrides ───────────────

def make_state(
    hole_cards: list[str] | None = None,
    community: list[str] | None = None,
    street: str = "preflop",
    chips: int = 1000,
    pot: int = 30,
    to_call: int = 0,
    current_bet: int = 0,
    min_raise: int = 40,
    num_players: int = 4,
    player_bets: dict | None = None,
    player_chips: dict | None = None,
    player_folded: dict | None = None,
    player_allin: dict | None = None,
    my_pid: int = 0,
    history: list | None = None,
) -> bot.GameState:
    """
    Convenience factory for GameState with sensible defaults.
    All dict keys are strings to match the server protocol.
    """
    if hole_cards is None:
        hole_cards = ["Ah", "Kd"]
    if community is None:
        community = [] if street == "preflop" else ["Ts", "9h", "2c"]
    if player_bets is None:
        player_bets = {str(i): 0 for i in range(num_players)}
    if player_chips is None:
        player_chips = {str(i): chips for i in range(num_players)}
    if player_folded is None:
        player_folded = {str(i): False for i in range(num_players)}
    if player_allin is None:
        player_allin = {str(i): False for i in range(num_players)}
    if history is None:
        history = []

    raw = {
        "hole_cards": hole_cards,
        "community": community,
        "street": street,
        "chips": chips,
        "pot": pot,
        "to_call": to_call,
        "current_bet": current_bet,
        "min_raise": min_raise,
        "num_players": num_players,
        "player_bets": player_bets,
        "player_chips": player_chips,
        "player_folded": player_folded,
        "player_allin": player_allin,
    }
    return bot.GameState(raw, history, my_pid)


def make_sim_result(
    equity: float = 0.60,
    variance: float = 0.10,
    cvar: float = 0.20,
    risk_adj_equity: float = 0.55,
    continuation_value: float = 0.60,
    draw_premium: float = 0.0,
    ev_by_action: dict | None = None,
    street: str = "flop",
    n_opponents: int = 1,
    n_sims: int = 600,
    lambda_: float = 0.10,
) -> bot.SimResult:
    """Convenience factory for SimResult with sensible defaults."""
    if ev_by_action is None:
        ev_by_action = {
            "fold": 0.0,
            "check": 10.0,
            "call": 15.0,
            "bet_half": 20.0,
            "bet_full": 18.0,
            "shove": 5.0,
        }
    return bot.SimResult(
        equity=equity,
        variance=variance,
        cvar=cvar,
        risk_adj_equity=risk_adj_equity,
        continuation_value=continuation_value,
        draw_premium=draw_premium,
        ev_by_action=ev_by_action,
        street=street,
        n_opponents=n_opponents,
        n_sims=n_sims,
        lambda_=lambda_,
    )


def setup_opponent(pid: int, hands: int, vpip: int, pfr: int,
                   raises: int, calls: int) -> None:
    """Directly inject opponent stats into the singleton model."""
    bot._OPP._s[pid]['hands'] = hands
    bot._OPP._s[pid]['vpip'] = vpip
    bot._OPP._s[pid]['pfr'] = pfr
    bot._OPP._s[pid]['raises'] = raises
    bot._OPP._s[pid]['calls'] = calls
