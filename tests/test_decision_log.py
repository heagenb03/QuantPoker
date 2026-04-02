"""
tests/test_decision_log.py — Unit tests for poker/decision_log.py

Run with:
    python -m pytest tests/test_decision_log.py -v
"""

import io
import json

import pytest

import bot
import poker.decision_log as decision_log
from conftest import make_state, make_sim_result


# ── Fixture: capture log output to a StringIO buffer ─────────────────


@pytest.fixture
def capture_log(monkeypatch):
    """Enable logging and redirect output to an in-memory buffer."""
    monkeypatch.setattr(bot, "LOG_DECISIONS", True)
    monkeypatch.setattr(bot, "BOT_NAME", "test_bot")
    buf = io.StringIO()
    monkeypatch.setattr(decision_log, "_log_fh", buf)
    monkeypatch.setattr(decision_log, "_ensure_open", lambda: True)
    return buf


def _parse_last_line(buf: io.StringIO) -> dict:
    """Return the last JSON-lines entry from the buffer."""
    lines = [l for l in buf.getvalue().strip().splitlines() if l]
    assert lines, "No log lines written"
    return json.loads(lines[-1])


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_logging_disabled_no_write() -> None:
    """With LOG_DECISIONS=False, log_decision() is a no-op."""
    state = make_state()
    decision_log.log_decision(state, pos=2, bb=20, action="fold")
    # No file handle opened — _log_fh should still be None
    assert decision_log._log_fh is None


@pytest.mark.unit
def test_preflop_fields_present(capture_log) -> None:
    """Preflop log entry has all expected fields and no sim fields."""
    state = make_state(hole_cards=["Ah", "Kd"], street="preflop")
    decision_log.log_decision(state, pos=2, bb=20, action="fold")
    entry = _parse_last_line(capture_log)

    assert entry["street"] == "preflop"
    assert entry["hole_cards"] == ["Ah", "Kd"]
    assert entry["community"] == []
    assert entry["position"] == 2
    assert entry["action"] == "fold"
    assert "stack_bb" in entry
    assert "pot" in entry
    assert "to_call" in entry
    assert "pot_odds" in entry
    assert "archetypes" in entry
    assert "hand" in entry
    # No sim fields on preflop
    assert "equity" not in entry
    assert "risk_adj_equity" not in entry


@pytest.mark.unit
def test_postflop_includes_sim_result_fields(capture_log) -> None:
    """Postflop log entry includes all SimResult fields."""
    state = make_state(
        hole_cards=["Ah", "Kd"],
        street="flop",
        community=["As", "9h", "2c"],
    )
    sim = make_sim_result(equity=0.72, risk_adj_equity=0.68, draw_premium=0.05)
    decision_log.log_decision(state, pos=0, bb=20, action=("raise", 80), sim_result=sim)
    entry = _parse_last_line(capture_log)

    assert entry["equity"] == pytest.approx(0.72, abs=1e-4)
    assert entry["risk_adj_equity"] == pytest.approx(0.68, abs=1e-4)
    assert entry["draw_premium"] == pytest.approx(0.05, abs=1e-4)
    assert "cvar" in entry
    assert "variance" in entry
    assert "continuation_value" in entry
    assert "ev_by_action" in entry
    assert "lambda" in entry
    assert "n_sims" in entry


@pytest.mark.unit
def test_action_tuple_serialized_as_list(capture_log) -> None:
    """Tuple actions like ('raise', 100) are written as JSON arrays."""
    state = make_state()
    decision_log.log_decision(state, pos=1, bb=20, action=("raise", 100))
    entry = _parse_last_line(capture_log)
    assert entry["action"] == ["raise", 100]


@pytest.mark.unit
def test_hand_counter_in_entry(capture_log) -> None:
    """hand field reflects current _hand_counter."""
    decision_log.increment_hand()
    decision_log.increment_hand()
    state = make_state()
    decision_log.log_decision(state, pos=0, bb=20, action="check")
    entry = _parse_last_line(capture_log)
    assert entry["hand"] == 2


@pytest.mark.unit
def test_archetypes_excludes_folded_opponents(capture_log) -> None:
    """Folded opponents are omitted from archetypes dict."""
    state = make_state(
        num_players=3,
        player_folded={"0": False, "1": True, "2": False},
        my_pid=0,
    )
    decision_log.log_decision(state, pos=0, bb=20, action="check")
    entry = _parse_last_line(capture_log)
    # pid 0 = self, pid 1 = folded → only pid 2 should appear
    assert "1" not in entry["archetypes"]
    assert "2" in entry["archetypes"]


@pytest.mark.unit
def test_archetypes_excludes_self(capture_log) -> None:
    """Bot's own pid is never in the archetypes dict."""
    state = make_state(num_players=2, my_pid=0)
    decision_log.log_decision(state, pos=0, bb=20, action="call")
    entry = _parse_last_line(capture_log)
    assert "0" not in entry["archetypes"]


@pytest.mark.unit
def test_multiple_decisions_multiple_lines(capture_log) -> None:
    """Each log_decision call produces exactly one new JSON line."""
    state = make_state()
    decision_log.log_decision(state, pos=0, bb=20, action="fold")
    decision_log.log_decision(state, pos=1, bb=20, action="call")
    lines = [l for l in capture_log.getvalue().strip().splitlines() if l]
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "fold"
    assert json.loads(lines[1])["action"] == "call"
