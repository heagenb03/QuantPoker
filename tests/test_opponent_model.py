"""
tests/test_opponent_model.py -- Opponent model tracking and archetype classification.

Run modes:
    pytest -m opponent_model
    pytest tests/test_opponent_model.py -v
"""

import pytest

import bot
from conftest import setup_opponent


def _flop_action_msgs(pid: int, actions: list[tuple[str, str]]) -> list[dict]:
    """Build a sequence of postflop player_action messages.

    Each element in *actions* is (acting_pid_or_'other', action_str).
    Wraps in a hand_start so the model sees street transitions.
    """
    msgs: list[dict] = [
        {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
         "stacks": {"0": 1000, str(pid): 1000, "99": 1000}},
    ]
    for actor, action in actions:
        msgs.append({
            "type": "player_action",
            "pid": int(actor),
            "action": action,
            "street": "flop",
        })
    return msgs


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

    def test_call_increments_calls_fold_and_check_do_not(self) -> None:
        """Only 'call' increments the calls counter. Fold and check are neutral."""
        messages = [
            {"type": "player_action", "pid": 1, "action": "call", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "fold", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "check", "street": "flop"},
        ]
        bot._OPP.update(messages)
        assert bot._OPP._s[1]["calls"] == 1


# ═══════════════════════════════════════════════════════════════════
# BAYESIAN FREQUENCY MODEL
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.opponent_model
class TestBayesianFrequencyModel:
    """Tests for per-opponent Bayesian fold-to-bet / bet-when-checked tracking."""

    # ── 1. Defaults fall back to archetype lookup ─────────────────

    def test_fold_to_bet_default_returns_archetype_value(self) -> None:
        """With no postflop observations, fold_to_bet_est equals the
        archetype lookup from _FOLD_TO_BET."""
        for arch, expected in [("nit", 0.65), ("tag", 0.45),
                               ("station", 0.08), ("maniac", 0.20),
                               ("unknown", 0.40)]:
            pid = hash(arch) % 1000 + 100  # unique pid per archetype
            if arch == "nit":
                setup_opponent(pid=pid, hands=20, vpip=3, pfr=2,
                               raises=5, calls=10)
            elif arch == "tag":
                setup_opponent(pid=pid, hands=20, vpip=5, pfr=3,
                               raises=8, calls=8)
            elif arch == "station":
                setup_opponent(pid=pid, hands=20, vpip=8, pfr=2,
                               raises=3, calls=12)
            elif arch == "maniac":
                setup_opponent(pid=pid, hands=20, vpip=8, pfr=6,
                               raises=25, calls=5)
            else:  # unknown
                setup_opponent(pid=pid, hands=5, vpip=3, pfr=2,
                               raises=5, calls=5)

            assert bot._OPP.archetype(pid) == arch, f"setup error for {arch}"
            result = bot._OPP.fold_to_bet_est(pid)
            assert abs(result - expected) < 1e-9, (
                f"fold_to_bet_est for {arch}: got {result}, expected {expected}"
            )

    def test_bet_when_checked_default_returns_archetype_value(self) -> None:
        """With no postflop observations, bet_when_checked_est equals the
        archetype lookup from _BET_WHEN_CHECKED."""
        for arch, expected in [("nit", 0.20), ("tag", 0.45),
                               ("station", 0.35), ("maniac", 0.75),
                               ("unknown", 0.35)]:
            pid = hash(arch) % 1000 + 200
            if arch == "nit":
                setup_opponent(pid=pid, hands=20, vpip=3, pfr=2,
                               raises=5, calls=10)
            elif arch == "tag":
                setup_opponent(pid=pid, hands=20, vpip=5, pfr=3,
                               raises=8, calls=8)
            elif arch == "station":
                setup_opponent(pid=pid, hands=20, vpip=8, pfr=2,
                               raises=3, calls=12)
            elif arch == "maniac":
                setup_opponent(pid=pid, hands=20, vpip=8, pfr=6,
                               raises=25, calls=5)
            else:
                setup_opponent(pid=pid, hands=5, vpip=3, pfr=2,
                               raises=5, calls=5)

            result = bot._OPP.bet_when_checked_est(pid)
            assert abs(result - expected) < 1e-9, (
                f"bet_when_checked_est for {arch}: got {result}, expected {expected}"
            )

    # ── 2. Fold-to-bet updates on postflop fold/call ──────────────

    def test_fold_to_bet_updates_on_postflop_fold(self) -> None:
        """When opponent faces a bet on the flop and folds,
        ftb_alpha should increment (raising fold-to-bet estimate)."""
        pid = 5
        setup_opponent(pid=pid, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        # Someone bets, then pid folds
        msgs = _flop_action_msgs(pid, [
            (99, "bet"),    # opponent 99 bets — sets _had_bet_this_street
            (pid, "fold"),  # pid folds facing bet — ftb_alpha += 1
        ])
        bot._OPP.update(msgs)
        s = bot._OPP._s[pid]
        assert s.get("ftb_alpha", 0) > 0, "ftb_alpha should have incremented"

    def test_fold_to_bet_updates_on_postflop_call(self) -> None:
        """When opponent faces a bet on the flop and calls,
        ftb_beta should increment (lowering fold-to-bet estimate)."""
        pid = 6
        setup_opponent(pid=pid, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        msgs = _flop_action_msgs(pid, [
            (99, "bet"),
            (pid, "call"),
        ])
        bot._OPP.update(msgs)
        s = bot._OPP._s[pid]
        assert s.get("ftb_beta", 0) > 0, "ftb_beta should have incremented"

    # ── 3. Bet-when-checked updates ───────────────────────────────

    def test_bet_when_checked_updates_on_postflop_bet(self) -> None:
        """When no bet has occurred this street and opponent bets,
        bwc_alpha should increment."""
        pid = 7
        setup_opponent(pid=pid, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        # pid bets into an unchecked-to street
        msgs = _flop_action_msgs(pid, [
            (pid, "bet"),
        ])
        bot._OPP.update(msgs)
        s = bot._OPP._s[pid]
        assert s.get("bwc_alpha", 0) > 0, "bwc_alpha should have incremented"

    def test_bet_when_checked_updates_on_postflop_check(self) -> None:
        """When no bet has occurred this street and opponent checks,
        bwc_beta should increment."""
        pid = 8
        setup_opponent(pid=pid, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        msgs = _flop_action_msgs(pid, [
            (pid, "check"),
        ])
        bot._OPP.update(msgs)
        s = bot._OPP._s[pid]
        assert s.get("bwc_beta", 0) > 0, "bwc_beta should have incremented"

    # ── 4. Preflop actions do NOT touch Beta params ───────────────

    def test_preflop_actions_do_not_update_beta(self) -> None:
        """Preflop raise/fold/call must NOT touch ftb/bwc counters."""
        pid = 9
        msgs = [
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, "9": 1000}},
            {"type": "player_action", "pid": pid, "action": "raise",
             "street": "preflop"},
            {"type": "player_action", "pid": pid, "action": "fold",
             "street": "preflop"},
        ]
        bot._OPP.update(msgs)
        s = bot._OPP._s[pid]
        assert s.get("ftb_alpha", 0) == 0
        assert s.get("ftb_beta", 0) == 0
        assert s.get("bwc_alpha", 0) == 0
        assert s.get("bwc_beta", 0) == 0

    # ── 5. Weak prior: early observations have Bayesian smoothing ─────

    def test_weak_prior_applied_on_first_postflop_obs(self) -> None:
        """First postflop observation on an unknown opponent seeds a weak prior
        so subsequent observations update a proper Beta, not raw counts."""
        pid = 20
        setup_opponent(pid=pid, hands=5, vpip=2, pfr=1, raises=1, calls=1)
        assert bot._OPP.archetype(pid) == "unknown"

        # One fold facing a bet on the flop
        msgs = _flop_action_msgs(pid, [(99, "bet"), (pid, "fold")])
        bot._OPP.update(msgs)

        s = bot._OPP._s[pid]
        # Alpha should include weak prior + 1 fold observation
        from poker.config import BETA_PSEUDOCOUNT_INITIAL, FOLD_TO_BET
        prior_a = FOLD_TO_BET['unknown'] * BETA_PSEUDOCOUNT_INITIAL   # 0.40 * 2.0 = 0.80
        assert abs(s['ftb_alpha'] - (prior_a + 1.0)) < 1e-9, (
            f"ftb_alpha should be prior({prior_a}) + 1 fold = {prior_a + 1.0}, got {s['ftb_alpha']}"
        )
        assert s['ftb_weakly_seeded'] is True

    def test_weak_prior_not_applied_twice(self) -> None:
        """A second postflop observation must not re-add the weak prior."""
        pid = 21
        setup_opponent(pid=pid, hands=5, vpip=2, pfr=1, raises=1, calls=1)
        from poker.config import BETA_PSEUDOCOUNT_INITIAL, FOLD_TO_BET
        prior_a = FOLD_TO_BET['unknown'] * BETA_PSEUDOCOUNT_INITIAL

        # Two folds facing a bet
        msgs = _flop_action_msgs(pid, [
            (99, "bet"), (pid, "fold"),
            (99, "bet"), (pid, "fold"),
        ])
        bot._OPP.update(msgs)

        s = bot._OPP._s[pid]
        # Alpha = prior + 2 folds (not prior + prior + 2)
        assert abs(s['ftb_alpha'] - (prior_a + 2.0)) < 1e-9, (
            f"ftb_alpha should be {prior_a + 2.0}, got {s['ftb_alpha']}"
        )

    def test_single_fold_obs_shifts_estimate_above_prior(self) -> None:
        """After one fold observation the estimate should exceed the unknown prior."""
        pid = 22
        setup_opponent(pid=pid, hands=5, vpip=2, pfr=1, raises=1, calls=1)

        msgs = _flop_action_msgs(pid, [(99, "bet"), (pid, "fold")])
        bot._OPP.update(msgs)

        est = bot._OPP.fold_to_bet_est(pid)
        assert est > 0.40, f"After one fold, estimate {est:.3f} should exceed unknown prior 0.40"
        assert est < 1.0

    def test_bwc_weak_prior_applied_on_first_postflop_bet(self) -> None:
        """First postflop bet/check observation seeds bwc weak prior symmetrically."""
        pid = 23
        setup_opponent(pid=pid, hands=5, vpip=2, pfr=1, raises=1, calls=1)
        assert bot._OPP.archetype(pid) == "unknown"

        # pid bets into an unchecked street
        msgs = _flop_action_msgs(pid, [(pid, "bet")])
        bot._OPP.update(msgs)

        s = bot._OPP._s[pid]
        from poker.config import BETA_PSEUDOCOUNT_INITIAL, BET_WHEN_CHECKED
        prior_a = BET_WHEN_CHECKED['unknown'] * BETA_PSEUDOCOUNT_INITIAL  # 0.35 * 2.0 = 0.70
        assert abs(s['bwc_alpha'] - (prior_a + 1.0)) < 1e-9, (
            f"bwc_alpha should be prior({prior_a}) + 1 bet = {prior_a + 1.0}, got {s['bwc_alpha']}"
        )
        assert s['bwc_weakly_seeded'] is True

    # ── 6. Convergence: observations dominate prior ───────────────

    def test_convergence_dominates_prior(self) -> None:
        """After 40 folds and 10 calls (true rate ~0.80), the Bayesian
        estimate should be closer to 0.80 than to the tag prior of 0.45."""
        pid = 10
        # Set up as tag (prior ftb = 0.45)
        setup_opponent(pid=pid, hands=20, vpip=5, pfr=3, raises=8, calls=8)
        assert bot._OPP.archetype(pid) == "tag"

        # Feed 40 folds and 10 calls facing a bet
        msgs: list[dict] = [
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, str(pid): 1000, "99": 1000}},
        ]
        for i in range(50):
            # Each iteration: someone bets, then pid responds
            msgs.append({"type": "player_action", "pid": 99,
                         "action": "bet", "street": "flop"})
            action = "fold" if i < 40 else "call"
            msgs.append({"type": "player_action", "pid": pid,
                         "action": action, "street": "flop"})
            # New hand to reset street state
            msgs.append({"type": "hand_start", "dealer": 0, "sb": 1,
                         "bb": 2,
                         "stacks": {"0": 1000, str(pid): 1000, "99": 1000}})

        bot._OPP.update(msgs)

        est = bot._OPP.fold_to_bet_est(pid)
        tag_prior = 0.45
        true_rate = 0.80
        # Estimate should be closer to 0.80 than to 0.45
        assert abs(est - true_rate) < abs(est - tag_prior), (
            f"After 50 observations, estimate {est:.3f} should be closer to "
            f"true rate {true_rate} than prior {tag_prior}"
        )


# ═══════════════════════════════════════════════════════════════════
# IN-HAND AGGRESSION TRACKING
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.opponent_model
class TestInHandAggression:
    """Tests for per-hand aggressive action counter and reset behaviour."""

    def test_fresh_pid_has_zero_aggression(self) -> None:
        assert bot._OPP.in_hand_aggression(1) == 0

    def test_raise_increments_in_hand_aggression(self) -> None:
        msgs = [{"type": "player_action", "pid": 1, "action": "raise", "street": "preflop"}]
        bot._OPP.update(msgs)
        assert bot._OPP.in_hand_aggression(1) == 1

    def test_bet_and_allin_also_increment(self) -> None:
        msgs = [
            {"type": "player_action", "pid": 1, "action": "bet",   "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "allin", "street": "turn"},
        ]
        bot._OPP.update(msgs)
        assert bot._OPP.in_hand_aggression(1) == 2

    def test_fold_check_call_do_not_increment(self) -> None:
        msgs = [
            {"type": "player_action", "pid": 1, "action": "call",  "street": "preflop"},
            {"type": "player_action", "pid": 1, "action": "check", "street": "flop"},
            {"type": "player_action", "pid": 1, "action": "fold",  "street": "turn"},
        ]
        bot._OPP.update(msgs)
        assert bot._OPP.in_hand_aggression(1) == 0

    def test_resets_on_hand_start(self) -> None:
        msgs = [
            {"type": "player_action", "pid": 1, "action": "raise", "street": "preflop"},
            {"type": "player_action", "pid": 1, "action": "bet",   "street": "flop"},
        ]
        bot._OPP.update(msgs)
        assert bot._OPP.in_hand_aggression(1) == 2

        bot._OPP.update([
            {"type": "hand_start", "dealer": 0, "sb": 1, "bb": 2,
             "stacks": {"0": 1000, "1": 1000}},
        ])
        assert bot._OPP.in_hand_aggression(1) == 0

    def test_independent_per_pid(self) -> None:
        msgs = [
            {"type": "player_action", "pid": 1, "action": "raise", "street": "preflop"},
            {"type": "player_action", "pid": 2, "action": "raise", "street": "preflop"},
            {"type": "player_action", "pid": 2, "action": "bet",   "street": "flop"},
        ]
        bot._OPP.update(msgs)
        assert bot._OPP.in_hand_aggression(1) == 1
        assert bot._OPP.in_hand_aggression(2) == 2
