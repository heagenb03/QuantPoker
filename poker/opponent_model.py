"""
poker/opponent_model.py — _OpponentModel class.

Tracks VPIP, PFR, AF, showdowns per pid. Classifies into archetypes.
Quant parallel: factor model + alpha signal construction.
Showdown reveals = labeled training data (unique to this server).
"""

from collections import defaultdict

from poker.config import (
    FOLD_TO_BET as _FOLD_TO_BET, BET_WHEN_CHECKED as _BET_WHEN_CHECKED,
    MIN_HANDS_FOR_ARCHETYPE, MIN_HANDS_FOR_STATS,
    DEFAULT_VPIP, DEFAULT_PFR, DEFAULT_AF,
    NIT_VPIP_MAX, STATION_VPIP_MIN, STATION_AF_MAX, MANIAC_VPIP_MIN, MANIAC_AF_MIN,
    ARCHETYPE_RANGE_WIDTH, BETA_PSEUDOCOUNT, BETA_PSEUDOCOUNT_INITIAL,
)


class _OpponentModel:
    """
    Incremental opponent model built from state.history messages.
    Key signals (minimum 10 hands before trusting a read):
      VPIP   — how often they voluntarily invest preflop
      PFR    — how often they raise preflop
      AF     — aggression factor (raises / calls)
      showdown cards — ground-truth range calibration
    """

    def __init__(self):
        self._s = defaultdict(lambda: {
            'hands': 0,
            'vpip':  0,
            'pfr':   0,
            'raises': 0,
            'calls':  0,
            'showdown': [],        # list of hole-card pairs seen at showdown
            # Bayesian frequency tracking (Beta distribution parameters)
            'ftb_alpha': 0.0,      # fold-to-bet successes
            'ftb_beta':  0.0,      # fold-to-bet failures (call/raise)
            'bwc_alpha': 0.0,      # bet-when-checked successes
            'bwc_beta':  0.0,      # bet-when-checked failures (check)
            'ftb_seeded': False,         # whether full archetype prior has been seeded
            'bwc_seeded': False,
            'ftb_weakly_seeded': False,  # weak unknown prior applied on first postflop obs
            'bwc_weakly_seeded': False,
        })
        self._vpip_this_hand = set()
        self._pfr_this_hand  = set()
        self._had_bet_this_street: bool = False
        self._current_street: str = ''
        self._in_hand_raises: defaultdict = defaultdict(int)

    def update(self, messages: list) -> None:
        """Process only the NEW messages appended since last call."""
        for msg in messages:
            t = msg.get('type')

            if t == 'hand_start':
                # Commit VPIP / PFR counts from the hand that just ended
                for pid in self._vpip_this_hand:
                    self._s[pid]['vpip'] += 1
                for pid in self._pfr_this_hand:
                    self._s[pid]['pfr']  += 1
                # Count one hand seen for every player still at the table
                for pid_str in msg.get('stacks', {}):
                    self._s[int(pid_str)]['hands'] += 1
                self._vpip_this_hand = set()
                self._pfr_this_hand  = set()
                # Reset street tracking for new hand
                self._had_bet_this_street = False
                self._current_street = ''
                self._in_hand_raises.clear()

            elif t == 'player_action':
                pid    = msg['pid']
                action = msg['action']
                street = msg.get('street', '')

                # Detect street transition → reset bet tracking
                if street != self._current_street:
                    self._had_bet_this_street = False
                    self._current_street = street

                if street == 'preflop':
                    if action in ('call', 'raise', 'allin'):
                        self._vpip_this_hand.add(pid)
                    if action in ('raise', 'allin'):
                        self._pfr_this_hand.add(pid)

                # Bayesian updates — postflop only
                if street in ('flop', 'turn', 'river'):
                    if self._had_bet_this_street:
                        # Opponent faces a bet
                        if action == 'fold':
                            self._reseed_priors(pid, 'ftb')
                            self._s[pid]['ftb_alpha'] += 1
                        elif action in ('call', 'raise', 'allin'):
                            self._reseed_priors(pid, 'ftb')
                            self._s[pid]['ftb_beta'] += 1
                    else:
                        # No bet yet this street
                        if action in ('raise', 'bet', 'allin'):
                            self._reseed_priors(pid, 'bwc')
                            self._s[pid]['bwc_alpha'] += 1
                        elif action == 'check':
                            self._reseed_priors(pid, 'bwc')
                            self._s[pid]['bwc_beta'] += 1

                    # After processing, mark if a bet occurred
                    if action in ('raise', 'bet', 'allin'):
                        self._had_bet_this_street = True

                # Global aggression tracking (all streets)
                # Only count voluntary aggressive/passive actions — fold and check
                # are neutral and must not inflate the call count (which deflates AF).
                if action in ('raise', 'allin', 'bet'):
                    self._s[pid]['raises'] += 1
                    self._in_hand_raises[pid] += 1
                elif action == 'call':
                    self._s[pid]['calls']  += 1

            elif t == 'showdown':
                # Ground-truth hole cards — the richest signal available.
                # Tells us exactly what they called a raise with, etc.
                for pid_str, info in msg.get('hands', {}).items():
                    cards = info.get('cards', [])
                    if cards:
                        self._s[int(pid_str)]['showdown'].append(cards)

    # ── Bayesian frequency accessors ─────────────────────────────

    def _reseed_priors(self, pid: int, param: str) -> None:
        """Seed Beta prior from archetype lookup on first observation.

        Deferred until archetype is not 'unknown' — seeding from the
        unknown prior (0.40/0.35) before hand 10 would lock in a stale
        prior that persists even after the player reclassifies. Holding
        off means the early observations accumulate without a prior, and
        the prior seeds from the correct archetype the moment it stabilises.

        Uses pseudocount of 5 so the prior is worth ~5 observations.
        *param* is 'ftb' or 'bwc'.

        NOTE: seeding (=) must happen before the caller increments (+=1).
        Do not reorder those two lines.
        """
        seeded_key = f'{param}_seeded'
        if self._s[pid][seeded_key]:
            return
        arch = self.archetype(pid)
        if arch == 'unknown':
            # Apply a weak prior from the unknown archetype on the first postflop
            # observation so that early data has a prior to anchor to rather than
            # being used as raw counts with no smoothing.  Do NOT mark as seeded —
            # the full archetype-specific prior is still applied once the archetype
            # stabilises at ≥10 hands.
            weakly_key = f'{param}_weakly_seeded'
            if not self._s[pid][weakly_key]:
                prior_mean = _FOLD_TO_BET['unknown'] if param == 'ftb' else _BET_WHEN_CHECKED['unknown']
                self._s[pid][f'{param}_alpha'] += prior_mean * BETA_PSEUDOCOUNT_INITIAL
                self._s[pid][f'{param}_beta'] += (1.0 - prior_mean) * BETA_PSEUDOCOUNT_INITIAL
                self._s[pid][weakly_key] = True
            return   # defer full seeding until archetype stabilises (≥10 hands)
        if param == 'ftb':
            prior_mean = _FOLD_TO_BET.get(arch, 0.40)
        else:
            prior_mean = _BET_WHEN_CHECKED.get(arch, 0.35)
        # Add pseudocount on top of any raw observations already accumulated
        # while archetype was 'unknown'. This preserves early data rather
        # than overwriting it.
        self._s[pid][f'{param}_alpha'] += prior_mean * BETA_PSEUDOCOUNT
        self._s[pid][f'{param}_beta'] += (1.0 - prior_mean) * BETA_PSEUDOCOUNT
        self._s[pid][seeded_key] = True

    def fold_to_bet_est(self, pid: int) -> float:
        """Bayesian estimate of how often *pid* folds to a postflop bet.

        Pre-archetype-stabilisation (< 10 hands): raw observation counts
        are used directly (no pseudocount prior yet).
        Post-stabilisation: pseudocount prior has been added via _reseed_priors,
        so the posterior mean is (prior + observations) / total.
        Falls back to archetype lookup only when zero observations exist.
        """
        s = self._s[pid]
        total = s['ftb_alpha'] + s['ftb_beta']
        if total <= 0:
            return _FOLD_TO_BET.get(self.archetype(pid), 0.40)
        return s['ftb_alpha'] / total

    def bet_when_checked_est(self, pid: int) -> float:
        """Bayesian estimate of how often *pid* bets when checked to.

        See fold_to_bet_est for the seeding/prior semantics.
        """
        s = self._s[pid]
        total = s['bwc_alpha'] + s['bwc_beta']
        if total <= 0:
            return _BET_WHEN_CHECKED.get(self.archetype(pid), 0.35)
        return s['bwc_alpha'] / total

    # ── Derived stats ──────────────────────────────────────────────

    def vpip(self, pid: int) -> float:
        s = self._s[pid]
        return s['vpip'] / s['hands'] if s['hands'] >= MIN_HANDS_FOR_STATS else DEFAULT_VPIP

    def pfr(self, pid: int) -> float:
        s = self._s[pid]
        return s['pfr'] / s['hands'] if s['hands'] >= MIN_HANDS_FOR_STATS else DEFAULT_PFR

    def af(self, pid: int) -> float:
        """Aggression factor: raises / calls. >2 = aggressive."""
        s = self._s[pid]
        total = s['raises'] + s['calls']
        return s['raises'] / max(s['calls'], 1) if total >= MIN_HANDS_FOR_STATS else DEFAULT_AF

    def archetype(self, pid: int) -> str:
        """
        Classify into one of four exploitable archetypes.
        Requires ≥10 hands; returns 'unknown' before that.

        Archetype → counter-strategy:
          nit     → steal often, fold to their aggression
          station → value bet relentlessly, never bluff
          maniac  → tighten preflop, trap with strong hands
          tag     → GTO-approximate play
        """
        if self._s[pid]['hands'] < MIN_HANDS_FOR_ARCHETYPE:
            return 'unknown'
        v  = self.vpip(pid)
        af = self.af(pid)
        if v < NIT_VPIP_MAX:                          return 'nit'
        if v >= STATION_VPIP_MIN and af < STATION_AF_MAX:  return 'station'
        if v >= MANIAC_VPIP_MIN  and af >= MANIAC_AF_MIN:  return 'maniac'
        return 'tag'

    def range_width(self, pid: int) -> float:
        """Estimated fraction of hands this opponent plays."""
        return ARCHETYPE_RANGE_WIDTH[self.archetype(pid)]

    def in_hand_aggression(self, pid: int) -> int:
        """Number of aggressive actions (raise/allin/bet) by *pid* in the current hand."""
        return self._in_hand_raises[pid]
