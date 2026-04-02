"""
bot.py — Tournament Poker Bot
==============================
Entry point and global state hub. All logic lives in the poker/ package.

Architecture:
1. Parse history → _OpponentModel (VPIP, PFR, AF, showdown calibration)
2. Detect position from hand_start messages stored in history
3. Preflop: situation classifier → polarized ranges (position × table_size)
4. Postflop: run_unified_simulation() → SimResult → _postflop()

Usage:
    pip install eval7
    python bot.py [--host HOST] [--port PORT] [--name NAME]
"""

import argparse
import random  # re-exported so tests can patch via `bot.random.random`

# ── Module-level globals (kept here for test compatibility) ────────
# conftest resets these between tests as: bot.BIG_BLIND = 20, etc.
BIG_BLIND      = 20       # updated from "welcome"
DEALER_PID     = None     # updated from "hand_start"
NUM_DECKS      = 1        # updated from "welcome" via num_players
_HLEN          = 0        # index of last processed history message
BOT_NAME       = "Bot"    # updated from --name CLI arg
LOG_DECISIONS  = False    # set True via --log-decisions or POKER_DECISION_LOG=1

# ── Re-exports ─────────────────────────────────────────────────────
# Tests do `from bot import X` and `bot.X` — everything must be here.
from poker.constants import RANKS, SUITS, ALL_CARDS, RANK_IDX
from poker.game_state import GameState
from poker.eval import _eval, _eval_5card_multideck, _eval_multideck
from poker.simulation import (SimResult, _FOLD_TO_BET, _BET_WHEN_CHECKED,
                               run_unified_simulation, monte_carlo_equity)
from poker.position import get_position, players_behind_preflop, open_range_pct
from poker.preflop import (preflop_strength, _HAND_STRENGTHS,
                            open_strength_threshold, _preflop, _push_fold)
from poker.postflop import _postflop, _validate
from poker.opponent_model import _OpponentModel
from poker.decide import decide
from poker.client import BotClient

# ── Singleton opponent model ───────────────────────────────────────
# conftest resets this as: bot._OPP = bot._OpponentModel()
_OPP = _OpponentModel()


if __name__ == "__main__":
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()  # loads .env from the working directory into os.environ
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description="Poker Bot")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--name", default="Bot")
    ap.add_argument("--log-decisions", action="store_true",
                    help="Write per-decision JSON log to {name}_decisions.log")
    args = ap.parse_args()

    BOT_NAME = args.name
    LOG_DECISIONS = args.log_decisions or os.environ.get("POKER_DECISION_LOG") == "1"

    client = BotClient(args.host, args.port, args.name)
    client.run()
