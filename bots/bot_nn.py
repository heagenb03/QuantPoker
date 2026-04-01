"""
bot.py — Poker Bot Skeleton

All networking, parsing, and game-state tracking is handled for you.
Your ONLY job is to implement the `decide()` function at the bottom.

Usage:
    python bot.py [--host HOST] [--port PORT] [--name NAME]

You receive:
    state  — a GameState object with everything you need (see below)

You return one of:
    "fold"
    "check"         (only valid when to_call == 0)
    "call"
    "allin"
    ("raise", amount)   ← amount is the total bet level you want to set
"""

import socket
import json
import argparse
import random

# ─────────────────────────────────────────────
# GAME STATE OBJECT  (read-only, given to you)
# ─────────────────────────────────────────────

class GameState:
    """Everything you could ever want to know about the current decision."""

    def __init__(self, raw: dict, history: list, my_pid: int):
        # ── Identity ─────────────────────────
        self.my_pid        = my_pid

        # ── Cards ────────────────────────────
        self.hole_cards    = raw["hole_cards"]      # e.g. ["Ah","Kd"]
        self.community     = raw["community"]       # e.g. ["2c","7h","Td"]  (0–5 cards)
        self.street        = raw["street"]          # "preflop" | "flop" | "turn" | "river"

        # ── Money ────────────────────────────
        self.chips         = raw["chips"]           # your remaining chips
        self.pot           = raw["pot"]             # total pot size
        self.to_call       = raw["to_call"]         # chips you must add to call
        self.current_bet   = raw["current_bet"]     # highest bet this street
        self.min_raise     = raw["min_raise"]       # minimum legal raise total

        # ── Table ────────────────────────────
        self.num_players   = raw["num_players"]     # players in this hand
        self.player_bets   = raw["player_bets"]     # {pid: bet} this street
        self.player_chips  = raw["player_chips"]    # {pid: chips}
        self.player_folded = raw["player_folded"]   # {pid: bool}
        self.player_allin  = raw["player_allin"]    # {pid: bool}

        # ── History ──────────────────────────
        self.history       = history                # list of past action dicts

    # ── Derived helpers ──────────────────────

    @property
    def can_check(self):
        return self.to_call == 0

    @property
    def active_opponents(self):
        """Number of players still in the hand (not folded, not me)."""
        return sum(1 for pid, folded in self.player_folded.items()
                   if not folded and str(pid) != str(self.my_pid))

    @property
    def pot_odds(self):
        """Fraction of pot you need to invest to call (0.0 if check is free)."""
        if self.to_call == 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    @property
    def all_cards(self):
        """Your hole cards + community cards."""
        return self.hole_cards + self.community

    def __repr__(self):
        return (f"<GameState street={self.street} "
                f"hole={self.hole_cards} community={self.community} "
                f"pot={self.pot} chips={self.chips} to_call={self.to_call}>")


# ─────────────────────────────────────────────
# ════════════════════════════════════════════
#   YOUR BOT LOGIC — EDIT ONLY THIS SECTION
# ════════════════════════════════════════════
# ─────────────────────────────────────────────

def decide(state: GameState):
    street_map = {"preflop": 0.2, "flop": 0.4, "turn": 0.6, "river": 0.8}
    features = [
        state.chips / 1000.0,
        state.pot / 1000.0,
        state.to_call / 1000.0,
        street_map.get(state.street, 0.5)
    ]
    # Simple perceptron weights: fold, call/check, raise
    weights = [[0.5, -0.2, -0.8, 0.1],
               [0.1, 0.5, 0.2, 0.3],
               [0.2, 0.8, -0.5, 0.6]]
    scores = []
    for w in weights:
        score = sum(f*ww for f, ww in zip(features, w))
        scores.append(score)
        
    action_idx = scores.index(max(scores))
    if action_idx == 0:
        if state.can_check: return "check"
        return "fold"
    elif action_idx == 1:
        if state.can_check: return "check"
        return "call"
    else:
        return ("raise", min(state.min_raise, state.chips + state.current_bet))

# ─────────────────────────────────────────────
# BOT CLIENT  (networking — do not edit)
# ─────────────────────────────────────────────

class BotClient:
    def __init__(self, host, port, name="Bot"):
        self.host    = host
        self.port    = port
        self.name    = name
        self.pid     = None
        self.player_names = {}
        self.history = []
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._buf    = b''     # persistent recv buffer

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"[{self.name}] Connected to {self.host}:{self.port}")
        self.send({"type": "login", "name": self.name})

    def send(self, msg: dict):
        data = json.dumps(msg) + '\n'
        self.sock.sendall(data.encode())

    def recv(self):
        """Read exactly one message; leftovers stay buffered for next call."""
        while b'\n' not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        return json.loads(line.decode())

    def run(self):
        self.connect()
        while True:
            msg = self.recv()
            if msg is None:
                print(f"[{self.name}] Server disconnected.")
                break

            mtype = msg.get("type")

            # ── Welcome ──────────────────────────────────
            if mtype == "welcome":
                self.pid = msg["pid"]
                self.player_names = msg.get("player_names", {})
                print(f"[{self.name}] Assigned PID={self.pid}, "
                      f"chips={msg['chips']}, bb={msg['big_blind']}, "
                      f"players={msg['num_players']}")

            # ── Hole cards dealt ─────────────────────────
            elif mtype == "hole_cards":
                print(f"[{self.name}] Dealt: {msg['cards']}  "
                      f"chips={msg['chips']}  pot={msg['pot']}")
                self.history.append(msg)

            # ── Community cards ──────────────────────────
            elif mtype == "community_cards":
                print(f"[{self.name}] Board ({msg['street']}): {msg['cards']}  pot={msg['pot']}")
                self.history.append(msg)

            # ── It's your turn ───────────────────────────
            elif mtype == "action_request":
                state  = GameState(msg, self.history, self.pid)
                action = decide(state)

                # Normalise and validate the action
                response = self._build_response(action, state)
                print(f"[{self.name}] Action → {response}")
                self.send(response)
                self.history.append({"type": "my_action", **response})

            # ── Someone else acted ───────────────────────
            elif mtype == "player_action":
                pid = msg["pid"]
                name = self.player_names.get(str(pid), f"Player {pid}")
                act = msg["action"]
                amt = msg.get("amount", "")
                print(f"[{self.name}] {name} → {act} {amt}  chips={msg.get('chips','?')}")
                self.history.append(msg)

            # ── Showdown ─────────────────────────────────
            elif mtype == "showdown":
                winners_names = [self.player_names.get(str(w), f"Player {w}") for w in msg['winners']]
                print(f"[{self.name}] SHOWDOWN — winners: {winners_names}  "
                      f"pot={msg['pot']}  best hand: {msg['hand_name']}")
                for pid, cards in msg["hands"].items():
                    name = self.player_names.get(str(pid), f"Player {pid}")
                    print(f"           {name}: {cards}")
                print(f"           Stacks: {msg['stacks']}")
                self.history.append(msg)

            elif mtype == "winner":
                name = self.player_names.get(str(msg['pid']), f"Player {msg['pid']}")
                print(f"[{self.name}] {name} wins pot={msg['pot']} "
                      f"({msg['reason']})")
                print(f"           Stacks: {msg['stacks']}")

            elif mtype == "hand_start":
                dealer = self.player_names.get(str(msg['dealer']), f"Player {msg['dealer']}")
                sb = self.player_names.get(str(msg['sb']), f"Player {msg['sb']}")
                bb = self.player_names.get(str(msg['bb']), f"Player {msg['bb']}")
                print(f"[{self.name}] ── New hand ── dealer={dealer}  "
                      f"sb={sb}  bb={bb}")

            elif mtype == "game_over":
                winner = self.player_names.get(str(msg['winner']), f"Player {msg['winner']}") if msg['winner'] != -1 else "none"
                print(f"[{self.name}] GAME OVER — winner: {winner}")
                break

            else:
                # Unknown message — just log it
                print(f"[{self.name}] MSG: {msg}")

    def _build_response(self, action, state: GameState) -> dict:
        """Convert the decide() return value into a server message."""
        if isinstance(action, tuple):
            verb, amount = action
            amount = max(int(amount), state.min_raise)
            amount = min(amount, state.chips + state.current_bet)
            return {"action": verb, "amount": amount}

        action = action.lower()

        if action == "check":
            if not state.can_check:
                print(f"[{self.name}] WARNING: tried to check but must call {state.to_call} — folding")
                return {"action": "fold"}
            return {"action": "check"}

        if action == "allin":
            return {"action": "allin", "amount": state.chips + state.current_bet}

        if action in ("fold", "call"):
            return {"action": action}

        # Fallback
        print(f"[{self.name}] WARNING: unknown action '{action}' — folding")
        return {"action": "fold"}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Poker Bot")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--name", default="Bot")
    args = ap.parse_args()

    bot = BotClient(args.host, args.port, args.name)
    bot.run()
