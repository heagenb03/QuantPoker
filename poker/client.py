"""
poker/client.py — BotClient: TCP networking and server protocol.

Connects to the poker server, handles message routing, and calls decide()
for every action_request.
"""

import socket
import json

from poker.game_state import GameState
from poker.decide import decide
from poker.position import get_position


class BotClient:
    def __init__(self, host, port, name="Bot"):
        self.host    = host
        self.port    = port
        self.name    = name
        self.pid     = None
        self.history = []
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._buf    = b''

    def connect(self):
        self.sock.connect((self.host, self.port))
        print(f"[{self.name}] Connected to {self.host}:{self.port}")

    def send(self, msg: dict):
        self.sock.sendall((json.dumps(msg) + '\n').encode())

    def recv(self):
        while b'\n' not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        return json.loads(line.decode())

    def run(self):
        import bot  # to write bot.BIG_BLIND, bot.DEALER_PID, bot.NUM_DECKS, read bot._OPP
        self.connect()

        while True:
            msg = self.recv()
            if msg is None:
                print(f"[{self.name}] Server disconnected.")
                break

            t = msg.get("type")

            # ── EDIT 1: capture big_blind from welcome ─────────────
            if t == "welcome":
                self.pid      = msg["pid"]
                bot.BIG_BLIND = msg["big_blind"]
                n_p = msg["num_players"]
                bot.NUM_DECKS = max(1, min(4, (n_p // 5) + 1))
                print(f"[{self.name}] PID={self.pid}  chips={msg['chips']}  "
                      f"bb={bot.BIG_BLIND}  players={n_p}  decks={bot.NUM_DECKS}")

            # ── EDIT 2: capture dealer from hand_start + append it ─
            elif t == "hand_start":
                bot.DEALER_PID = msg["dealer"]
                self.history.append(msg)       # opponent model needs this
                print(f"[{self.name}] ── New hand ── "
                      f"dealer={msg['dealer']}  sb={msg['sb']}  bb={msg['bb']}")

            elif t == "hole_cards":
                print(f"[{self.name}] Dealt: {msg['cards']}  "
                      f"chips={msg['chips']}  pot={msg['pot']}")
                self.history.append(msg)

            elif t == "community_cards":
                print(f"[{self.name}] Board ({msg['street']}): "
                      f"{msg['cards']}  pot={msg['pot']}")
                self.history.append(msg)

            elif t == "action_request":
                state  = GameState(msg, self.history, self.pid)
                action = decide(state)
                resp   = self._build_response(action, state)
                print(f"[{self.name}] Action → {resp}  "
                      f"[pos={get_position(self.pid, bot.DEALER_PID, state.num_players)} "
                      f"bb={state.chips // bot.BIG_BLIND}BB "
                      f"street={state.street}]")
                self.send(resp)
                self.history.append({"type": "my_action", **resp})

            elif t == "player_action":
                pid = msg["pid"]
                arch = bot._OPP.archetype(pid)
                print(f"[{self.name}] Player {pid} ({arch}) → "
                      f"{msg['action']} {msg.get('amount','')}  "
                      f"chips={msg.get('chips','?')}")
                self.history.append(msg)

            elif t == "showdown":
                print(f"[{self.name}] SHOWDOWN — winners: {msg['winners']}  "
                      f"pot={msg['pot']}")
                for pid, info in msg["hands"].items():
                    arch = bot._OPP.archetype(int(pid))
                    print(f"           Player {pid} ({arch}): "
                          f"{info['cards']} ({info['hand']})")
                print(f"           Stacks: {msg['stacks']}")
                self.history.append(msg)

            elif t == "winner":
                print(f"[{self.name}] Player {msg['pid']} wins "
                      f"pot={msg['pot']} ({msg['reason']})")
                print(f"           Stacks: {msg['stacks']}")
                self.history.append(msg)       # opponent model reads this

            elif t == "game_over":
                print(f"[{self.name}] GAME OVER — winner: player {msg['winner']}")
                break

            else:
                print(f"[{self.name}] MSG: {msg}")

    def _build_response(self, action, state: GameState) -> dict:
        if isinstance(action, tuple):
            verb, amount = action
            amount = max(int(amount), state.min_raise)
            amount = min(amount, state.chips + state.current_bet)
            return {"action": verb, "amount": amount}

        action = action.lower()

        if action == "check":
            if not state.can_check:
                print(f"[{self.name}] WARNING: tried check, must call — folding")
                return {"action": "fold"}
            return {"action": "check"}

        if action == "allin":
            return {"action": "allin",
                    "amount": state.chips + state.current_bet}

        if action in ("fold", "call"):
            return {"action": action}

        print(f"[{self.name}] WARNING: unknown action '{action}' — folding")
        return {"action": "fold"}
