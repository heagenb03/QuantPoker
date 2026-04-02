"""
poker/game_state.py — GameState: read-only snapshot given to decide().
"""


class GameState:
    """Every fact about the current decision point."""

    def __init__(self, raw: dict, history: list, my_pid: int):
        self.my_pid        = my_pid
        self.hole_cards    = raw["hole_cards"]      # e.g. ["Ah","Kd"]
        self.community     = raw["community"]        # 0–5 cards
        self.street        = raw["street"]           # preflop|flop|turn|river
        self.chips         = raw["chips"]
        self.pot           = raw["pot"]
        self.to_call       = raw["to_call"]
        self.current_bet   = raw["current_bet"]
        self.min_raise     = raw["min_raise"]
        self.num_players   = raw["num_players"]      # players IN THIS HAND
        self.player_bets   = raw["player_bets"]
        self.player_chips  = raw["player_chips"]
        self.player_folded = raw["player_folded"]
        self.player_allin  = raw["player_allin"]
        self.history       = history

    @property
    def can_check(self):
        return self.to_call == 0

    @property
    def active_opponents(self):
        """Non-folded opponents still in the hand (excluding me)."""
        return sum(1 for pid, folded in self.player_folded.items()
                   if not folded and str(pid) != str(self.my_pid))

    @property
    def pot_odds(self):
        """Fraction of (pot + call) I must invest to call. 0 if free."""
        if self.to_call == 0:
            return 0.0
        return self.to_call / (self.pot + self.to_call)

    def __repr__(self):
        return (f"<GameState {self.street} hole={self.hole_cards} "
                f"board={self.community} pot={self.pot} "
                f"chips={self.chips} to_call={self.to_call}>")
