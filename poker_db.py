import duckdb
import json
import datetime
import os
import threading

class PokerDB:
    def __init__(self, db_path="poker_game.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._create_tables()

    def _get_conn(self):
        return duckdb.connect(self.db_path)

    def _create_tables(self):
        with self.lock:
            conn = self._get_conn()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hands (
                    hand_id INTEGER PRIMARY KEY,
                    game_id VARCHAR,
                    timestamp TIMESTAMP,
                    dealer_pid INTEGER,
                    sb_pid INTEGER,
                    bb_pid INTEGER,
                    pot INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    action_id INTEGER PRIMARY KEY,
                    hand_id INTEGER,
                    street VARCHAR,
                    pid INTEGER,
                    action VARCHAR,
                    amount INTEGER,
                    chips INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS community_cards (
                    hand_id INTEGER,
                    street VARCHAR,
                    cards VARCHAR
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS showdowns (
                    hand_id INTEGER,
                    pid INTEGER,
                    cards VARCHAR,
                    hand_name VARCHAR,
                    score INTEGER,
                    is_winner BOOLEAN,
                    gain INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    pid INTEGER PRIMARY KEY,
                    name VARCHAR
                )
            """)
            
            conn.close()

    def log_hand_start(self, game_id, dealer, sb, bb, pot):
        with self.lock:
            conn = self._get_conn()
            hand_id = conn.execute("SELECT COALESCE(MAX(hand_id), 0) + 1 FROM hands").fetchone()[0]
            conn.execute("""
                INSERT INTO hands (hand_id, game_id, timestamp, dealer_pid, sb_pid, bb_pid, pot)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (hand_id, game_id, datetime.datetime.now(), dealer, sb, bb, pot))
            conn.close()
            return hand_id

    def log_action(self, hand_id, street, pid, action, amount, chips):
        if hand_id is None: return
        with self.lock:
            conn = self._get_conn()
            action_id = conn.execute("SELECT COALESCE(MAX(action_id), 0) + 1 FROM actions").fetchone()[0]
            conn.execute("""
                INSERT INTO actions (action_id, hand_id, street, pid, action, amount, chips)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (action_id, hand_id, street, pid, action, int(amount), int(chips)))
            conn.close()

    def log_community(self, hand_id, street, cards):
        if hand_id is None: return
        with self.lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO community_cards (hand_id, street, cards)
                VALUES (?, ?, ?)
            """, (hand_id, street, json.dumps(cards)))
            conn.close()

    def log_showdown(self, hand_id, pid, cards, hand_name, score, is_winner, gain):
        if hand_id is None: return
        with self.lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT INTO showdowns (hand_id, pid, cards, hand_name, score, is_winner, gain)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (hand_id, pid, json.dumps(cards), hand_name, score, is_winner, int(gain)))
            conn.close()

    def update_hand_pot(self, hand_id, pot):
        if hand_id is None: return
        with self.lock:
            conn = self._get_conn()
            conn.execute("UPDATE hands SET pot = ? WHERE hand_id = ?", (int(pot), hand_id))
            conn.close()

    def add_player(self, pid, name):
        with self.lock:
            conn = self._get_conn()
            res = conn.execute("SELECT 1 FROM players WHERE pid = ?", (pid,)).fetchone()
            if not res:
                conn.execute("INSERT INTO players (pid, name) VALUES (?, ?)", (pid, name))
            else:
                conn.execute("UPDATE players SET name = ? WHERE pid = ?", (name, pid))
            conn.close()
