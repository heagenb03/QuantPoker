import re

with open('bots/bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# find where decide is
match = re.search(r'(def decide\(state: GameState\):.*?)(# ───+[\n\r]+# BOT CLIENT)', content, re.DOTALL)
if match:
    prefix = content[:match.start(1)]
    suffix = match.group(2) + content[match.end(2):]
else:
    print('Failed to find decide block')
    exit(1)

trad_logic = '''def decide(state: GameState):
    ranks = "23456789TJQKA"
    try:
        val1 = ranks.index(state.hole_cards[0][0])
        val2 = ranks.index(state.hole_cards[1][0])
        hand_val = val1 + val2
        if val1 == val2: hand_val += 10
    except:
        hand_val = 10
        
    strength = hand_val / 34.0

    if state.to_call == 0:
        if strength > 0.6:
            return ("raise", min(state.min_raise, state.chips + state.current_bet))
        return "check"

    if state.pot_odds > 0:
        if strength > state.pot_odds * 1.5:
            return "call"
        
    return "fold"

'''

nn_logic = '''def decide(state: GameState):
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

'''

comp_logic = '''def decide(state: GameState):
    import random
    ranks = "23456789TJQKA"
    try:
        val1 = ranks.index(state.hole_cards[0][0])
        val2 = ranks.index(state.hole_cards[1][0])
        is_suited = state.hole_cards[0][1] == state.hole_cards[1][1]
        hand_val = val1 + val2 + (4 if is_suited else 0) + (10 if val1 == val2 else 0)
    except:
        hand_val = 10
        
    if state.street == "preflop":
        if hand_val > 18:
            return ("raise", min(state.min_raise + state.pot, state.chips + state.current_bet))
        elif hand_val > 12:
            return "call" if not state.can_check else "check"
        else:
            return "fold" if not state.can_check else "check"
    else:
        if state.to_call > state.pot * 0.5:
            if random.random() < 0.2:
                return "call"
            return "fold"
        
        if state.can_check:
            if random.random() < 0.3:
                return ("raise", min(state.min_raise, state.chips + state.current_bet))
            return "check"
        return "call"

'''

maniac_logic = '''def decide(state: GameState):
    if state.can_check and state.to_call == 0:
        import random
        if random.random() < 0.5:
            return ("raise", min(state.min_raise + state.pot, state.chips + state.current_bet))
    return ("raise", min(state.min_raise + state.pot, state.chips + state.current_bet))

'''

station_logic = '''def decide(state: GameState):
    if state.to_call == 0:
        return "check"
    return "call"

'''

with open('bots/bot_trad.py', 'w', encoding='utf-8') as f:
    f.write(prefix + trad_logic + suffix)

with open('bots/bot_nn.py', 'w', encoding='utf-8') as f:
    f.write(prefix + nn_logic + suffix)

with open('bots/bot_complex.py', 'w', encoding='utf-8') as f:
    f.write(prefix + comp_logic + suffix)

with open('bots/bot_maniac.py', 'w', encoding='utf-8') as f:
    f.write(prefix + maniac_logic + suffix)

with open('bots/bot_calling_station.py', 'w', encoding='utf-8') as f:
    f.write(prefix + station_logic + suffix)

print('Bots created!')
