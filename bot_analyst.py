"""
bot_collector.py
Scans the bots/ folder and dumps every bot's full source code into bot_strategies.json.
Plug in Gemini / Flywheel later when APIs are available.
"""

import os
import json
import glob
from datetime import datetime

BOTS_DIR     = "bots"
OUTPUT_FILE  = "bot_strategies.json"

def extract_decide(source: str) -> str:
    """Pull out just the decide() function body."""
    lines   = source.splitlines()
    result  = []
    inside  = False
    base_indent = None

    for line in lines:
        if line.strip().startswith("def decide("):
            inside = True
            base_indent = len(line) - len(line.lstrip())
            result.append(line)
            continue

        if inside:
            if not line.strip():          # blank lines are fine
                result.append(line)
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and result:
                # Hit next top-level definition — stop
                break
            result.append(line)

    return "\n".join(result) if result else ""

def collect():
    bot_files = sorted(glob.glob(os.path.join(BOTS_DIR, "*.py")))
    if not bot_files:
        print(f"No .py files found in {BOTS_DIR}/")
        return

    strategies = []
    for path in bot_files:
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            source = f.read()

        decide_fn = extract_decide(source)

        strategies.append({
            "file":          fname,
            "path":          path,
            "collected_at":  datetime.now().isoformat(),
            "size_bytes":    os.path.getsize(path),
            "full_source":   source,
            "decide_fn":     decide_fn,
            # Placeholders — fill in later with Gemini or manually
            "analysis": {
                "strategy_name": "",
                "category":      "",
                "description":   "",
                "key_heuristics": [],
                "strengths":     [],
                "weaknesses":    [],
            },
            "relationships": {
                "similar_to": [],
                "counters":   [],
                "loses_to":   [],
            }
        })

    output = {
        "generated_at":  datetime.now().isoformat(),
        "total_bots":    len(strategies),
        "strategies":    strategies,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Collected {len(strategies)} bot(s) → {OUTPUT_FILE}")
    for s in strategies:
        lines = s["decide_fn"].count("\n") + 1
        print(f"  {s['file']:<30}  {s['size_bytes']:>5} bytes  |  decide() = {lines} lines")

if __name__ == "__main__":
    collect()
