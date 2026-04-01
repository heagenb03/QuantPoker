import duckdb
import json
import os
import webbrowser
from datetime import datetime

DB_PATH = "poker_game.db"
OUTPUT_HTML = "dashboard.html"

def get_data():
    if not os.path.exists(DB_PATH):
        return None
    
    conn = duckdb.connect(DB_PATH)
    
    try:
        games_query = "SELECT DISTINCT game_id FROM hands WHERE game_id IS NOT NULL ORDER BY game_id DESC"
        games = [r[0] for r in conn.execute(games_query).fetchall()]
    except Exception:
        # Before game_id was added
        games = []

    if not games:
        conn.close()
        return None

    # 1. Win Counts
    win_query = """
        SELECT h.game_id, s.pid, COUNT(*) as wins
        FROM showdowns s
        JOIN hands h ON s.hand_id = h.hand_id
        WHERE s.is_winner = True AND h.game_id IS NOT NULL
        GROUP BY h.game_id, s.pid
    """
    win_rates_raw = conn.execute(win_query).fetchall()
    
    # 2. Stack Size
    history_query = """
        SELECT h.game_id, a.hand_id, a.pid, a.chips
        FROM actions a
        JOIN hands h ON a.hand_id = h.hand_id
        WHERE h.game_id IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY a.hand_id, a.pid ORDER BY a.action_id DESC) = 1
        ORDER BY h.game_id, a.hand_id, a.pid
    """
    stack_history_raw = conn.execute(history_query).fetchall()

    # 3. Hand Statistics Table
    table_query = """
        SELECT h.game_id, h.hand_id, h.timestamp, h.pot, 
               s.pid as winner_pid, s.hand_name, s.gain
        FROM hands h
        LEFT JOIN showdowns s ON h.hand_id = s.hand_id AND s.is_winner = True
        WHERE h.game_id IS NOT NULL
        ORDER BY h.game_id, h.hand_id DESC
    """
    table_data_raw = conn.execute(table_query).fetchall()

    # 4. Hand Strength Distribution
    strength_query = """
        SELECT h.game_id, s.hand_name, COUNT(*) as count 
        FROM showdowns s
        JOIN hands h ON s.hand_id = h.hand_id
        WHERE s.hand_name IS NOT NULL AND s.hand_name != 'everyone_folded' AND h.game_id IS NOT NULL
        GROUP BY h.game_id, s.hand_name
    """
    strengths_raw = conn.execute(strength_query).fetchall()

    try:
        names_data = conn.execute("SELECT pid, name FROM players").fetchall()
        player_names = {r[0]: r[1] for r in names_data}
    except Exception:
        player_names = {}

    conn.close()

    data_by_game = {}
    for g in games:
        data_by_game[g] = {
            "win_rates": [],
            "stack_history": [],
            "table": [],
            "strengths": []
        }

    for r in win_rates_raw:
        if r[0] in data_by_game: data_by_game[r[0]]["win_rates"].append({"pid": r[1], "wins": r[2]})
    for r in stack_history_raw:
        if r[0] in data_by_game: data_by_game[r[0]]["stack_history"].append({"hand": r[1], "pid": r[2], "chips": r[3]})
    for r in table_data_raw:
        if r[0] in data_by_game: data_by_game[r[0]]["table"].append({"id": r[1], "time": str(r[2]), "pot": r[3], "winner": r[4], "hand": r[5], "gain": r[6]})
    for r in strengths_raw:
        if r[0] in data_by_game: data_by_game[r[0]]["strengths"].append({"name": r[1], "count": r[2]})
    
    return {
        "games": games,
        "player_names": player_names,
        "data_by_game": data_by_game
    }

def generate_html(data):
    json_data = json.dumps(data)
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Poker Analytics Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #ffffff;
            color: #333;
            line-height: 1.5;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin: 0;
        }}
        select {{
            padding: 8px 16px;
            font-size: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        h2 {{
            font-size: 18px;
            font-weight: 500;
            margin-top: 40px;
            margin-bottom: 20px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            position: relative;
            margin-bottom: 40px;
            background: #fff;
            padding: 20px;
            border: 1px solid #efefef;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}
        th, td {{
            text-align: left;
            padding: 12px 8px;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f9f9f9;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #fcfcfc;
        }}
        .winner-cell {{
            font-weight: 600;
            color: #2c7be5;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .summary-card {{
            border: 1px solid #eee;
            padding: 20px;
            text-align: center;
        }}
        .summary-card .label {{
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Poker Performance Analysis</h1>
            <select id="gameSelect" onchange="renderGame(this.value)"></select>
        </div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Total Hands</div>
                <div class="value" id="stat-hands">0</div>
            </div>
            <div class="summary-card">
                <div class="label">Average Pot Size</div>
                <div class="value" id="stat-avg-pot">$0</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Hands Witnessed</div>
                <div class="value" id="stat-total-volume">0</div>
            </div>
        </div>

        <h2>Stack Size Progression</h2>
        <div class="chart-container">
            <canvas id="stackChart"></canvas>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px;">
            <div>
                <h2>Win Count by Player</h2>
                <div class="chart-container">
                    <canvas id="winChart"></canvas>
                </div>
            </div>
            <div>
                <h2>Hand Strength Frequency</h2>
                <div class="chart-container">
                    <canvas id="strengthChart"></canvas>
                </div>
            </div>
        </div>

        <h2>Detailed Game Logs</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Timestamp</th>
                    <th>Final Pot</th>
                    <th>Winner</th>
                    <th>Hand Description</th>
                    <th>Tokens Won</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
    </div>

    <script>
        const payload = {json_data};
        const getName = (pid) => payload.player_names[pid] || ('Player ' + pid);
        
        let charts = {{}};

        const gSelect = document.getElementById('gameSelect');
        payload.games.forEach(g => {{
            const opt = document.createElement('option');
            opt.value = g;
            opt.text = g;
            gSelect.appendChild(opt);
        }});

        function renderGame(gameId) {{
            const raw = payload.data_by_game[gameId];
            if(!raw) return;

            document.getElementById('stat-hands').innerText = raw.table.length;
            const avgPot = raw.table.length > 0 ? (raw.table.reduce((a, b) => a + b.pot, 0) / raw.table.length).toFixed(0) : 0;
            document.getElementById('stat-avg-pot').innerText = '$' + Number(avgPot).toLocaleString();
            document.getElementById('stat-total-volume').innerText = raw.table.reduce((a, b) => a + b.pot, 0).toLocaleString();

            const pids = [...new Set(raw.stack_history.map(d => d.pid))].sort();
            const hands = [...new Set(raw.stack_history.map(d => d.hand))].sort((a,b) => a-b);
            
            const stackDatasets = pids.map(pid => {{
                const playerHistory = raw.stack_history.filter(d => d.pid === pid);
                const dataMap = {{}};
                playerHistory.forEach(d => dataMap[d.hand] = d.chips);
                
                let lastVal = 1000; 
                const finalData = hands.map(h => {{
                    if (dataMap[h] !== undefined) lastVal = dataMap[h];
                    return lastVal;
                }});

                return {{
                    label: getName(pid),
                    data: finalData,
                    borderColor: pid === 0 ? '#2c7be5' : (pid === 1 ? '#d39e00' : (pid === 2 ? '#e63757' : (pid === 3 ? '#00d97e' : '#6e84a3'))),
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }};
            }});

            if(charts.stack) charts.stack.destroy();
            charts.stack = new Chart(document.getElementById('stackChart'), {{
                type: 'line',
                data: {{ labels: hands, datasets: stackDatasets }},
                options: {{
                    responsive: true,
                    interaction: {{ intersect: false, mode: 'index' }},
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Hand Number' }} }},
                        y: {{ title: {{ display: true, text: 'Stack Size' }}, beginAtZero: false }}
                    }},
                    plugins: {{ legend: {{ position: 'bottom' }} }}
                }}
            }});

            if(charts.win) charts.win.destroy();
            charts.win = new Chart(document.getElementById('winChart'), {{
                type: 'bar',
                data: {{
                    labels: raw.win_rates.map(w => getName(w.pid)),
                    datasets: [{{
                        label: 'Wins',
                        data: raw.win_rates.map(w => w.wins),
                        backgroundColor: '#2c7be5'
                    }}]
                }},
                options: {{ scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }}
            }});

            if(charts.strength) charts.strength.destroy();
            charts.strength = new Chart(document.getElementById('strengthChart'), {{
                type: 'bar',
                data: {{
                    labels: raw.strengths.map(s => s.name),
                    datasets: [{{
                        label: 'Count',
                        data: raw.strengths.map(s => s.count),
                        backgroundColor: '#6e84a3'
                    }}]
                }},
                options: {{ indexAxis: 'y', plugins: {{ legend: {{ display: false }} }} }}
            }});

            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';
            raw.table.forEach(row => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${{row.id}}</td>
                    <td>${{row.time}}</td>
                    <td>$${{row.pot.toLocaleString()}}</td>
                    <td class="winner-cell">${{row.winner !== null ? getName(row.winner) : 'N/A'}}</td>
                    <td>${{row.hand || '-'}}</td>
                    <td>${{row.gain ? '+' + row.gain : '0'}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        if (payload.games.length > 0) {{
            renderGame(payload.games[0]);
        }}
    </script>
</body>
</html>
    """
    
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_template)

def main():
    print(f"Generating Data Science Analytics Report from {{DB_PATH}}...")
    data = get_data()
    if not data:
        print("No valid game data found. Please run your poker server first.")
        return
    
    generate_html(data)
    print(f"Report generated: {{OUTPUT_HTML}}")
    
    path = os.path.abspath(OUTPUT_HTML)
    webbrowser.open(f"file://{{path}}")

if __name__ == "__main__":
    main()
