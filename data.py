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
    
    # 1. Win Counts
    win_query = """
        SELECT pid, COUNT(*) as wins
        FROM showdowns
        WHERE is_winner = True
        GROUP BY pid
        ORDER BY pid
    """
    win_rates = conn.execute(win_query).fetchall()
    
    # 2. Stack Size Over Time (Line Chart Data)
    # We'll track chips after each action/showdown for each player
    stack_query = """
        SELECT hand_id, pid, chips, action_id
        FROM actions
        UNION ALL
        SELECT hand_id, pid, 0 as chips, 999999 as action_id -- Placeholder for showdown
        FROM showdowns
        ORDER BY hand_id, action_id
    """
    # Actually, a better way to get stack per hand is to look at the actions table's 'chips' value
    # but that's 'chips remaining' at the time of action.
    # Let's get the 'chips' value from the last action of each player in each hand.
    
    history_query = """
        SELECT hand_id, pid, chips
        FROM actions
        QUALIFY ROW_NUMBER() OVER (PARTITION BY hand_id, pid ORDER BY action_id DESC) = 1
        ORDER BY hand_id, pid
    """
    stack_history = conn.execute(history_query).fetchall()

    # 3. Hand Statistics Table
    table_query = """
        SELECT h.hand_id, h.timestamp, h.pot, 
               s.pid as winner_pid, s.hand_name, s.gain
        FROM hands h
        LEFT JOIN showdowns s ON h.hand_id = s.hand_id AND s.is_winner = True
        ORDER BY h.hand_id DESC
    """
    table_data = conn.execute(table_query).fetchall()

    # 4. Hand Strength Distribution
    strength_query = """
        SELECT hand_name, COUNT(*) as count 
        FROM showdowns 
        WHERE hand_name IS NOT NULL AND hand_name != 'everyone_folded'
        GROUP BY hand_name 
        ORDER BY count DESC
    """
    strengths = conn.execute(strength_query).fetchall()

    conn.close()
    
    return {
        "win_rates": [{"pid": r[0], "wins": r[1]} for r in win_rates],
        "stack_history": [{"hand": r[0], "pid": r[1], "chips": r[2]} for r in stack_history],
        "table": [{"id": r[0], "time": str(r[1]), "pot": r[2], "winner": r[3], "hand": r[4], "gain": r[5]} for r in table_data],
        "strengths": [{"name": r[0], "count": r[1]} for r in strengths]
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
        h1 {{
            font-size: 24px;
            font-weight: 600;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 30px;
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
        <h1>Poker Performance Analysis</h1>
        
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
        const raw = {json_data};
        
        // Populate Stats
        document.getElementById('stat-hands').innerText = raw.table.length;
        const avgPot = raw.table.length > 0 ? (raw.table.reduce((a, b) => a + b.pot, 0) / raw.table.length).toFixed(0) : 0;
        document.getElementById('stat-avg-pot').innerText = '$' + Number(avgPot).toLocaleString();
        document.getElementById('stat-total-volume').innerText = raw.table.reduce((a, b) => a + b.pot, 0).toLocaleString();

        // 1. Stack Size Over Time (Line Chart)
        const pids = [...new Set(raw.stack_history.map(d => d.pid))].sort();
        const hands = [...new Set(raw.stack_history.map(d => d.hand))].sort((a,b) => a-b);
        
        const stackDatasets = pids.map(pid => {{
            const playerHistory = raw.stack_history.filter(d => d.pid === pid);
            const dataMap = {{}};
            playerHistory.forEach(d => dataMap[d.hand] = d.chips);
            
            // Fill in gaps if any
            let lastVal = 1000; // Starting chips fallback
            const finalData = hands.map(h => {{
                if (dataMap[h] !== undefined) lastVal = dataMap[h];
                return lastVal;
            }});

            return {{
                label: `Player ${{pid}}`,
                data: finalData,
                borderColor: pid === 0 ? '#2c7be5' : (pid === 1 ? '#d39e00' : (pid === 2 ? '#e63757' : '#00d97e')),
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.1
            }};
        }});

        new Chart(document.getElementById('stackChart'), {{
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

        // 2. Win Count (Bar Chart)
        new Chart(document.getElementById('winChart'), {{
            type: 'bar',
            data: {{
                labels: raw.win_rates.map(w => 'P' + w.pid),
                datasets: [{{
                    label: 'Wins',
                    data: raw.win_rates.map(w => w.wins),
                    backgroundColor: '#2c7be5'
                }}]
            }},
            options: {{ scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }}
        }});

        // 3. Hand Strength (Horizontal Bar)
        new Chart(document.getElementById('strengthChart'), {{
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

        // Table
        const tbody = document.getElementById('table-body');
        raw.table.forEach(row => {{
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${{row.id}}</td>
                <td>${{row.time}}</td>
                <td>$${{row.pot.toLocaleString()}}</td>
                <td class="winner-cell">${{row.winner !== null ? 'Player ' + row.winner : 'N/A'}}</td>
                <td>${{row.hand || '-'}}</td>
                <td>${{row.gain ? '+' + row.gain : '0'}}</td>
            `;
            tbody.appendChild(tr);
        }});

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
        print("No data found. Please run your poker server first.")
        return
    
    generate_html(data)
    print(f"Report generated: {{OUTPUT_HTML}}")
    
    path = os.path.abspath(OUTPUT_HTML)
    webbrowser.open(f"file://{{path}}")

if __name__ == "__main__":
    main()
