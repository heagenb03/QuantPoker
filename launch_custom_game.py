import subprocess
import time
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Launch a custom poker game with various bot strategies")
    parser.add_argument("--chips", type=int, default=1000, help="Starting chips per player")
    parser.add_argument("--duration", type=int, default=25, help="How many seconds to let the game run")
    parser.add_argument("--bb", type=int, default=20, help="Big blind amount")
    parser.add_argument("--port", type=int, default=9998, help="Port to run the test server on")
    
    # Types of bots
    parser.add_argument("--baselines", type=int, default=1, help="Number of baseline bots")
    parser.add_argument("--traditionals", type=int, default=1, help="Number of traditional bots")
    parser.add_argument("--nns", type=int, default=1, help="Number of neural net bots")
    parser.add_argument("--complexes", type=int, default=1, help="Number of complex bots")
    parser.add_argument("--maniacs", type=int, default=0, help="Number of maniac bots")
    parser.add_argument("--stations", type=int, default=0, help="Number of calling station bots")
    
    args = parser.parse_args()
    
    total_players = args.baselines + args.traditionals + args.nns + args.complexes + args.maniacs + args.stations
    if total_players < 2:
        print("Need at least 2 players")
        return

    print(f"Starting poker server with {total_players} players on port {args.port}...")
    server = subprocess.Popen([sys.executable, "poker_server.py", "--players", str(total_players), "--chips", str(args.chips), "--bb", str(args.bb), "--port", str(args.port)])
    time.sleep(2) # Wait for server to boot

    print("Connecting bots...")
    bots = []
    
    def add_bots(count, script_name, prefix):
        for i in range(count):
            bots.append(subprocess.Popen([sys.executable, f"bots/{script_name}", "--name", f"{prefix}{i+1}", "--port", str(args.port)]))
            time.sleep(0.5)
            
    add_bots(args.baselines, "bot.py", "Baseline")
    add_bots(args.traditionals, "bot_trad.py", "Trad")
    add_bots(args.nns, "bot_nn.py", "NN")
    add_bots(args.complexes, "bot_complex.py", "Complex")
    add_bots(args.maniacs, "bot_maniac.py", "Maniac")
    add_bots(args.stations, "bot_calling_station.py", "Station")

    try:
        print(f"Letting them play for {args.duration} seconds...")
        time.sleep(args.duration)
        print("Stopping the game now.")
    except KeyboardInterrupt:
        pass

    for b in bots:
        try:
            b.terminate()
        except:
            pass
    try:
        server.terminate()
    except:
        pass

    print("Generating dashboard...")
    subprocess.run([sys.executable, "data.py"])
    print("Done!")

if __name__ == "__main__":
    main()
