import json
from pathlib import Path

def analyze_scores() -> None:
    log_file = Path("d:/Tradeidesa/grclaudescript/logs/sovereign.jsonl")
    if not log_file.exists():
        print("Log file not found.")
        return

    with open(log_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("event") == "portfolio_built":
                    size = data.get("size", 0)
                    mean_prob = data.get("mean_prob_win", 0)
                    ts = data.get("ts")
                    tickers = data.get("tickers", [])

                    if size > 0 and mean_prob >= 0.55:
                        print(f"Time: {ts} | Mean Prob: {mean_prob:.4f} | Size: {size} | Tickers: {tickers}")
            except Exception:
                continue

if __name__ == "__main__":
    analyze_scores()
