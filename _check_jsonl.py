from pathlib import Path
import json
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")

for p in sorted(BASE.glob("*.jsonl")) + sorted((BASE/"cache_offline").glob("*.jsonl")):
    print("JSONL:", p)
    with open(p, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    print("  linhas:", len(lines))
    if lines:
        first = json.loads(lines[0])
        print("  chaves:", list(first.keys()))
        print("  id:", str(first.get("id", "")), "| texto_60:", str(first.get("texto_limpo", first.get("texto", "")))[:60])
