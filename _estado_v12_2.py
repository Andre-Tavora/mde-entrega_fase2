from pathlib import Path
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")

print("=== log_regressao.txt (ultimas 40 linhas) ===")
try:
    txt = (BASE/"log_regressao.txt").read_text(encoding="utf-8")
    print("\n".join(txt.splitlines()[-40:]))
except Exception as e: print("erro:", e)

print("\n=== publicacao.csv: distribuicao de decisao ===")
import csv
from collections import Counter
with open(BASE/"publicacao.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
c = Counter(r.get("decisao","") for r in rows)
print("total:", len(rows), "|", dict(c))
print("primeiras 3 linhas:")
for r in rows[:3]: print(r)

print("\n=== previsao.txt (se existir) ===")
p = BASE/"previsao.txt"
if p.is_file():
    print(p.read_text(encoding="utf-8")[:1500])
else:
    print("(nao existe previsao.txt na raiz)")
