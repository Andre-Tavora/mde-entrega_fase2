from pathlib import Path
import csv, hashlib
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
for nome in ["publicacao.csv","previsao.txt","LEIA-ME.md"]:
    p=BASE/nome
    if p.is_file():
        b=p.read_bytes()
        print(f"{nome:20s} {len(b):8d} bytes  sha256={hashlib.sha256(b).hexdigest()[:16]}")
    else:
        print(f"{nome:20s} AUSENTE")
with open(BASE/"publicacao.csv",encoding="utf-8-sig") as f:
    rows=list(csv.DictReader(f)); cols=list(rows[0].keys())
print("publicacao.csv: %d linhas, colunas=%s" % (len(rows), cols))
print("amostra:"); [print("  ",r) for r in rows[:2]]
