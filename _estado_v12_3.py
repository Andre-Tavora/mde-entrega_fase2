from pathlib import Path
import json, csv
from collections import Counter
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")

# 1. v12_calibracao.json completo (nao so o resumo)
p = BASE/"v12_calibracao.json"
print("=== v12_calibracao.json (completo) ===")
if p.is_file(): print(p.read_text(encoding="utf-8")[:2000])

# 2. log_regressao.txt ler como latin-1 (utf-8 falhou)
for nome, enc in [("log_regressao.txt","latin-1"), ("log_regressao.txt","cp1252")]:
    p2 = BASE/nome
    if p2.is_file():
        try:
            print(f"\n=== {nome} (enc={enc}, ultimas 40) ===")
            print("\n".join(p2.read_text(encoding=enc).splitlines()[-40:]))
            break
        except Exception as e: print(f"  {enc} falhou: {e}")

# 3. publicacao.csv: precisao real nas 92 da folha
print("\n=== PRECISAO NAS 92 (publicacao vs folha) ===")
with open(BASE/"folha_verdade.csv", encoding="utf-8-sig") as f:
    fol = {r["id"]: r["tema_central_id"] for r in csv.DictReader(f) if r.get("fora_da_taxonomia","")!="1" and r.get("tema_central_id")}
with open(BASE/"publicacao.csv", encoding="utf-8-sig") as f:
    pub = {r["id"]: r for r in csv.DictReader(f)}
ac = er = 0
for qid, gold in fol.items():
    if qid in pub and pub[qid]["decisao"]=="publicar" and gold in (pub[qid]["tema_central_id"] + ";" + pub[qid]["extras_ids"]).split(";"):
        ac += 1
    elif qid in pub and pub[qid]["decisao"]=="publicar":
        er += 1
npub = ac+er
print(f"publicadas na folha={npub} | acertos={ac} | precisao={(ac/npub*100) if npub else 0:.1f}% | cobertura={npub/92*100:.1f}%")

# 4. distribuicao de confianca nas 290 publicadas (pra achar limiar alternativo)
vals = sorted(float(pub[x]["confianca"]) for x in pub if pub[x]["decisao"]=="publicar")
if vals:
    import numpy as np
    print(f"\nconfianca das publicadas: min={vals[0]:.4f} | mediana={np.median(vals):.4f} | max={vals[-1]:.4f}")
    for q in (10,50,90): print(f"  pctil{q}: {np.percentile(vals,q):.4f}")
