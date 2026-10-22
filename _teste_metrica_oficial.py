from pathlib import Path
import csv
import numpy as np
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE/"cache_offline"

# parentesco
t_map={}
with open(CACHE/"temas_limpos.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): t_map[str(r["tema_id"])]=r["caminho"].split(" > ")
def rel(g,p):
    if g==p: return "exata"
    gp=set(t_map.get(g,[])); pp=set(t_map.get(p,[]))
    return "pai_filho" if (gp&pp or gp.issubset(pp) or pp.issubset(gp)) else "outro"

# ler features_v12.csv
rows={}
with open(BASE/"features_v12.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): rows[r["id"]]=r

# aceito = exata ou pai_filho
feats, y = [], []
for qid,r in rows.items():
    feats.append([float(r["margem"]),float(r["gap"]),float(r["score_gold"]),float(r["logit"])])
    y.append(1 if rel(r["gold"],r["predito"])!="outro" else 0)
F=np.array(feats); y=np.array(y)
names=["margem","gap","score_gold","logit"]

print("== Varredura: precisao na METRICA OFICIAL (exata|pai|filho aceito) ==")
for j,name in enumerate(names):
    s=F[:,j]
    best=None
    for cut in np.percentile(s, np.arange(50,100,2)):
        sel=s>=cut
        if sel.sum()<3: continue
        acc=y[sel].mean()
        if best is None or (acc>=0.95 and sel.sum()>best[1]): best=(float(cut),int(sel.sum()),acc)
    print(f"  {name:12s} melhor>=95%: limiar={best[0]:.3f} | publica={best[1]:3d} ({best[1]/92*100:4.1f}%) | precisao={best[2]*100:5.1f}%" if best else f"  {name:12s} NENHUM limiar alcanca 95%")

# tambem: precisao bruta por faixa de score_gold
sg=F[:,2]
print("\n== Precisao (oficial) por faixa de score_gold ==")
for lo,hi in [(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,1.0)]:
    m=(sg>=lo)&(sg<hi)
    if m.sum():
        print(f"  score {lo:.1f}-{hi:.1f}: n={m.sum():2d} | precisao={y[m].mean()*100:5.1f}%")
