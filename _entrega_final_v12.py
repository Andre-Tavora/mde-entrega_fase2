from pathlib import Path
import csv, json
import numpy as np

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE/"cache_offline"

E  = np.load(CACHE/"emb_659.npy").astype(np.float32)
Et = np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
ids = json.load(open(CACHE/"emb_659_ids.json",encoding="utf-8-sig"))
tid = json.load(open(CACHE/"emb_temas_v10_ids.json",encoding="utf-8-sig"))
def zn(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
E, Et = zn(E), zn(Et)

# limiar escolhido: GAP top1-top2 (melhor precisao oficial, 91.3%)
GAP_CUT = 17.848
FORA_CUT = 0.10   # score maximo baixo -> fora da taxonomia

# tema_name p/ extras
tname={}
with open(CACHE/"temas_limpos.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): tname[str(r["tema_id"])]=r.get("tema",r["tema_id"])

out=[]
for i,qid in enumerate(ids):
    sims = E[i]@Et.T
    top = np.argsort(sims)[::-1]
    p1, p2 = str(tid[top[0]]), str(tid[top[1]])
    m1, m2 = float(sims[top[0]]), float(sims[top[1]])
    gap = m1-m2
    if gap >= GAP_CUT:   decisao="publicar"
    elif m1 < FORA_CUT:  decisao="fora"
    else:                decisao="revisar"
    extras = ";".join(str(tid[t]) for t in top[1:4])
    out.append(dict(id=qid, decisao=decisao, tema_central_id=p1,
                    extras_ids=extras, confianca=round(gap,4),
                    motivo=f"gap={gap:.4f}"))

# grava
with open(BASE/"publicacao.csv","w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

from collections import Counter
c=Counter(r["decisao"] for r in out)
print("publicacao.csv:", dict(c), "| total=%d" % len(out))

# acerto nas 92 (metrica oficial pai/filho)
fol={}
with open(BASE/"folha_verdade.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("fora_da_taxonomia")!="1" and r.get("tema_central_id"): fol[str(r["id"])]=str(r["tema_central_id"])
t_map={}
with open(CACHE/"temas_limpos.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): t_map[str(r["tema_id"])]=r["caminho"].split(" > ")
def rel(g,p):
    if g==p: return True
    gp=set(t_map.get(g,[])); pp=set(t_map.get(p,[]))
    return bool(gp&pp or gp.issubset(pp) or pp.issubset(gp))

pub=[r for r in out if r["decisao"]=="publicar" and r["id"] in fol]
ac=sum(1 for r in pub if rel(fol[r["id"]], r["tema_central_id"]))
print("publicadas na folha: %d/%d | acertos=%d | precisao=%.1f%% | cobertura=%.1f%%" % (len(pub),len(fol),ac, 100*ac/max(len(pub),1), 100*len(pub)/len(fol)))
