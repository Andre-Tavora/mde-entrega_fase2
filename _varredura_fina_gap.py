from pathlib import Path
import csv, json, math, numpy as np
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE/"cache_offline"
E  = np.load(CACHE/"emb_659.npy").astype(np.float32); Et=np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
ids=json.load(open(CACHE/"emb_659_ids.json",encoding="utf-8-sig")); tid=json.load(open(CACHE/"emb_temas_v10_ids.json",encoding="utf-8-sig"))
def zn(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
E,Et=zn(E),zn(Et)

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

# gap por questao na folha
gaps=[]
for i,qid in enumerate(ids):
    if qid not in fol: continue
    s=E[i]@Et.T; top=np.argsort(s)[::-1]
    p1=str(tid[top[0]]); g=float(s[top[0]]-s[top[1]])
    gaps.append((g, rel(fol[qid],p1), qid))

print("== Varredura fina do GAP (metrica oficial) ==")
z=1.96
best=None
for cut in np.arange(0, 45, 1.0):
    sel=[x for x in gaps if x[0]>=cut]
    n=len(sel)
    if n<3: continue
    ac=sum(1 for x in sel if x[1]); p=ac/n
    d=1+z*z/n; c=(p+z*z/(2*n))/d; w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    lo=max(0,c-w); hi=min(1,c+w)
    flag = " <== IC ok" if lo>=0.95 else ""
    if best is None or lo>=0.95 and n>best[2]: best=(cut,n,p,lo)
    if cut%5==0 or lo>=0.94: print(f"  gap>={cut:5.1f} | n={n:3d} | prec={100*p:5.1f}% | IC=[{100*lo:4.1f}%,{100*hi:4.1f}%]{flag}")

if best:
    cut,n,p,lo=best
    print(f"\n>>> MELHOR: gap>={cut:.1f} | publica={n} ({100*n/92:.1f}%) | prec={100*p:.1f}% | IC inferior={100*lo:.1f}%")
else:
    print("\n>>> NENHUM limiar alcanca IC inferior >= 95%. Manter GAP=17.848 (91.3%).")
