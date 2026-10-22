from pathlib import Path
import csv, json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE / "cache_offline"

# dados
E  = np.load(CACHE/"emb_659.npy").astype(np.float32)
Et = np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
ids = json.load(open(CACHE/"emb_659_ids.json",encoding="utf-8-sig"))
tid = json.load(open(CACHE/"emb_temas_v10_ids.json",encoding="utf-8-sig"))

# normaliza
def zn(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
E, Et = zn(E), zn(Et)

# parentesco: mapa tema -> set[pai,filhos] (a partir do caminho)
t_map = {}
with open(CACHE/"temas_limpos.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        t_map[str(r["tema_id"])] = r["caminho"]
def rel(g,p):
    if g==p: return "exata"
    gp = set(t_map.get(g,"").split(" > ")); pp = set(t_map.get(p,"").split(" > "))
    if gp.issubset(pp) or pp.issubset(gp) or gp & pp: return "pai_filho"
    return "outro"

# gold
gold = {}
with open(BASE/"folha_verdade.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("fora_da_taxonomia")!="1" and r.get("tema_central_id"):
            gold[str(r["id"])] = str(r["tema_central_id"])

# features por questao
rows = []
X, y, meta = [], [], []
for i, qid in enumerate(ids):
    if qid not in gold: continue
    sims = E[i] @ Et.T
    top = np.argsort(sims)[::-1]
    p1, p2 = str(tid[top[0]]), str(tid[top[1]])
    m1, m2 = float(sims[top[0]]), float(sims[top[1]])
    g = gold[qid]
    sg = float(sims[tid.index(g)]) if g in tid else np.nan
    r = rel(g,p1)
    X.append([m1, m1-m2, sg, sg-m2])
    y.append(1 if r!="outro" else 0)   # aceito = exata ou pai/filho
    meta.append((qid, g, p1, m1, m1-m2, sg, r))

X, y = np.array(X, dtype=float), np.array(y, int)
names = ["margem_top1","gap_top1_top2","score_gold","gap_gold_top2"]

print("n=%d | aceitos=%d (%d%%)" % (len(y), y.sum(), 100*y.mean()))
print("\n=== AUC por feature (quanto >0.5 melhor discrimina) ===")
for j,name in enumerate(names):
    mask = ~np.isnan(X[:,j])
    if mask.sum()>10:
        auc = roc_auc_score(y[mask], X[mask,j])
        print(f"  {name:20s} AUC={auc:.3f}")

print("\n=== AUC do logit com TODAS as features (OOF, LOO) ===")
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut(); preds = np.zeros(len(y))
for tr, te in loo.split(X):
    mis = ~np.isnan(X[te][0]); clf = LogisticRegression(max_iter=2000)
    clf.fit(X[tr][:,mis], y[tr]); preds[te] = clf.predict_proba(X[te][:,mis])[:,1]
print(f"  Logit LOO AUC={roc_auc_score(y,preds):.3f}")

# varredura de limiar sobre o logit -> precisao x cobertura
print("\n=== Curva precisao/cobertura (gold aceito) ===")
ord = np.argsort(preds)[::-1]
for cut in np.percentile(preds,[75,85,90,93,95,97,98]):
    sel = preds>=cut
    acc = y[sel].mean() if sel.sum() else 0
    cov = sel.mean()
    print(f"  limiar={cut:.3f} | publica={sel.sum():3d} ({cov*100:4.1f}%) | precisao={acc*100:5.1f}%")

# salvar features
with open(BASE/"features_v12.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","gold","predito","margem","gap","score_gold","rel","logit"])
    w.writeheader()
    for (qid,g,p1,m1,gap,sg,r),pr in zip(meta,preds):
        w.writerow(dict(id=qid,gold=g,predito=p1,margem=round(m1,4),gap=round(gap,4),score_gold=round(sg,4),rel=r,logit=round(float(pr),4)))
print("\nOK features_v12.csv salvo")
