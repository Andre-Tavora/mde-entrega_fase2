# -*- coding: utf-8 -*-
# v12_02_calibra.py — corrigido
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.stats import ks_2samp

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache_offline"
sys.path.insert(0, str(BASE))
import v10_comum as C

q = C.ler_csv(CACHE / "questoes_limpas.csv")
q["id"] = q["id"].map(C.nid)
t = C.carregar_temas()
fol = C.ler_csv(BASE / "folha_verdade.csv")
fol["id"] = fol["id"].map(C.nid)
fora_col = fol["fora_da_taxonomia"].map(C.nid) == "1" if "fora_da_taxonomia" in fol else pd.Series(False, index=fol.index)
idx = {x: i for i, x in enumerate(t["tema_id"])}
ok = (~fora_col) & fol["tema_central_id"].map(C.nid).isin(idx) & fol["id"].isin(set(q["id"]))
fol = fol[ok].reset_index(drop=True)
n = len(fol)
golds = [idx[x] for x in fol["tema_central_id"].map(C.nid)]
rows = [q["id"].tolist().index(x) for x in fol["id"]]

E = np.load(CACHE / "emb_659.npy").astype(np.float32)
Et = np.load(CACHE / "emb_temas_v10.npy").astype(np.float32)

from sklearn.feature_extraction.text import TfidfVectorizer
W_EMB = 0.6
TFIDF_KW = dict(ngram_range=(1, 2), max_features=20000, sublinear_tf=True, strip_accents="unicode", lowercase=True)
q0 = q["texto_limpo"].fillna("").astype(str).tolist()
tf = TfidfVectorizer(**TFIDF_KW)
tf.fit(t["texto"].tolist() + q0)
Stf = (tf.transform(q0) @ tf.transform(t["texto"].tolist()).T).toarray().astype(np.float32)

def zrow(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)

S = W_EMB * zrow(E @ Et.T) + (1.0 - W_EMB) * zrow(Stf)

eq = np.asarray(C.grupos_equivalencia(t))
caminhos = t["caminho"].tolist()

def ordem(s, lim=5):
    out = []
    vi = set()
    for j in np.argsort(-s, kind="stable"):
        gg = int(eq[j])
        if gg in vi:
            continue
        vi.add(gg)
        out.append(int(j))
        if len(out) >= lim:
            break
    return out

def rel_oficial(g, p):
    if int(eq[g]) == int(eq[p]):
        return True
    pa = caminhos[g].split(" > ")
    pb = caminhos[p].split(" > ")
    la, lb = len(pa), len(pb)
    return (lb == la + 1 and pb[:la] == pa) or (la == lb + 1 and pa[:lb] == pb)

def pai_direto(p, c):
    pa = caminhos[p].split(" > ")
    pc = caminhos[c].split(" > ")
    return len(pc) == len(pa) + 1 and pc[:-1] == pa

def top_prev(s):
    o = ordem(s)
    for k in (1, 2):
        if k >= len(o):
            break
        if (s[o[0]] - s[o[k]]) < 0.5 and pai_direto(o[k], o[0]):
            return o[k]
    return o[0]

def margem_m1(s, p):
    eg = int(eq[p])
    others = np.array([s[j] for j in range(len(t)) if int(eq[j]) != eg])
    return float(s[p] - others.max()) if len(others) else 0.0

prevs = [top_prev(S[r]) for r in rows]
m1 = np.array([margem_m1(S[i], p) for i, p in zip(rows, prevs)], dtype=float)
y = np.array([1 if rel_oficial(golds[i], p) else 0 for i, p in enumerate(prevs)])

# drift shift
old = [i for i, x in enumerate(fol["id"]) if int("".join(ch for ch in x if ch.isdigit())) < 360]
new = [i for i, x in enumerate(fol["id"]) if int("".join(ch for ch in x if ch.isdigit())) >= 360]
ks = ks_2samp(m1[old], m1[new]) if len(old) >= 8 and len(new) >= 8 else ("pequeno", len(old), len(new))

# Calibração OOF
rsk = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
oof = np.zeros(n)
for tr, te in rsk.split(m1.reshape(-1, 1), y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(m1[tr].reshape(-1, 1), y[tr])
    oof[te] = lr.predict_proba(m1[te].reshape(-1, 1))[:, 1]

grid = np.arange(0.50, 0.999, 0.005)
best_t, prec_b, cov_b = 0.5, 0, 0
for Tt in grid:
    pub = np.where(oof >= Tt)[0]
    if len(pub) >= 8:
        prec = 100 * y[pub].mean()
        if prec >= 96.0 and prec > prec_b:
            best_t, prec_b, cov_b = Tt, prec, 100 * len(pub) / n

# Regra A (Wilson)
def wilson(pn, erros):
    from math import sqrt
    if pn == 0:
        return 0.0
    z = 1.96
    p = 1 - erros / pn
    d = z * sqrt(p * (1 - p) / pn + z * z / (4 * pn * pn))
    return (p + z * z / (2 * pn) - d) / (1 + z * z / pn)

m1ord = np.argsort(-m1)
A_lim = None
for cut in range(n, 0, -1):
    suby = y[m1ord[:cut]]
    if wilson(cut, 0) >= 0.95 and cut >= 8:
        A_lim = cut
        break
A_n = A_lim if A_lim else 0
A_prec = 100 * y[m1ord[:A_n]].mean() if A_n else 0
A_cov = 100 * A_n / n

print("[V12-02] n=%d shift_ks=%s" % (n, str(ks)[:60]))
if A_lim:
    print("[V12-02] Regra A(Wilson95): lim=%d itens prec=%.2f cov=%.2f" % (A_lim, A_prec, A_cov))
else:
    print("[V12-02] Regra A(Wilson95): impossivel atingir IC>=95%% mesmo com 0 erros (n pequeno)")
print("[V12-02] Regra B(>96%% OOF): T_prob=%.3f prec=%.2f cov=%.2f" % (best_t, prec_b, cov_b))

lr_full = LogisticRegression(max_iter=1000).fit(m1.reshape(-1, 1), y)
json.dump(dict(
    calibrador=dict(coef=float(lr_full.coef_[0][0]), intercept=float(lr_full.intercept_[0])),
    limiar_B=float(best_t),
    regra_A=dict(n=A_n, prec=A_prec, cov=A_cov) if A_lim else dict(n=0, prec=0, cov=0, msg="inviavel"),
    regra_B=dict(prec=prec_b, cov=cov_b),
    shift=str(ks)[:100],
    n=int(n),
    decisao_fora="nao_emitido_motivo=criterio_sem_precisao"),
    open(BASE / "v12_calibracao.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("[V12-02] OK saved v12_calibracao.json")