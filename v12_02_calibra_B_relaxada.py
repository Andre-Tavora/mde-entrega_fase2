# -*- coding: utf-8 -*-
# v12_02_calibra_B_relaxada.py
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
    out = []; vi = set()
    for j in np.argsort(-s, kind="stable"):
        gg = int(eq[j])
        if gg in vi: continue
        vi.add(gg); out.append(int(j))
        if len(out) >= lim: break
    return out

def rel_oficial(g, p):
    if int(eq[g]) == int(eq[p]): return True
    pa = caminhos[g].split(" > "); pb = caminhos[p].split(" > ")
    la, lb = len(pa), len(pb)
    return (lb == la + 1 and pb[:la] == pa) or (la == lb + 1 and pa[:lb] == pb)

def pai_direto(p, c):
    pa = caminhos[p].split(" > "); pc = caminhos[c].split(" > ")
    return len(pc) == len(pa) + 1 and pc[:-1] == pa

def top_prev(s):
    o = ordem(s)
    for k in (1, 2):
        if k >= len(o): break
        if (s[o[0]] - s[o[k]]) < 0.5 and pai_direto(o[k], o[0]):
            return o[k]
    return o[0]

prevs = [top_prev(S[r]) for r in rows]
m1 = np.array([
    float(S[i][p] - max(S[i][j] for j in range(len(t)) if int(eq[j]) != int(eq[p])))
    if any(int(eq[j]) != int(eq[p]) for j in range(len(t))) else 0.0
    for i, p in zip(rows, prevs)
], dtype=float)
y = np.array([1 if rel_oficial(golds[i], p) else 0 for i, p in enumerate(prevs)])

old = [i for i, x in enumerate(fol["id"]) if int("".join(ch for ch in x if ch.isdigit())) < 360]
new = [i for i, x in enumerate(fol["id"]) if int("".join(ch for ch in x if ch.isdigit())) >= 360]
ks = ks_2samp(m1[old], m1[new]) if len(old) >= 8 and len(new) >= 8 else ("pequeno", len(old), len(new))

rsk = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
oof = np.zeros(n)
for tr, te in rsk.split(m1.reshape(-1, 1), y):
    lr = LogisticRegression(max_iter=1000)
    lr.fit(m1[tr].reshape(-1, 1), y[tr])
    oof[te] = lr.predict_proba(m1[te].reshape(-1, 1))[:, 1]

# varredura RELAXADA: de 0.40 a 0.95, passo 0.01, precisao minima 88%
grid = np.arange(0.40, 0.955, 0.01)
best_t, prec_b, cov_b = 0.5, 0, 0
for Tt in grid:
    pub = np.where(oof >= Tt)[0]
    if len(pub) >= 5:
        prec = 100 * y[pub].mean()
        if prec >= 88.0 and prec > prec_b:
            best_t, prec_b, cov_b = Tt, prec, 100 * len(pub) / n

print("[V12-02B] n=%d shift_ks=%s" % (n, str(ks)[:60]))
if best_t > 0.5 or prec_b > 0:
    print("[V12-02B] Regra B(>88%% OOF, relaxada): T_prob=%.3f prec=%.2f cov=%.2f" % (best_t, prec_b, cov_b))
else:
    print("[V12-02B] Regra B: nenhum limiar atingiu 88%% de precisao OOF")

# Mesma Regra A
ps = [i for i, x in enumerate(y) if x == 1]
print("[V12-02B] Golds: %d/%d (%.1f%%)" % (len(ps), n, 100 * len(ps) / n))

lr_full = LogisticRegression(max_iter=1000).fit(m1.reshape(-1, 1), y)
json.dump(dict(
    calibrador=dict(coef=float(lr_full.coef_[0][0]), intercept=float(lr_full.intercept_[0])),
    limiar_B=float(best_t),
    regra_B=dict(prec=prec_b, cov=cov_b, n_pub=0),
    shift=str(ks)[:100], n=int(n)),
    open(BASE / "v12_calibracao.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("[V12-02B] OK saved v12_calibracao.json")