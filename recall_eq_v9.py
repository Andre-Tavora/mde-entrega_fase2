import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "saida_pipeline_v4"))
from inferir_v9 import _carregar

SEP = " > "
KS = (1, 3, 5)
CRIT = ["A", "B", "C", "D"]
NOMES = {"A": "A estrito", "B": "B equivalencia", "C": "C B+ancestrais", "D": "D C+descendentes"}


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def ncam(s):
    return SEP.join([x.strip() for x in str(s).split(">") if x.strip()])


def abortar(msg):
    print("RECALL_EQ FALHOU: " + msg)
    sys.exit(1)


def ler(p):
    if not p.exists():
        abortar("arquivo ausente: " + str(p))
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def exigir(df, cols, nome):
    for c in cols:
        if c not in df.columns:
            abortar("coluna ausente em %s: %s" % (nome, c))


# ---------- 1. hierarquia ----------
P = _carregar()
parent = np.asarray(P["parent"]).astype(int)
eq = np.asarray(P["eq"])
tema_ids = [nid(x) for x in P["tema_ids"]]
caminhos = [ncam(x) for x in P["caminhos"]]
m = len(tema_ids)
id2idx = {}
for j, t in enumerate(tema_ids):
    id2idx.setdefault(t, j)
cam2idx = {}
for j, c in enumerate(caminhos):
    cam2idx.setdefault(c, j)

anc = []
for j in range(m):
    s, cur, guard = set(), int(parent[j]), 0
    while cur != -1 and guard < 100:
        s.add(cur)
        cur = int(parent[cur])
        guard += 1
    anc.append(s)
desc = [set() for _ in range(m)]
for j in range(m):
    for a in anc[j]:
        desc[a].add(j)

grupos = {}
for j in range(m):
    grupos.setdefault(int(eq[j]), []).append(j)
print("[HIERARQUIA] nos = %d | grupos de equivalencia > 1: %d" % (m, sum(1 for v in grupos.values() if len(v) > 1)))


def aceitos(g):
    A = {g}
    B = set(grupos[int(eq[g])])
    C = set(B)
    D = set()
    for b in B:
        C |= anc[b]
    D = set(C)
    for b in B:
        D |= desc[b]
    return {"A": A, "B": B, "C": C, "D": D}


# ---------- 2. dados ----------
fol = ler(BASE / "folha_verdade.csv")
exigir(fol, ["id", "tema_central_id", "tema_central_caminho", "fora_da_taxonomia"], "folha_verdade.csv")
fol["id"] = fol["id"].map(nid)
fol = fol.drop_duplicates("id", keep="first")
fol["fora"] = fol["fora_da_taxonomia"].map(nid)
fol["cam"] = fol["tema_central_caminho"].map(ncam)
fol["cid"] = fol["tema_central_id"].map(nid)
fol = fol[(fol["fora"] != "1") & (fol["cam"] != "")].reset_index(drop=True)

err = ler(BASE / "erros_tipo_v9.csv")
exigir(err, ["id", "tipo"], "erros_tipo_v9.csv")
err["id"] = err["id"].map(nid)
err = err.drop_duplicates("id", keep="first")
tipo_map = dict(zip(err["id"], err["tipo"].str.strip().str.upper()))

tk = ler(BASE / "topk_v9.csv")
exigir(tk, ["id", "rank", "tema_id", "caminho"], "topk_v9.csv")
tk["id"] = tk["id"].map(nid)
tk["rank"] = pd.to_numeric(tk["rank"], errors="coerce")
if tk["rank"].isna().any():
    abortar("rank nao numerico em topk_v9.csv")
tk["tema_id"] = tk["tema_id"].map(nid)
tk["cam"] = tk["caminho"].map(ncam)
tk = tk.sort_values(["id", "rank"], kind="stable")
topk = {}
for qid, grp in tk.groupby("id", sort=False):
    nos = []
    for tid, cam in zip(grp["tema_id"], grp["cam"]):
        if tid in id2idx:
            nos.append(id2idx[tid])
        elif cam in cam2idx:
            nos.append(cam2idx[cam])
        else:
            nos.append(-1)
    topk[qid] = nos[:5]

Q = []
sem_no, sem_topk = 0, 0
for _, r in fol.iterrows():
    g = id2idx.get(r["cid"], -1) if r["cid"] else -1
    if g < 0:
        g = cam2idx.get(r["cam"], -1)
    if g < 0:
        sem_no += 1
        continue
    if r["id"] not in topk:
        sem_topk += 1
        continue
    Q.append(dict(id=r["id"], g=g, tipo=tipo_map.get(r["id"], ""), acc=aceitos(g), top=topk[r["id"]]))
n = len(Q)
print("[DADOS] folha valida = %d | sem no no gabarito = %d | sem top-k = %d | avaliadas n = %d (esperado 92)" % (
    len(fol), sem_no, sem_topk, n))
if n == 0:
    abortar("nenhuma questao avaliavel")

# ---------- 3. recall ----------
hits = {c: {k: np.array([any(t in q["acc"][c] for t in q["top"][:k]) for q in Q]) for k in KS} for c in CRIT}
grp_if = np.array([q["tipo"] in ("IRMAO", "FILHO") for q in Q])
n_if = int(grp_if.sum())

print("\n=== RECALL@K POR CRITERIO (top-k de topk_v9.csv contem algum no aceito) ===")
print("%-18s %-14s %5s %9s %9s %9s" % ("criterio", "grupo", "n", "recall@1", "recall@3", "recall@5"))
for c in CRIT:
    for nome, mk in ((("todas (%d)" % n), np.ones(n, dtype=bool)), ("IRMAO+FILHO", grp_if)):
        nn = int(mk.sum())
        if nn == 0:
            print("%-18s %-14s %5d %9s %9s %9s" % (NOMES[c], nome, 0, "-", "-", "-"))
        else:
            print("%-18s %-14s %5d %8.1f%% %8.1f%% %8.1f%%" % (
                NOMES[c], nome, nn, hits[c][1][mk].mean() * 100, hits[c][3][mk].mean() * 100, hits[c][5][mk].mean() * 100))
print("n do grupo IRMAO+FILHO = %d" % n_if)

# ---------- 4. misses de A recuperados ----------
missA = ~hits["A"][5]
nA = int(missA.sum())
print("\n=== 1. MISSES DO CRITERIO A (fora do top-5): %d (esperado 36) ===" % nA)
for c in ("B", "C", "D"):
    rec = int((missA & hits[c][5]).sum())
    print("passam a acertar em %s: %d de %d | misses restantes em %s: %d" % (c, rec, nA, c, int((~hits[c][5]).sum())))

# ---------- 5. misses restantes em B ----------
missB = np.where(~hits["B"][5])[0]
print("\n=== 2. MISSES QUE SOBRAM EM B (fora do top-5 mesmo com equivalencia): %d ===" % len(missB))
for i in missB:
    q = Q[i]
    t1 = q["top"][0] if q["top"] else -1
    cam1 = caminhos[t1] if t1 >= 0 else "(sem top-1)"
    print("id=%s | tipo=%s" % (q["id"], q["tipo"] or "(sem tipo)"))
    print("  gabarito: " + caminhos[q["g"]])
    print("  top-1   : " + cam1)

# ---------- 6. top-1 aceito ----------
t1A = int(hits["A"][1].sum())
t1B = int(hits["B"][1].sum())
print("\n=== 3. TOP-1 BRUTO ACEITO ===")
print("critério A (estrito)     : %d/%d = %.1f%%" % (t1A, n, t1A / n * 100))
print("critério B (equivalencia): %d/%d = %.1f%%" % (t1B, n, t1B / n * 100))
print("acertos extras por equivalencia (B - A): %d" % (t1B - t1A))

print("\nAVISO: n=92; ganhos de poucas questoes sao ruido. Equivalencia (B) e legitima; C/D afrouxam o criterio e servem so como diagnostico.")
print("\nRECALL_EQ OK")