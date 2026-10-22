import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys, re, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "saida_pipeline_v4"))
from classificar_final import classificar, _carregar, _codificar

META = dict(n=92, exato=41, pub=29, aceito=61, aceito_pub=25)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def to_int(v):
    s = nid(v)
    return int(s) if s.isdigit() else -1


# ---------- dados ----------
q = pd.read_csv(BASE / "cache_offline" / "questoes_limpas.csv", encoding="utf-8", dtype=str, keep_default_na=False)
g = pd.read_csv(BASE / "folha_verdade.csv", encoding="utf-8", dtype=str, keep_default_na=False)
q["id"] = q["id"].map(nid)
g["id"] = g["id"].map(nid)
g = g[g["fora_da_taxonomia"].map(nid) != "1"]
df = g.merge(q[["id", "texto_limpo"]], on="id", how="inner").reset_index(drop=True)
df["central"] = df["tema_central_id"].map(to_int)
df["extras"] = df["extras_ids"].map(
    lambda v: {int(x) for x in re.split(r"[;,|\s]+", str(v)) if x.isdigit()}
)
n = len(df)
ids = df["id"].tolist()
textos = df["texto_limpo"].fillna("").astype(str).tolist()
print(f"[DADOS] n avaliadas = {n} | central vazio/invalido = {(df['central'] < 0).sum()}")

# ---------- embeddings (cache) ----------
p_npy = BASE / "cache_offline" / "emb_folha.npy"
p_ids = BASE / "cache_offline" / "emb_folha_ids.json"
P = _carregar()
E = None
if p_npy.exists() and p_ids.exists():
    with open(p_ids, encoding="utf-8") as f:
        if json.load(f) == ids:
            E = np.load(p_npy)
            print(f"[CACHE] embeddings reutilizados {E.shape}")
if E is None:
    E = np.asarray(_codificar(textos, P))  # mesmo encoder local usado por classificar()
    np.save(p_npy, E)
    with open(p_ids, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"[CACHE] embeddings gerados e salvos {E.shape}")

# ---------- predicao ----------
pred = classificar(textos, embeddings=E, ids=ids).reset_index(drop=True)
pred["tema_id"] = pred["tema_id"].map(to_int)
tema_ids = [int(x) for x in P["tema_ids"]]
idx = {t: i for i, t in enumerate(tema_ids)}
parent, depth, eq = P["parent"], P["depth"], P["eq"]
m = len(tema_ids)


def ancestors(i):
    out, cur, guard = [], int(parent[i]), 0
    while cur != -1 and guard < 100:
        out.append(cur)
        cur = int(parent[cur])
        guard += 1
    return out


ANC = [set(ancestors(i)) for i in range(m)]

# matriz Down2 (replica do V9): ancestral com depth>=2, via classes de equivalencia
AncM2 = np.zeros((m, m), dtype=np.int64)
for b in range(m):
    for a in ANC[b]:
        if depth[a] >= 2:
            AncM2[a, b] = 1
Cm = (eq[:, None] == eq[None, :]).astype(np.int64)
Down2 = (Cm @ AncM2 @ Cm) > 0

# ---------- definicoes de aceito ----------
rows = {k: [] for k in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]}
for i in range(n):
    p_t, c_t = int(pred.loc[i, "tema_id"]), int(df.loc[i, "central"])
    p, c = idx.get(p_t, -1), idx.get(c_t, -1)
    ex = df.loc[i, "extras"]
    a1 = p_t == c_t
    a2 = a1 or (p_t in ex)
    anc_ok = p >= 0 and c >= 0 and p in ANC[c] and depth[p] >= 2
    a3 = a2 or anc_ok
    a4 = a3 or (p >= 0 and c >= 0 and c in ANC[p])
    a5 = a2 or (p >= 0 and c >= 0 and eq[p] == eq[c])
    a6 = a3 or a5
    a7 = a1 or (p >= 0 and c >= 0 and (eq[p] == eq[c] or Down2[c, p] or Down2[p, c]))
    for k, v in zip(rows, [a1, a2, a3, a4, a5, a6, a7]):
        rows[k].append(bool(v))

pub = (pred["status"] == "PUBLICAR").values
exato = np.array(rows["A1"])

# ---------- saida ----------
out = pred.copy()
out["tema_central_id"] = df["central"]
out["extras_ids"] = df["extras"].map(lambda s: ";".join(map(str, sorted(s))))
for k, v in rows.items():
    out[f"aceito_{k}"] = v
out.to_csv(BASE / "saida_regressao_v9.csv", index=False, encoding="utf-8-sig")

# ---------- metricas ----------
npub = int(pub.sum())
print("\n=== METRICAS ===")
print(f"exato      : {exato.sum()}/{n} = {exato.mean()*100:.2f}%  (meta 41/92 = 44.57%)")
print(f"publicadas : {npub}/{n} = {npub/n*100:.2f}%  (meta 29 = 31.52%)")

tab = []
for k, v in rows.items():
    v = np.array(v)
    tab.append(dict(
        definicao=k,
        aceito_92=int(v.sum()),
        aceito_92_pct=round(v.mean() * 100, 2),
        aceito_publ=int((v & pub).sum()),
        prec_publ_pct=round((v & pub).sum() / npub * 100, 2) if npub else float("nan"),
    ))
tab = pd.DataFrame(tab)
print("\n(meta: aceito_92=61 (66.30%), aceito_publ=25/29 (86.21%))")
print(tab.to_string(index=False))

# ---------- veredito ----------
div = []
if n != META["n"]:
    div.append(f"n={n} (meta {META['n']})")
if int(exato.sum()) != META["exato"]:
    div.append(f"exato={int(exato.sum())} (meta {META['exato']})")
if npub != META["pub"]:
    div.append(f"publicadas={npub} (meta {META['pub']})")
ok_defs = tab[(tab["aceito_92"] == META["aceito"]) & (tab["aceito_publ"] == META["aceito_pub"])]["definicao"].tolist()
if not ok_defs:
    div.append("nenhuma definicao de aceito reproduziu 61/92 e 25/29")

print()
if div:
    print("REGRESSAO FALHOU")
    for d in div:
        print("  divergiu:", d)
else:
    print("REGRESSAO OK")
    print("  definicao(oes) que reproduziram as metas:", ", ".join(ok_defs))
print(f"\nSaida: {BASE / 'saida_regressao_v9.csv'}")
