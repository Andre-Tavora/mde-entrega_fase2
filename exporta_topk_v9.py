import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "saida_pipeline_v4"))
from inferir_v9 import _carregar, _zrow

SEP = " > "
TAUS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 999]
P_Q = BASE / "cache_offline" / "questoes_limpas.csv"
P_NPY = BASE / "cache_offline" / "emb_659.npy"
P_IDS = BASE / "cache_offline" / "emb_659_ids.json"
P_REF = BASE / "saida_classificacao_v9.csv"
P_FOL = BASE / "folha_verdade.csv"
P_ERR = BASE / "erros_tipo_v9.csv"
O_TOPK = BASE / "topk_v9.csv"
O_CAD = BASE / "cadeia_pais_v9.csv"
O_SWEEP = BASE / "sweep_tau_v9.csv"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def norm_cam(s):
    return SEP.join([x.strip() for x in str(s).split(">") if x.strip()])


def abortar(msg):
    print("TOPK FALHOU: " + msg)
    sys.exit(1)


def ler(p, enc):
    if not p.exists():
        abortar("arquivo ausente: " + str(p))
    df = pd.read_csv(p, encoding=enc, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


# ---------- 1. entrada ----------
q = ler(P_Q, "utf-8")
for c in ("id", "texto_limpo"):
    if c not in q.columns:
        abortar("coluna ausente em questoes_limpas.csv: " + c)
q["id"] = q["id"].map(nid)
ids = q["id"].tolist()
textos = q["texto_limpo"].fillna("").astype(str).tolist()
n = len(ids)

with open(P_IDS, encoding="utf-8") as f:
    ids_cache = [nid(x) for x in json.load(f)]
if ids_cache != ids:
    abortar("emb_659_ids.json difere da lista de ids de questoes_limpas.csv")
E = np.asarray(np.load(P_NPY))

P = _carregar()
if E.shape != (n, int(P["Et"].shape[1])):
    abortar("shape de embeddings %s incompativel" % (E.shape,))
print("[DADOS] questoes = %d | embeddings %s" % (n, E.shape))

# ---------- 2. score S (replica de inferir_v9.classificar) ----------
Zt = _zrow((P["tfidf"].transform(textos) @ P["Xt"].T).toarray())
Ze = _zrow(E @ P["Et"].T)
Zch = _zrow((P["vec_ch"].transform(textos) @ P["Xch_t"].T).toarray())
S = P["w_tfidf"] * Zt + P["w_embed"] * Ze + P["w_char"] * Zch

parent = np.asarray(P["parent"])
depth = np.asarray(P["depth"])
eq = np.asarray(P["eq"])
Down1 = np.asarray(P["Down1"], dtype=bool)
Cm = eq[:, None] == eq[None, :]
EXCL = Cm | Down1 | Down1.T          # EXCL[j] == Cm[j] | Down1[j] | Down1[:, j]
tema_ids = [nid(x) for x in P["tema_ids"]]
caminhos = [norm_cam(x) for x in P["caminhos"]]
lim = P["limiar"]
lim = float("inf") if lim is None else float(lim)
TAU0 = float(P["tau"])
m_nos = S.shape[1]


def decidir(tau):
    pred = np.empty(n, dtype=int)
    m1 = np.empty(n, dtype=float)
    for i in range(n):
        j = int(S[i].argmax())
        guard = 0
        while parent[j] != -1 and depth[parent[j]] >= 2 and S[i, j] - S[i, parent[j]] < tau and guard < 50:
            j = int(parent[j])
            guard += 1
        outros = ~EXCL[j]
        m1[i] = S[i, j] - S[i][outros].max() if outros.any() else 99.0
        pred[i] = j
    return pred, m1


def status_de(m1):
    return np.where(m1 >= lim - 1e-4, "PUBLICAR", "REVISAR")


# ---------- 3. validacao obrigatoria ----------
pred0, m10 = decidir(TAU0)
ref = ler(P_REF, "utf-8-sig")
for c in ("id", "tema_id", "m1", "status"):
    if c not in ref.columns:
        abortar("coluna ausente em saida_classificacao_v9.csv: " + c)
ref["id"] = ref["id"].map(nid)
ref["tema_id"] = ref["tema_id"].map(nid)
problemas = []
if ref["id"].tolist() != ids:
    problemas.append("ids/ordem de saida_classificacao_v9.csv diferem da entrada")
else:
    mine_t = [tema_ids[j] for j in pred0]
    div = [(ids[i], mine_t[i], ref["tema_id"].iloc[i]) for i in range(n) if mine_t[i] != ref["tema_id"].iloc[i]]
    dm1 = np.abs(m10 - pd.to_numeric(ref["m1"]).values)
    st_mine = status_de(m10)
    st_div = int((st_mine != ref["status"].str.strip().values).sum())
    if div:
        problemas.append("%d divergencias de tema_id; exemplos: %s" % (len(div), div[:5]))
    if float(dm1.max()) >= 1e-6:
        problemas.append("max|dm1| = %.3e (>= 1e-6)" % float(dm1.max()))
    if st_div:
        problemas.append("%d status divergentes" % st_div)
if problemas:
    print("VALIDACAO FALHOU")
    for p in problemas:
        print("  - " + p)
    sys.exit(1)
print("[VALIDACAO] tau=%.2f: 0 divergencias de tema_id | max|dm1| = %.2e | status identicos" % (TAU0, float(dm1.max())))

# ---------- 4. exportacoes ----------
top5 = np.argsort(-S, axis=1, kind="stable")[:, :5]
rows = []
for i in range(n):
    for r, j in enumerate(top5[i], start=1):
        rows.append(dict(id=ids[i], rank=r, tema_id=tema_ids[j], caminho=caminhos[j],
                         depth=int(depth[j]), S=float(S[i, j])))
df_topk = pd.DataFrame(rows)

rows = []
for i in range(n):
    j = int(S[i].argmax())
    nivel, guard = 0, 0
    while True:
        rows.append(dict(id=ids[i], nivel=nivel, tema_id=tema_ids[j], caminho=caminhos[j],
                         depth=int(depth[j]), S=float(S[i, j])))
        if parent[j] == -1 or guard >= 50:
            break
        j = int(parent[j])
        nivel += 1
        guard += 1
df_cad = pd.DataFrame(rows)

sweep_rows = []
res = {}
for tau in TAUS:
    pr, m1 = decidir(float(tau))
    st = status_de(m1)
    res[tau] = (pr, m1, st)
    for i in range(n):
        j = int(pr[i])
        sweep_rows.append(dict(id=ids[i], tau=tau, tema_id=tema_ids[j], caminho=caminhos[j],
                               depth=int(depth[j]), m1=float(m1[i]), status=st[i]))
df_sweep = pd.DataFrame(sweep_rows)

df_topk.to_csv(O_TOPK, index=False, encoding="utf-8-sig")
df_cad.to_csv(O_CAD, index=False, encoding="utf-8-sig")
df_sweep.to_csv(O_SWEEP, index=False, encoding="utf-8-sig")
print("[SAIDA] %s (%d linhas)" % (O_TOPK, len(df_topk)))
print("[SAIDA] %s (%d linhas)" % (O_CAD, len(df_cad)))
print("[SAIDA] %s (%d linhas)" % (O_SWEEP, len(df_sweep)))

# ---------- 5. avaliacao ----------
fol = ler(P_FOL, "utf-8-sig")
for c in ("id", "tema_central_caminho", "fora_da_taxonomia"):
    if c not in fol.columns:
        abortar("coluna ausente em folha_verdade.csv: " + c)
fol["id"] = fol["id"].map(nid)
fol = fol.drop_duplicates("id", keep="first")
fol["fora"] = fol["fora_da_taxonomia"].map(nid)
fol["gold"] = fol["tema_central_caminho"].map(norm_cam)
fol = fol[(fol["fora"] != "1") & (fol["gold"] != "") & (fol["id"].isin(set(ids)))]
pos = {x: i for i, x in enumerate(ids)}
av_ids = fol["id"].tolist()
av_idx = np.array([pos[x] for x in av_ids], dtype=int)
gold = fol["gold"].tolist()
na = len(av_ids)
print("\n[AVALIACAO] n = %d (esperado 92)" % na)


def eh_anc(a, b):
    return b.startswith(a + SEP)


print("\n=== VARREDURA DE TAU (subida restrita; avaliacao por caminho) ===")
print("%7s %8s %6s %8s %10s %16s %12s" % ("tau", "n_exato", "n_PAI", "n_FILHO", "n_publicar", "n_exato_publ", "prec_publ%"))
for tau in TAUS:
    pr, m1, st = res[tau]
    ne = npai = nfil = npub = nexp = 0
    for k in range(na):
        i = av_idx[k]
        pc = caminhos[pr[i]]
        g = gold[k]
        ex = pc == g
        ne += int(ex)
        npai += int(eh_anc(pc, g))
        nfil += int(eh_anc(g, pc))
        if st[i] == "PUBLICAR":
            npub += 1
            nexp += int(ex)
    prec = (nexp / npub * 100) if npub else float("nan")
    marca = "  <-- congelado" if abs(float(tau) - 2.0) < 1e-12 else ""
    print("%7s %8d %6d %8d %10d %16d %12.1f%s" % (tau, ne, npai, nfil, npub, nexp, prec, marca))

# ---------- 6. recall@k e margens ----------
err = ler(P_ERR, "utf-8-sig")
for c in ("id", "tipo"):
    if c not in err.columns:
        abortar("coluna ausente em erros_tipo_v9.csv: " + c)
err["id"] = err["id"].map(nid)
err = err.drop_duplicates("id", keep="first")
tipo_map = dict(zip(err["id"], err["tipo"].str.strip().str.upper()))
tipos = [tipo_map.get(x, "") for x in av_ids]
g_if = np.array([t in ("IRMAO", "FILHO") for t in tipos])
g_ex = np.array([t == "EXATO" for t in tipos])

top_cam = [[caminhos[j] for j in top5[i]] for i in range(n)]
hit = {k: np.array([gold[k_] in top_cam[av_idx[k_]][:k] for k_ in range(na)]) for k in (1, 3, 5)}

print("\n=== RECALL@K POR CAMINHO (S bruto) ===")
print("%-22s %5s %9s %9s %9s" % ("grupo", "n", "recall@1", "recall@3", "recall@5"))
for nome, mk in (("todas (avaliadas)", np.ones(na, dtype=bool)), ("IRMAO+FILHO", g_if)):
    nn = int(mk.sum())
    if nn == 0:
        print("%-22s %5d %9s %9s %9s" % (nome, 0, "-", "-", "-"))
        continue
    print("%-22s %5d %8.1f%% %8.1f%% %8.1f%%" % (nome, nn, hit[1][mk].mean() * 100, hit[3][mk].mean() * 100, hit[5][mk].mean() * 100))

srt = np.sort(S, axis=1)
marg = srt[:, -1] - srt[:, -2]
mav = marg[av_idx]
print("\n=== MARGEM BRUTA S1-S2 ===")
print("%-14s %5s %9s %9s" % ("grupo", "n", "media", "mediana"))
for nome, mk in (("EXATO", g_ex), ("IRMAO+FILHO", g_if)):
    if int(mk.sum()) == 0:
        print("%-14s %5d %9s %9s" % (nome, 0, "-", "-"))
    else:
        print("%-14s %5d %9.4f %9.4f" % (nome, int(mk.sum()), float(mav[mk].mean()), float(np.median(mav[mk]))))

print("\nAVISO: n=92: ganho de tau so vale se formar plato em varios valores vizinhos; pico isolado = sobreajuste; qualquer mudanca exige lote novo rotulado")
print("\nTOPK OK")