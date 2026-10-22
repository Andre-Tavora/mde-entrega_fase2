import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
P_REG = BASE / "saida_regressao_v9.csv"
P_FOL = BASE / "folha_verdade.csv"
P_OUT = BASE / "curva_calibracao_v9.csv"
LIM_ATUAL = 2.25
MIN_PUB = 8
SEP = " > "

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def abortar(msg):
    print("CALIBRACAO FALHOU: " + msg)
    sys.exit(1)


# ---------- 1. leitura e juncao ----------
reg = pd.read_csv(P_REG, encoding="utf-8-sig", dtype=str, keep_default_na=False)
fol = pd.read_csv(P_FOL, encoding="utf-8-sig", dtype=str, keep_default_na=False)
for c in ("id", "tema_id", "caminho_previsto", "m1", "status"):
    if c not in reg.columns:
        abortar("coluna ausente em saida_regressao_v9.csv: " + c)
for c in ("id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"):
    if c not in fol.columns:
        abortar("coluna ausente em folha_verdade.csv: " + c)

reg["id"] = reg["id"].map(nid)
reg["tema_id"] = reg["tema_id"].map(nid)
reg["status"] = reg["status"].str.strip()
reg["m1"] = pd.to_numeric(reg["m1"], errors="coerce")
if reg["m1"].isna().any():
    abortar("m1 nao numerico na regressao")

fol["id"] = fol["id"].map(nid)
if fol["id"].duplicated().any():
    print("[AVISO] ids duplicados na folha; mantendo a primeira ocorrencia")
    fol = fol.drop_duplicates("id", keep="first")
fol_s = fol[["id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"]].rename(
    columns={"tema_central_id": "central", "tema_central_caminho": "central_cam",
             "extras_ids": "extras_f", "fora_da_taxonomia": "fora"})
m = reg.merge(fol_s, on="id", how="left", indicator=True)
sem_folha = int((m["_merge"] != "both").sum())
for c in ("central", "central_cam", "extras_f", "fora"):
    m[c] = m[c].fillna("")
m["central"] = m["central"].map(nid)
m["fora"] = m["fora"].map(nid)
print("[DADOS] regressao = %d linhas | folha = %d linhas | sem correspondencia na folha: %d" % (len(reg), len(fol), sem_folha))

# ---------- 2. exclusoes ----------
excl = (m["fora"] == "1") | (m["central"] == "") | (m["_merge"] != "both")
n_excl = int(excl.sum())
d = m[~excl].copy().reset_index(drop=True)
n_valid = len(d)
print("[EXCLUIDAS] %d (fora_da_taxonomia=1, tema_central vazio ou sem folha) | n_valid = %d" % (n_excl, n_valid))
if n_valid == 0:
    abortar("nenhuma linha valida")


# ---------- 3. criterios de acerto ----------
def extras_set(v):
    return {nid(x) for x in re.split(r"[;,|\s]+", str(v)) if x.strip()}


def prefixo(a, b):
    """a e prefixo hierarquico de b (ou igual)."""
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    return a == b or b.startswith(a + SEP)


d["exato"] = d["tema_id"] == d["central"]
d["com_extras"] = d["exato"] | d.apply(lambda r: r["tema_id"] in extras_set(r["extras_f"]), axis=1)
d["linhagem"] = d["com_extras"] | d.apply(
    lambda r: prefixo(r["caminho_previsto"], r["central_cam"]) or prefixo(r["central_cam"], r["caminho_previsto"]), axis=1)
print("[ACERTO GLOBAL sobre n_valid] exato %d (%.1f%%) | com_extras %d (%.1f%%) | linhagem %d (%.1f%%)" % (
    d["exato"].sum(), d["exato"].mean() * 100, d["com_extras"].sum(), d["com_extras"].mean() * 100,
    d["linhagem"].sum(), d["linhagem"].mean() * 100))

# ---------- 4. curva ----------
v = d["m1"].values
ex = d["exato"].values
ce = d["com_extras"].values
li = d["linhagem"].values
ts = sorted(set(float(x) for x in v) | {LIM_ATUAL}, reverse=True)
linhas = []
for t in ts:
    mk = v >= t
    k = int(mk.sum())
    linhas.append(dict(
        t=t, n_publicar=k, cobertura=k / n_valid * 100,
        prec_exato=float(ex[mk].mean() * 100) if k else np.nan,
        prec_extras=float(ce[mk].mean() * 100) if k else np.nan,
        prec_linhagem=float(li[mk].mean() * 100) if k else np.nan,
        atual=bool(abs(t - LIM_ATUAL) < 1e-12)))
curva = pd.DataFrame(linhas)
curva.drop(columns=["atual"]).to_csv(P_OUT, index=False, encoding="utf-8-sig")

# tabela compacta: pontos onde n_publicar muda + t=2.25, reduzida a ~25
unicos = curva.drop_duplicates("n_publicar", keep="first")
idx_atual = curva.index[curva["atual"]].tolist()
if len(unicos) > 25:
    pos = np.unique(np.linspace(0, len(unicos) - 1, 24).round().astype(int))
    unicos = unicos.iloc[pos]
comp = pd.concat([unicos, curva.loc[idx_atual]]).drop_duplicates("t").sort_values("t", ascending=False)
print("\n=== 4. CURVA COBERTURA x PRECISAO (PUBLICAR = m1 >= t) ===")
print("%9s %6s %8s %9s %10s %13s" % ("t", "n_pub", "cob%", "p_exato", "p_extras", "p_linhagem"))
for _, r in comp.iterrows():
    marca = "  <== ATUAL (2.25)" if r["atual"] else ""
    print("%9.4f %6d %8.1f %9.1f %10.1f %13.1f%s" % (
        r["t"], r["n_publicar"], r["cobertura"], r["prec_exato"], r["prec_extras"], r["prec_linhagem"], marca))
print("[SAIDA] curva completa (%d limiares): %s" % (len(curva), P_OUT))

# ---------- 5. ponto atual e menores limiares ----------
at = curva[curva["atual"]].iloc[0]
print("\n=== 5. PONTO ATUAL E LIMIARES ALTERNATIVOS ===")
print("ATUAL t=2.25: n_publicar=%d | cob=%.1f%% | p_exato=%.1f%% | p_extras=%.1f%% | p_linhagem=%.1f%%" % (
    at["n_publicar"], at["cobertura"], at["prec_exato"], at["prec_extras"], at["prec_linhagem"]))


def menor_t(col, alvo):
    c = curva[(curva["n_publicar"] >= MIN_PUB) & (curva[col] >= alvo)]
    return None if c.empty else c.loc[c["t"].idxmin()]


for col in ("prec_exato", "prec_extras", "prec_linhagem"):
    for alvo in (80, 90):
        r = menor_t(col, alvo)
        if r is None:
            print("%-14s >= %d%% (n_pub >= %d): nenhum" % (col, alvo, MIN_PUB))
        else:
            print("%-14s >= %d%% (n_pub >= %d): menor t = %.4f | n_pub=%d | cob=%.1f%% | prec=%.1f%%" % (
                col, alvo, MIN_PUB, r["t"], r["n_publicar"], r["cobertura"], r[col]))

# ---------- 6. faixas de m1 ----------
bins = [-np.inf, 0.5, 1.0, 1.5, 2.25, 3.0, 4.0, np.inf]
labels = ["<0.5", "0.5-1", "1-1.5", "1.5-2.25", "2.25-3", "3-4", ">=4"]
d["faixa"] = pd.cut(d["m1"], bins=bins, labels=labels, right=False)
print("\n=== 6. PRECISAO POR FAIXA DE m1 ===")
print("%-10s %4s %9s %10s %13s" % ("faixa", "n", "p_exato", "p_extras", "p_linhagem"))
for lab in labels:
    s = d[d["faixa"] == lab]
    if len(s):
        print("%-10s %4d %9.1f %10.1f %13.1f" % (lab, len(s), s["exato"].mean() * 100,
                                                 s["com_extras"].mean() * 100, s["linhagem"].mean() * 100))
    else:
        print("%-10s %4d %9s %10s %13s" % (lab, 0, "-", "-", "-"))

# ---------- 7. colunas aceito_A1..A7 (todas as 92) ----------
print("\n=== 7. aceito_A* EM PUBLICAR x REVISAR (todas as %d linhas da regressao) ===" % len(reg))
pub = (reg["status"] == "PUBLICAR").values
npub, nrev = int(pub.sum()), int((~pub).sum())
print("PUBLICAR = %d | REVISAR = %d" % (npub, nrev))
print("%-10s %12s %12s" % ("coluna", "%True PUBL", "%True REV"))
for c in [c for c in reg.columns if c.startswith("aceito_")]:
    ok = (reg[c].str.strip().str.lower() == "true").values
    pp = ok[pub].mean() * 100 if npub else float("nan")
    pr = ok[~pub].mean() * 100 if nrev else float("nan")
    print("%-10s %12.1f %12.1f" % (c, pp, pr))

# ---------- 8/9. aviso e fim ----------
print("\nn pequeno: so aceite mudanca de limiar com ganho robusto; validar em lote novo")
print("\nCALIBRACAO OK")