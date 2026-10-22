import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
P_REG = BASE / "saida_regressao_v9.csv"
P_FOL = BASE / "folha_verdade.csv"
P_ERR = BASE / "erros_tipo_v9.csv"
P_OUT = BASE / "simula_recuo_v9.csv"
SEP = " > "
GRUPOS = ["EXATO", "EXTRA", "FILHO", "OUTROS"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 100)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def abortar(msg):
    print("SIMULA RECUO FALHOU: " + msg)
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


def norm_cam(s):
    return SEP.join([x.strip() for x in str(s).split(">") if x.strip()])


def pai_cam(s):
    partes = [x for x in s.split(SEP) if x]
    return SEP.join(partes[:-1])


# ---------- 1. leitura e juncao ----------
reg = ler(P_REG)
fol = ler(P_FOL)
err = ler(P_ERR)
exigir(reg, ["id", "caminho_previsto", "depth", "m1", "m2", "status"], "saida_regressao_v9.csv")
exigir(fol, ["id", "tema_central_id", "tema_central_caminho", "fora_da_taxonomia"], "folha_verdade.csv")
exigir(err, ["id", "tipo"], "erros_tipo_v9.csv")

for df in (reg, fol, err):
    df["id"] = df["id"].map(nid)
fol = fol.drop_duplicates("id", keep="first")
err = err.drop_duplicates("id", keep="first")

reg_s = reg[["id", "caminho_previsto", "depth", "m1", "m2", "status"]].copy()
fol_s = fol[["id", "tema_central_id", "tema_central_caminho", "fora_da_taxonomia"]].copy()
err_s = err[["id", "tipo"]].copy()

d = reg_s.merge(fol_s, on="id", how="left", indicator=True)
d = d.merge(err_s, on="id", how="left")
for c in ("tema_central_id", "tema_central_caminho", "fora_da_taxonomia", "tipo"):
    d[c] = d[c].fillna("").astype(str)
d["tema_central_id"] = d["tema_central_id"].map(nid)
d["fora_da_taxonomia"] = d["fora_da_taxonomia"].map(nid)
d["tipo"] = d["tipo"].str.strip().str.upper()

for c in ("m1", "m2", "depth"):
    d[c] = pd.to_numeric(d[c], errors="coerce")
    if d[c].isna().any():
        abortar("valores nao numericos em " + c)
d["depth"] = d["depth"].astype(int)

excl = ((d["fora_da_taxonomia"] == "1") | (d["tema_central_id"] == "")
        | (d["_merge"] != "both") | (d["tipo"] == ""))
print("[DADOS] regressao = %d | folha = %d | erros_tipo = %d" % (len(reg), len(fol), len(err)))
print("[EXCLUIDAS] %d (fora_da_taxonomia=1, tema_central vazio, sem folha ou sem tipo)" % int(excl.sum()))
d = d[~excl].drop(columns=["_merge"]).reset_index(drop=True)
n = len(d)
print("[VALIDAS] n = %d" % n)
if n == 0:
    abortar("nenhuma linha valida")

d["cam_prev"] = d["caminho_previsto"].map(norm_cam)
d["cam_cent"] = d["tema_central_caminho"].map(norm_cam)
d["tipo_g"] = np.where(d["tipo"].isin(["EXATO", "EXTRA", "FILHO"]), d["tipo"], "OUTROS")

is3 = (d["depth"] == 3).values
d3 = d[is3]
n3 = len(d3)

# ---------- 2. depth 3: estatisticas ----------
print("\n=== 2. DEPTH 3: n = %d ===" % n3)
if n3 == 0:
    abortar("nenhuma linha com depth 3")
print("n por tipo detalhado:", {k: int(v) for k, v in d3["tipo"].value_counts().items()})
for col in ("m1", "m2"):
    t = d3.groupby("tipo_g")[col].agg(n="count", media="mean", mediana="median", min="min", max="max")
    t = t.reindex(GRUPOS)
    print("\n--- %s por grupo (depth 3) ---" % col)
    print(t.round(4).to_string())
if n3 >= 3 and d3["m1"].std() > 0 and d3["m2"].std() > 0:
    corr = float(d3["m1"].corr(d3["m2"]))
    print("\ncorrelacao m1 x m2 (depth 3, Pearson): %.4f" % corr)
else:
    print("\ncorrelacao m1 x m2 (depth 3): indefinida (n pequeno ou variancia zero)")

# ---------- 3. acerto por caminho ----------
base_ok = (d["cam_prev"] == d["cam_cent"]).values
pai = d["cam_prev"].map(pai_cam)
pai_ok = (pai == d["cam_cent"]).values
is_extra = (d["tipo"] == "EXTRA").values
m1 = d["m1"].values
m2 = d["m2"].values
n_base = int(base_ok.sum())
print("\n=== 3. BASELINE (acerto por caminho) ===")
print("exato_cam total: %d/%d = %.2f%%" % (n_base, n, n_base / n * 100))
print("exato_cam em depth 3: %d/%d" % (int(base_ok[is3].sum()), n3))


def regra_mask(fam, a, b):
    if fam == "R_m1":
        r = m1 < a
    elif fam == "R_m2":
        r = m2 >= b
    else:
        r = (m1 < a) & (m2 >= b)
    return r & is3


def rotular(rec):
    final_ok = np.where(rec, pai_ok, base_ok)
    ganho = rec & (~base_ok) & final_ok
    perda = rec & ((base_ok & ~final_ok) | (is_extra & ~base_ok & ~final_ok))
    lab = np.full(n, "", dtype=object)
    lab[rec] = "NEUTRO"
    lab[ganho] = "GANHO"
    lab[perda] = "PERDA"
    return lab, final_ok, ganho, perda


# ---------- 4/5. simulacao ----------
rows = []


def avaliar(fam, a, b):
    rec = regra_mask(fam, a, b)
    lab, final_ok, ganho, perda = rotular(rec)
    g = int(ganho.sum())
    p = int(perda.sum())
    p_extra = int((rec & is_extra & ~base_ok & ~final_ok).sum())
    nf = int(final_ok.sum())
    rows.append(dict(
        familia=fam, a=a, b=b, n_recuados=int(rec.sum()), ganhos=g, perdas=p, perdas_extra=p_extra,
        saldo=g - p, exato_antes_n=n_base, exato_antes_pct=round(n_base / n * 100, 2),
        exato_depois_n=nf, exato_depois_pct=round(nf / n * 100, 2)))


a_vals = np.sort(d3["m1"].unique())
b_vals = np.sort(d3["m2"].unique())
for a in a_vals:
    avaliar("R_m1", float(a), np.nan)
for b in b_vals:
    avaliar("R_m2", np.nan, float(b))
qa = np.quantile(d3["m1"].values, [0.25, 0.5, 0.75])
qb = np.quantile(d3["m2"].values, [0.25, 0.5, 0.75])
for a in qa:
    for b in qb:
        avaliar("R_comb", float(a), float(b))

tab = pd.DataFrame(rows)
tab.to_csv(P_OUT, index=False, encoding="utf-8-sig")

COLS = ["a", "b", "n_recuados", "ganhos", "perdas", "perdas_extra", "saldo",
        "exato_antes_n", "exato_antes_pct", "exato_depois_n", "exato_depois_pct"]


def fmt_param(x):
    return "-" if pd.isna(x) else "%.4f" % x


print("\n=== 4/5. SIMULACAO DE RECUO (apenas depth 3; top 15 por SALDO desc, menos recuados no desempate) ===")
print("PERDAS inclui exato_cam->errado e tipo EXTRA recuado (perdas_extra = parcela de EXTRA)")
for fam in ("R_m1", "R_m2", "R_comb"):
    sub = tab[tab["familia"] == fam].sort_values(["saldo", "n_recuados"], ascending=[False, True]).head(15)
    disp = sub[COLS].copy()
    disp["a"] = disp["a"].map(fmt_param)
    disp["b"] = disp["b"].map(fmt_param)
    print("\n--- %s (%d configuracoes testadas) ---" % (fam, int((tab["familia"] == fam).sum())))
    print(disp.to_string(index=False))
print("\n[SAIDA] tabela completa (%d linhas): %s" % (len(tab), P_OUT))

# ---------- 6. melhor regra ----------
cand = tab[tab["n_recuados"] > 0]
if len(cand) == 0:
    cand = tab
best = cand.sort_values(["saldo", "n_recuados"], ascending=[False, True]).iloc[0]
fam, a, b = best["familia"], best["a"], best["b"]
rec_b = regra_mask(fam, a, b)
lab_b, _, _, _ = rotular(rec_b)
npar = 1 if fam in ("R_m1", "R_m2") else 2
print("\n=== 6. MELHOR REGRA: %s a=%s b=%s | parametros=%d | recuados=%d | ganhos=%d | perdas=%d | SALDO=%d ===" % (
    fam, fmt_param(a), fmt_param(b), npar, int(best["n_recuados"]), int(best["ganhos"]),
    int(best["perdas"]), int(best["saldo"])))
idxs = np.where(rec_b)[0]
idxs = sorted(idxs, key=lambda i: m1[i])
if not idxs:
    print("nenhum caso recuado")
for i in idxs:
    print("id=%s | m1=%.4f | m2=%.4f | tipo=%s | %s" % (d["id"][i], m1[i], m2[i], d["tipo"][i], lab_b[i]))
    print("  previsto -> final: %s  ->  %s" % (d["cam_prev"][i], pai[i]))
    print("  correto          : %s" % d["cam_cent"][i])

# ---------- 7. aviso ----------
print()
print("AVISO: n=%d (depth 3). SALDO <= 2 com n pequeno = ruido; regra com varios parametros ajustados nos mesmos 92 itens esta sobreajustada; so adotar com saldo >= 4 e regra de 1 parametro, validada em lote novo" % n3)

# ---------- 8. fim ----------
print("\nSIMULA RECUO OK")