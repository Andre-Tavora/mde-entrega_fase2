import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
P_CLS = BASE / "saida_classificacao_v9.csv"
P_REG = BASE / "saida_regressao_v9.csv"
P_OUT = BASE / "diagnostico_revisar_v9.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 70)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def abortar(msg):
    print("DIAGNOSTICO FALHOU: " + msg)
    sys.exit(1)


# ---------- 1. leitura ----------
df = pd.read_csv(P_CLS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
reg = pd.read_csv(P_REG, encoding="utf-8-sig", dtype=str, keep_default_na=False)
for c in ("id", "tema_id", "depth", "m1", "m2", "status"):
    if c not in df.columns:
        abortar("coluna ausente em saida_classificacao_v9.csv: " + c)
df["id"] = df["id"].map(nid)
df["tema_id"] = df["tema_id"].map(nid)
reg["id"] = reg["id"].map(nid)
for c in ("m1", "m2", "depth"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[c].isna().any():
        abortar("valores nao numericos em " + c)
df["depth"] = df["depth"].astype(int)
df["status"] = df["status"].str.strip()
df["pub"] = df["status"] == "PUBLICAR"
n = len(df)
npub = int(df["pub"].sum())
print("[DADOS] linhas = %d | PUBLICAR = %d (%.2f%%) | REVISAR = %d (%.2f%%)" % (
    n, npub, npub / n * 100, n - npub, (n - npub) / n * 100))


# ---------- 2. describe ----------
def q25(s): return s.quantile(0.25)
def q75(s): return s.quantile(0.75)
def med(s): return s.quantile(0.50)


def resumo(chave):
    for col in ("m1", "m2"):
        t = df.groupby(chave)[col].agg(["count", "min", q25, med, q75, "max"])
        t.columns = ["n", "min", "p25", "mediana", "p75", "max"]
        print("\n--- %s por %s ---" % (col, chave))
        print(t.round(4).to_string())


print("\n=== 2. DESCRIBE m1 / m2 ===")
resumo("status")
resumo("depth")

# ---------- 3. status e funcao deterministica de (m1, m2)? ----------
print("\n=== 3. REGRAS CANDIDATAS CONTRA status ===")
m1 = df["m1"].values
m2 = df["m2"].values
pub = df["pub"].values
u1 = np.unique(m1)
u2 = np.unique(m2)

res = []  # (erros, regra, a, b)
for a in u1:
    res.append((int(((m1 >= a) != pub).sum()), "m1>=a", float(a), None))
for b in u2:
    res.append((int(((m2 >= b) != pub).sum()), "m2>=b", None, float(b)))
best2 = (n + 1, None, None)
for a in u1:
    ma = (m1 >= a)[None, :]
    pred = ma & (m2[None, :] >= u2[:, None])
    err = (pred != pub[None, :]).sum(axis=1)
    k = int(err.argmin())
    if int(err[k]) < best2[0]:
        best2 = (int(err[k]), float(a), float(u2[k]))
    for kk in np.where(err == 0)[0]:
        res.append((0, "m1>=a e m2>=b", float(a), float(u2[kk])))
res.append((best2[0], "m1>=a e m2>=b", best2[1], best2[2]))

perfeitas = [r for r in res if r[0] == 0]
if perfeitas:
    print("Regras com acuracia 100%% (%d encontradas; mostrando ate 15):" % len(perfeitas))
    vistos = 0
    for e, nome, a, b in perfeitas:
        if vistos >= 15:
            break
        print("  %-16s a=%s b=%s" % (nome, "-" if a is None else "%.6f" % a, "-" if b is None else "%.6f" % b))
        vistos += 1
    por_tipo = {}
    for e, nome, a, b in perfeitas:
        por_tipo[nome] = por_tipo.get(nome, 0) + 1
    print("  perfeitas por tipo:", por_tipo)
    if not any(r[1] == "m2>=b" for r in perfeitas) and not any(r[1] == "m1>=a e m2>=b" for r in perfeitas):
        print("  => status e funcao apenas de m1 (m2 irrelevante).")
    # intervalo de limiares equivalentes para m1>=a
    a_ok = sorted(r[2] for r in perfeitas if r[1] == "m1>=a")
    if a_ok:
        print("  m1>=a: menor a=%.6f | maior a=%.6f" % (a_ok[0], a_ok[-1]))
        lo = float(m1[~pub].max()) if (~pub).any() else float("nan")
        hi = float(m1[pub].min()) if pub.any() else float("nan")
        print("  maior m1 em REVISAR = %.6f | menor m1 em PUBLICAR = %.6f" % (lo, hi))
else:
    melhor = min(res, key=lambda r: r[0])
    e, nome, a, b = melhor
    print("Nenhuma regra atinge 100%%. Melhor: %s a=%s b=%s | erros=%d | acuracia=%.2f%%" % (
        nome, a, b, e, (n - e) / n * 100))
    if nome == "m1>=a":
        pred = m1 >= a
    elif nome == "m2>=b":
        pred = m2 >= b
    else:
        pred = (m1 >= a) & (m2 >= b)
    viol = df[pred != pub][["id", "tema_id", "depth", "m1", "m2", "status"]].head(15)
    print("Linhas que violam (max 15):")
    print(viol.to_string(index=False))

# ---------- 4. PUBLICAR por depth e por tema ----------
print("\n=== 4. TAXA DE PUBLICAR ===")
td = df.groupby("depth")["pub"].agg(n="count", n_publicar="sum")
td["pct_publicar"] = td["n_publicar"] / td["n"] * 100
print("\nPor depth:")
print(td.round(2).to_string())

cam = df.groupby("tema_id")["caminho_previsto"].first() if "caminho_previsto" in df.columns else None
tt = df.groupby("tema_id")["pub"].agg(n="count", n_publicar="sum")
tt["n_revisar"] = tt["n"] - tt["n_publicar"]
tt["pct_publicar"] = tt["n_publicar"] / tt["n"] * 100
tt["m1_mediana"] = df.groupby("tema_id")["m1"].median()
tt["m2_mediana"] = df.groupby("tema_id")["m2"].median()
if cam is not None:
    tt["caminho_previsto"] = cam
top15 = tt.sort_values(["n", "pct_publicar"], ascending=[False, False]).head(15)
print("\nTop 15 temas por volume:")
cols = ["n", "n_publicar", "pct_publicar"] + (["caminho_previsto"] if cam is not None else [])
print(top15[cols].round(2).to_string())

# ---------- 5. validadas x restantes ----------
print("\n=== 5. 92 VALIDADAS x RESTANTES ===")
ids_val = set(reg["id"])
df["validada"] = df["id"].isin(ids_val)
for nome, sub in (("validadas", df[df["validada"]]), ("restantes", df[~df["validada"]])):
    k = len(sub)
    p = int(sub["pub"].sum())
    dist = {int(d): int(c) for d, c in sorted(sub["depth"].value_counts().items())}
    pct_d = {d: round(c / k * 100, 1) for d, c in dist.items()} if k else {}
    print("%-10s n=%d | PUBLICAR=%d (%.2f%%) | depth=%s | depth%%=%s | m1 mediana=%.4f" % (
        nome, k, p, (p / k * 100) if k else 0.0, dist, pct_d, sub["m1"].median() if k else float("nan")))

# ---------- 6. temas grandes com 0% PUBLICAR ----------
print("\n=== 6. TOP 10 TEMAS POR VOLUME COM 0%% PUBLICAR ===")
zero = tt[tt["n_publicar"] == 0].sort_values("n", ascending=False).head(10)
if len(zero):
    print(zero[["n", "m1_mediana", "m2_mediana"] + (["caminho_previsto"] if cam is not None else [])].round(4).to_string())
else:
    print("nenhum tema com 0% PUBLICAR")
n_zero = int((tt["n_publicar"] == 0).sum())
vol_zero = int(tt.loc[tt["n_publicar"] == 0, "n"].sum())
print("temas com 0%% PUBLICAR: %d de %d | questoes nesses temas: %d (%.1f%%)" % (n_zero, len(tt), vol_zero, vol_zero / n * 100))

# ---------- 7. saida ----------
saida = tt.reset_index()
ordem = ["tema_id"] + (["caminho_previsto"] if cam is not None else []) + [
    "n", "n_publicar", "n_revisar", "pct_publicar", "m1_mediana", "m2_mediana"]
saida = saida[ordem].sort_values(["n", "pct_publicar"], ascending=[False, False])
saida.to_csv(P_OUT, index=False, encoding="utf-8-sig")
print("\n[SAIDA] %s (%d temas)" % (P_OUT, len(saida)))

print("\nDIAGNOSTICO OK")