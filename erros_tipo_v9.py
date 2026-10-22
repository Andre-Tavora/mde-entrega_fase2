import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
P_REG = BASE / "saida_regressao_v9.csv"
P_FOL = BASE / "folha_verdade.csv"
P_OUT = BASE / "erros_tipo_v9.csv"
SEP = " > "
TIPOS = ["EXATO", "EXTRA", "PAI", "FILHO", "IRMAO", "MESMO_RAMO", "OUTRO_RAMO"]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 90)


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def abortar(msg):
    print("ERROS TIPO FALHOU: " + msg)
    sys.exit(1)


def partes(c):
    return [x.strip() for x in str(c).split(SEP) if x.strip()]


def extras_set(v):
    return {nid(x) for x in re.split(r"[;,|\s]+", str(v)) if x.strip()}


# ---------- 1. leitura e juncao ----------
reg = pd.read_csv(P_REG, encoding="utf-8", dtype=str, keep_default_na=False)
fol = pd.read_csv(P_FOL, encoding="utf-8", dtype=str, keep_default_na=False)
if reg.columns[0].startswith("\ufeff"):
    reg.columns = [reg.columns[0].lstrip("\ufeff")] + list(reg.columns[1:])
if fol.columns[0].startswith("\ufeff"):
    fol.columns = [fol.columns[0].lstrip("\ufeff")] + list(fol.columns[1:])

for c in ("id", "tema_id", "caminho_previsto", "depth", "m1", "status"):
    if c not in reg.columns:
        abortar("coluna ausente em saida_regressao_v9.csv: " + c)
for c in ("id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"):
    if c not in fol.columns:
        abortar("coluna ausente em folha_verdade.csv: " + c)

reg["id"] = reg["id"].map(nid)
reg["tema_id"] = reg["tema_id"].map(nid)
reg["status"] = reg["status"].str.strip()
reg["m1"] = pd.to_numeric(reg["m1"], errors="coerce")
reg["depth"] = pd.to_numeric(reg["depth"], errors="coerce")
if reg["m1"].isna().any() or reg["depth"].isna().any():
    abortar("m1 ou depth nao numerico")
reg["depth"] = reg["depth"].astype(int)

fol["id"] = fol["id"].map(nid)
if fol["id"].duplicated().any():
    print("[AVISO] ids duplicados na folha; mantendo a primeira ocorrencia")
    fol = fol.drop_duplicates("id", keep="first")
fol_s = fol[["id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"]].rename(
    columns={"tema_central_id": "central", "tema_central_caminho": "central_cam",
             "extras_ids": "extras_f", "fora_da_taxonomia": "fora"})
m = reg[["id", "tema_id", "caminho_previsto", "depth", "m1", "status"]].merge(
    fol_s, on="id", how="left", indicator=True)
for c in ("central", "central_cam", "extras_f", "fora"):
    m[c] = m[c].fillna("")
m["central"] = m["central"].map(nid)
m["fora"] = m["fora"].map(nid)

excl = (m["fora"] == "1") | (m["central"] == "") | (m["_merge"] != "both")
n_excl = int(excl.sum())
d = m[~excl].copy().reset_index(drop=True)
print("[DADOS] regressao = %d | folha = %d | excluidas = %d (fora_da_taxonomia=1, central vazio ou sem folha) | validas = %d" % (
    len(reg), len(fol), n_excl, len(d)))
if len(d) == 0:
    abortar("nenhuma linha valida")


# ---------- 2. classificacao ----------
def tipo_de(r):
    if r["tema_id"] == r["central"]:
        return "EXATO"
    if r["tema_id"] in extras_set(r["extras_f"]):
        return "EXTRA"
    p = partes(r["caminho_previsto"])
    t = partes(r["central_cam"])
    if not p or not t:
        return "OUTRO_RAMO"
    if len(p) < len(t) and t[:len(p)] == p:
        return "PAI"
    if len(t) < len(p) and p[:len(t)] == t:
        return "FILHO"
    if len(p) >= 2 and len(p) == len(t) and p[:-1] == t[:-1] and p != t:
        return "IRMAO"
    if p[0] == t[0]:
        return "MESMO_RAMO"
    return "OUTRO_RAMO"


d["tipo"] = d.apply(tipo_de, axis=1)
n_idem = int(((d["tipo"] == "MESMO_RAMO") &
              (d.apply(lambda r: partes(r["caminho_previsto"]) == partes(r["central_cam"]), axis=1))).sum())
if n_idem:
    print("[NOTA] %d linha(s) com caminho identico ao correto mas tema_id diferente (ficaram em MESMO_RAMO)" % n_idem)


# ---------- 3. tabelas ----------
def tabela(chave, titulo):
    ct = pd.crosstab(d["tipo"], d[chave]).reindex(TIPOS).fillna(0).astype(int)
    tot = ct.sum(axis=0)
    out = pd.DataFrame(index=TIPOS)
    for col in ct.columns:
        out["n_%s" % col] = ct[col]
        out["%%_%s" % col] = (ct[col] / tot[col] * 100).round(1) if tot[col] else 0.0
    out["n_total"] = ct.sum(axis=1)
    out["%_total"] = (out["n_total"] / len(d) * 100).round(1)
    print("\n--- %s ---" % titulo)
    print("totais por coluna:", {str(k): int(v) for k, v in tot.items()})
    print(out.to_string())


print("\n=== 3. TIPOS DE ERRO ===")
tabela("status", "por status")
tabela("depth", "por depth previsto")

# ---------- 4. media de m1 ----------
g = d.groupby("tipo")["m1"].agg(["count", "mean", "median"]).reindex(TIPOS)
g["count"] = g["count"].fillna(0).astype(int)
print("\n=== 4. m1 POR TIPO ===")
print(g.rename(columns={"count": "n", "mean": "m1_media", "median": "m1_mediana"}).round(4).to_string())

# ---------- 5. PUBLICAR com erro ----------
ok_tipos = ("EXATO", "EXTRA")
pub_err = d[(d["status"] == "PUBLICAR") & (~d["tipo"].isin(ok_tipos))].sort_values("m1", ascending=False)
print("\n=== 5. PUBLICAR COM TIPO != EXATO/EXTRA: %d (mostrando ate 20) ===" % len(pub_err))
for _, r in pub_err.head(20).iterrows():
    print("id=%s | m1=%.4f | depth=%d | tipo=%s" % (r["id"], r["m1"], r["depth"], r["tipo"]))
    print("  previsto: " + r["caminho_previsto"])
    print("  correto : " + r["central_cam"])

# ---------- 6. pares mais frequentes ----------
err = d[~d["tipo"].isin(ok_tipos)]
print("\n=== 6. TOP 10 PARES previsto -> correto (erros: %d) ===" % len(err))
if len(err):
    pares = err.groupby(["caminho_previsto", "central_cam"]).size().sort_values(ascending=False).head(10)
    for (a, b), c in pares.items():
        print("%dx | %s  ->  %s" % (c, a, b))
else:
    print("nenhum erro")

# ---------- 7. saida ----------
saida = d[["id", "status", "depth", "m1", "tipo", "caminho_previsto", "central_cam"]].rename(
    columns={"central_cam": "tema_central_caminho"})
saida.to_csv(P_OUT, index=False, encoding="utf-8-sig")
print("\n[SAIDA] %s (%d linhas)" % (P_OUT, len(saida)))

print("\nERROS TIPO OK")