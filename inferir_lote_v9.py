import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import time
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "saida_pipeline_v4"))

import numpy as np
import pandas as pd
from classificar_final import classificar, _carregar, _codificar

N_ESPERADO = 659
BLOCO = 64
P_Q = BASE / "cache_offline" / "questoes_limpas.csv"
P_REG = BASE / "saida_regressao_v9.csv"
P_NPY = BASE / "cache_offline" / "emb_659.npy"
P_IDS = BASE / "cache_offline" / "emb_659_ids.json"
P_OUT = BASE / "saida_classificacao_v9.csv"


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def falhar(motivo):
    print("LOTE FALHOU: " + motivo)
    sys.exit(1)


# ---------- 1. entrada ----------
q = pd.read_csv(P_Q, encoding="utf-8", dtype=str, keep_default_na=False)
for c in ("id", "texto_limpo"):
    if c not in q.columns:
        falhar("coluna ausente em questoes_limpas.csv: " + c)
if "leis_citadas" not in q.columns:
    print("[AVISO] coluna leis_citadas ausente; saida levara coluna vazia")
    q["leis_citadas"] = ""
q["id"] = q["id"].map(nid)

dup = q["id"][q["id"].duplicated()].unique().tolist()
if dup:
    falhar("ids duplicados em questoes_limpas.csv: " + ", ".join(dup[:10]))
n = len(q)
if n != N_ESPERADO:
    print("[AVISO] total de linhas = %d (esperado %d)" % (n, N_ESPERADO))

ids = q["id"].tolist()
textos = q["texto_limpo"].fillna("").astype(str).tolist()
n_vazios = sum(1 for t in textos if not t.strip())
print("[DADOS] questoes = %d | textos vazios = %d" % (n, n_vazios))

# ---------- 2. embeddings (cache) ----------
P = _carregar()
dim = int(P["Et"].shape[1])
E = None
if P_NPY.exists() and P_IDS.exists():
    with open(P_IDS, encoding="utf-8") as f:
        ids_cache = json.load(f)
    if ids_cache == ids:
        E_c = np.load(P_NPY)
        if E_c.shape == (n, dim):
            E = E_c
            print("[CACHE] embeddings reutilizados %s" % (E.shape,))
if E is None:
    t0 = time.time()
    nb = (n + BLOCO - 1) // BLOCO
    partes = []
    for b in range(nb):
        blk = textos[b * BLOCO:(b + 1) * BLOCO]
        partes.append(np.asarray(_codificar(blk, P)))
        print("[EMB] bloco %d/%d | %.1fs decorridos" % (b + 1, nb, time.time() - t0))
    E = np.vstack(partes)
    np.save(P_NPY, E)
    with open(P_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print("[CACHE] embeddings gerados e salvos %s" % (E.shape,))

# ---------- 3. classificacao ----------
pred = classificar(textos, embeddings=E, ids=ids).reset_index(drop=True)
if len(pred) != n:
    falhar("classificar devolveu %d linhas, entrada tem %d" % (len(pred), n))
pred["id"] = pred["id"].map(nid)
if pred["id"].tolist() != ids:
    falhar("ordem/ids da predicao diferem da entrada")
pred["leis_citadas"] = q["leis_citadas"].values
pred.to_csv(P_OUT, index=False, encoding="utf-8-sig")
print("[SAIDA] " + str(P_OUT))

# ---------- 4. consistencia com as 92 validadas ----------
reg = pd.read_csv(P_REG, encoding="utf-8-sig", dtype=str, keep_default_na=False)
reg["id"] = reg["id"].map(nid)
reg["tema_id"] = reg["tema_id"].map(nid)
novo = pred[["id", "tema_id", "status"]].copy()
novo["tema_id"] = novo["tema_id"].map(nid)
mg = reg[["id", "tema_id", "status"]].merge(novo, on="id", how="inner", suffixes=("_reg", "_novo"))
div = mg[(mg["tema_id_reg"] != mg["tema_id_novo"]) | (mg["status_reg"] != mg["status_novo"])]
print("\n=== CONSISTENCIA COM saida_regressao_v9.csv ===")
print("ids comparados: %d de %d | divergencias: %d (esperado: 0)" % (len(mg), len(reg), len(div)))
if len(div):
    print(div.head(20).to_string(index=False))

# ---------- 5. resumo ----------
npub = int((pred["status"] == "PUBLICAR").sum())
nrev = int((pred["status"] == "REVISAR").sum())
print("\n=== RESUMO ===")
print("total: %d" % n)
print("PUBLICAR: %d (%.2f%%) | REVISAR: %d (%.2f%%)" % (npub, npub / n * 100, nrev, nrev / n * 100))
print("depth:", {int(k): int(v) for k, v in sorted(pred["depth"].astype(int).value_counts().items())})
print("top 10 tema_id previstos:")
for tid, c in pred["tema_id"].value_counts().head(10).items():
    cam = pred.loc[pred["tema_id"] == tid, "caminho_previsto"].iloc[0]
    print("  %s | %dx | %s" % (tid, c, str(cam)[:70]))
print("textos vazios: %d" % n_vazios)
h = hashlib.sha256()
with open(P_OUT, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
print("SHA256 saida: " + h.hexdigest())

# ---------- 6. veredito ----------
n_out = len(pd.read_csv(P_OUT, encoding="utf-8-sig", dtype=str, keep_default_na=False))
motivos = []
if len(div):
    motivos.append("%d divergencias na checagem" % len(div))
if len(mg) == 0:
    motivos.append("nenhum id da regressao encontrado na saida")
if n_out != n:
    motivos.append("CSV com %d linhas, entrada com %d" % (n_out, n))
print()
if motivos:
    print("LOTE FALHOU: " + "; ".join(motivos))
    sys.exit(1)
print("LOTE OK")