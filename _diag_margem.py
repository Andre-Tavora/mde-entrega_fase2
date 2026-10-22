# -*- coding: utf-8 -*-
"""Diagnostico de margem: o gold esta no top-k do ranking por cosseno? (offline, sem LLM)"""
import json, re, sys, unicodedata
from pathlib import Path

import numpy as np

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE / "cache_offline"
FOLHA = BASE / "folha_verdade.csv"

P_T_NPY = CACHE / "emb_temas_v10.npy"
P_T_IDS = CACHE / "emb_temas_v10_ids.json"
P_T_NPY_B = CACHE / "emb_temas_v10b.npy"
P_F_NPY = CACHE / "emb_folha.npy"
P_F_IDS = CACHE / "emb_folha_ids.json"
P_TEMAS_CSV = CACHE / "temas_limpos.csv"
P_FOL_CSV = BASE / "folha_verdade.csv"


def nid(v):
    if v is None:
        return ""
    return re.sub(r"\.0$", "", str(v).strip())


def falhar(msg):
    print("DIAG_MARGEM FALHOU: " + msg)
    sys.exit(1)


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def norm_blob(s):
    return re.sub(r"\s+", " ", sem_acento(str(s)).strip().lower())


# ---------- 1. entradas obrigatorias (sem re-embeddings) ----------
for p in (P_T_NPY, P_T_IDS, P_F_NPY, P_F_IDS, P_TEMAS_CSV, P_FOL_CSV):
    if not p.is_file():
        falhar("arquivo ausente: " + str(p))

Et = np.load(P_T_NPY, allow_pickle=False).astype(np.float32).squeeze()
with open(P_T_IDS, encoding="utf-8") as f:
    ids_t = [nid(x) for x in json.load(f)]
Ef = np.load(P_F_NPY, allow_pickle=False).astype(np.float32).squeeze()
with open(P_F_IDS, encoding="utf-8") as f:
    ids_f = [nid(x) for x in json.load(f)]

# dims da ultima camada precisam bater; senao fallback para emb_temas_v10b
if Et.ndim == 2 and Ef.ndim == 2 and Et.shape[1] == Ef.shape[1]:
    base_nome = "emb_temas_v10.npy"
else:
    if not P_T_NPY_B.is_file():
        falhar("dimensoes divergentes e emb_temas_v10b.npy ausente para fallback")
    Et = np.load(P_T_NPY_B, allow_pickle=False).astype(np.float32).squeeze()
    if not (Et.ndim == 2 and Ef.ndim == 2 and Et.shape[1] == Ef.shape[1]):
        falhar("dimensoes ainda divergentes apos fallback para emb_temas_v10b.npy")
    base_nome = "emb_temas_v10b.npy (fallback)"

if Et.ndim != 2 or Ef.ndim != 2 or len(ids_t) != Et.shape[0] or len(ids_f) != Ef.shape[0]:
    falhar("forma inesperada (Et=%s ids_t=%d | Ef=%s ids_f=%d)" % (Et.shape, len(ids_t), Ef.shape, len(ids_f)))
if len(ids_t) != 377 or len(ids_f) != 92:
    falhar("contagens inesperadas (temas=%d, folha=%d); esperado 377 e 92" % (len(ids_t), len(ids_f)))

print("[DADOS] base=%s | temas=%d | folha(com emb)=%d" % (base_nome, len(ids_t), len(ids_f)))

# ---------- 2. ler folha_verdade (dtype str, keep_default_na False, remover BOM) ----------
import csv

with open(FOLHA, encoding="utf-8-sig", newline="") as f:
    rd = csv.DictReader(f)
    cols = [c.strip().lstrip("\ufeff") for c in (rd.fieldnames or [])]
    fol = []
    for row in rd:
        fol.append({k.strip().lstrip("\ufeff"): (v or "") for k, v in row.items()})

def col_df(rows, *alvos):
    for a in alvos:
        for c in cols:
            if c.strip().lower() == a.lower():
                return c
    return None

CID = col_df(fol, "id")
CGOLD = col_df(fol, "tema_central_id")
CFORA = col_df(fol, "fora_da_taxonomia")
CGOLD_CAM = col_df(fol, "tema_central_caminho")
if not (CID and CGOLD):
    falhar("folha_verdade.csv sem colunas id/tema_central_id; encontradas: " + ", ".join(cols))

# ---------- 3. mapa de caminhos/ids de temas para resolver nomes ----------
for c in ("tema_id", "caminho", "nome"):
    if c not in cols:
        pass

temas_path = {}
try:
    with P_TEMAS_CSV.open(encoding="utf-8-sig", newline="") as f:
        rd2 = csv.DictReader(f)
        cols2 = [c.strip().lstrip("\ufeff") for c in (rd2.fieldnames or [])]
        tid_c = next((c for c in cols2 if c.lower() in ("tema_id", "id")), None)
        cam_c = next((c for c in cols2 if c.lower() == "caminho"), None)
        if tid_c and cam_c:
            for row in rd2:
                temas_path[nid(row.get(tid_c, ""))] = row.get(cam_c, "")
except Exception as e:
    print("[AVISO] nao foi possivel ler temas_limpos.csv para resolver caminhos: %s" % e)

# ---------- 4. indice id -> idx ----------
fpos = {qid: i for i, qid in enumerate(ids_f)}

# ---------- 5. normalizar / rodar ----------
def norm_v(v):
    if not v.strip():
        return 0.0
    return float(np.linalg.norm(v))

Ef_n = Ef / (np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-12)
Et_n = Et / (np.linalg.norm(Et, axis=1, keepdims=True) + 1e-12)

stats = {"top1": 0, "top3": 0, "top5": 0, "vizinho_faixa": 0, "fora_faixa": 0, "sem_embedding": 0, "sem_gold": 0, "OUT": 0}
detalhes = []

for row in fol:
    qid = nid(row.get(CID, ""))
    gold = nid(row.get(CGOLD, ""))
    fora = nid(row.get(CFORA, "")) if CFORA else ""
    gold_cam = row.get(CGOLD_CAM, "") if CGOLD_CAM else ""

    if qid not in fpos:
        stats["sem_embedding"] += 1
        detalhes.append((qid, "", gold, None, "sem_embedding", gold_cam))
        continue

    if not gold:
        # sem gold: so OUT se fora_da_taxonomia=1
        if fora == "1":
            stats["OUT"] += 1
        else:
            stats["sem_gold"] += 1
        continue

    qi = fpos[qid]
    sim = Ef_n[qi] @ Et_n.T                     # cosseno (577x)
    top5 = np.argsort(-sim, kind="stable")[:5]

    pred = ids_t[top5[0]]
    score_top1 = float(sim[top5[0]])
    tops = [ids_t[j] for j in top5]
    scores = [float(sim[j]) for j in top5]

    pred_num = re.match(r"^(\d+)", pred)
    gold_num = re.match(r"^(\d+)", gold)
    pn = int(pred_num.group(1)) if pred_num else None
    gn = int(gold_num.group(1)) if gold_num else None

    if gold == pred:
        stats["top1"] += 1
    elif gold in tops[:3]:
        stats["top3"] += 1
    elif gold in tops[:5]:
        stats["top5"] += 1
    elif pn is not None and gn is not None and (pn - 10) <= gn <= (pn + 10):
        stats["vizinho_faixa"] += 1
        detalhes.append((qid, pred, gold, score_top1, "vizinho_faixa", gold_cam, tops, scores))
    else:
        stats["fora_faixa"] += 1
        detalhes.append((qid, pred, gold, score_top1, "fora_faixa", gold_cam, tops, scores))

# ---------- 6. resumo ----------
n = len(fol)
com = sum(stats[k] for k in ("top1", "top3", "top5", "vizinho_faixa", "fora_faixa", "sem_embedding", "sem_gold", "OUT"))
com = len([r for r in fol if nid(r.get(CID,"")) in fpos])  # com_embeddings = presentes no embedding
den = max(1, com)

def pct(v):
    return v / den * 100

print("[DIAG] total_folha=%d | com_embeddings=%d | sem_embedding=%d | sem_gold=%d | top1=%d (%.1f%%) | top3=%d (%.1f%%) | top5=%d (%.1f%%) | vizinho_faixa=%d (%.1f%%) | fora_faixa=%d (%.1f%%) | OUT=%d" % (
    n, com, stats["sem_embedding"], stats["sem_gold"],
    stats["top1"], pct(stats["top1"]), stats["top3"], pct(stats["top3"]),
    stats["top5"], pct(stats["top5"]), stats["vizinho_faixa"], pct(stats["vizinho_faixa"]),
    stats["fora_faixa"], pct(stats["fora_faixa"]), stats["OUT"]))

# ---------- 7. detalhamento ----------
t5_str = lambda tops, scores: "(" + ", ".join("%s:%.4f" % (t, s) for t, s in zip(tops, scores)) + ")"
for d in detalhes:
    if d[4] == "sem_embedding":
        qid, pred, gold, score, cls, gcam, tops, scores = d + (None, None, None, None)
        pred = ""
        score = None
        t5 = ""
    else:
        qid, pred, gold, score, cls, gcam, tops, scores = d
        t5 = t5_str(tops, scores)
    gc = temas_path.get(gold, "") or gcam or ""
    print("id=%s | pred=%s | gold=%s | score_top1=%s | top5=%s | class=%s | gold_caminho=%s" % (
        qid, pred, gold, ("%.4f" % score) if score is not None else "", t5, cls, gc[:70]))

# ---------- 8. conclusao ----------
if (stats["top1"] + stats["top3"]) >= 0.6 * com:
    print("DIAG: kNN acha o tema no top-3; problema e decisao top-1 (sugere top-k+calibracao)")
elif (stats["vizinho_faixa"] + stats["fora_faixa"]) > 0.3 * com:
    print("DIAG: representacao do texto erra o ramo; precisa revisar texto/semantica")
else:
    print("DIAG: padrao misto (top-3 + vizinho); revisar ambos")

print("DIAG_MARGEM OK")