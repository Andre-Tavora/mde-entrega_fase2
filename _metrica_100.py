# _metrica_100.py - avalia as 100 linhas de folha_verdade.csv contra publicacao.csv. Zero LLM.
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
P_FOL = BASE / "folha_verdade.csv"
P_PUB = BASE / "publicacao.csv"
SEP = " > "


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def ler_csv(p):
    """Le CSV com aspas simples/sem pandas (biblioteca padrao + csv)."""
    import csv
    if not p.exists():
        raise SystemExit("FALHOU: arquivo ausente: %s" % p)
    with open(p, encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        cols = [c.strip().lstrip("\ufeff") for c in (rd.fieldnames or [])]
        rows = []
        for row in rd:
            rows.append({k.strip(): (v or "") for k, v in row.items()})
    return cols, rows


def wilson95(k, n, z=1.96):
    """Intervalo de confianca de Wilson 95% (duas caudas) para proporcao."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def caminho_ok(pai, filho):
    """True se 'pai' e prefixo direto de 'filho' (1 nivel)."""
    p = [x.strip() for x in str(pai).split(SEP) if x.strip()]
    f = [x.strip() for x in str(filho).split(SEP) if x.strip()]
    if not p or not f:
        return False
    if len(f) == len(p) + 1 and f[:len(p)] == p:
        return True
    if len(p) == len(f) + 1 and p[:len(f)] == f:
        return True
    return False


# ---------- 1. leitura ----------
fol_cols, fol = ler_csv(P_FOL)
pub_cols, pub = ler_csv(P_PUB)

print("[DADOS] folha_verdade.csv: %d linhas | publicacao.csv: %d linhas" % (len(fol), len(pub)))

# ---------- 2. ligacao pred x gold ----------
pred_map = {}
for r in pub:
    pred_map[nid(r.get("id", ""))] = {
        "tema_id": nid(r.get("tema_central_id", "")),
        "extras": {nid(x) for x in str(r.get("extras_ids", "")).replace(";", ",").replace("|", ",").split(",") if nid(x)},
        "caminho": r.get("caminho_previsto", ""),
    }

# mapa de caminho por tema_id
caminho_tema = {}
for r in pub:
    tid = nid(r.get("tema_id", ""))
    if tid and tid not in caminho_tema:
        caminho_tema[tid] = r.get("caminho_previsto", "")

# ---------- 3. avaliacao ----------
def avaliar(com_out):
    a = b = c = 0
    erros = []
    for r in fol:
        qid = nid(r.get("id", ""))
        gold = nid(r.get("tema_central_id", "")) or ""
        fora = r.get("fora_da_taxonomia", "") == "1"
        print_gold = r.get("tema_central_caminho", "")
        pred = pred_map.get(qid, {})
        pred_tid = pred.get("tema_id", "")
        pred_cam = pred.get("caminho", "")

        # gold vazio: acerto apenas se fora_da_taxonomia=1
        if not gold:
            if com_out and fora:
                a += 1
                b += 1
                c += 1
            else:
                erros.append((qid, pred_tid or "-", gold or "-", "gold vazio s/ fora_da_taxonomia=1"))
            continue

        ok_a = (pred_tid == gold)
        ok_b = ok_a or (pred_tid in pred.get("extras", set()))

        # C: pai/filho direto
        ok_c = ok_b
        if not ok_c:
            gcam = print_gold or ""
            if gcam and pred_cam:
                ok_c = caminho_ok(gcam, pred_cam)
            elif gcam and pred_tid:
                pcam = caminho_tema.get(pred_tid, "")
                if pcam:
                    ok_c = caminho_ok(gcam, pcam)

        if ok_a:
            a += 1
        elif not com_out or not (not gold and fora):
            pass

        if ok_b:
            b += 1
        if ok_c:
            c += 1

        if not ok_c:
            erros.append((qid, pred_tid or "-", gold or "-", "sem acerto em C"))
    return a, b, c, erros


a, b, c, erros = avaliar(com_out=False)
a_out, b_out, c_out, erros_out = avaliar(com_out=True) if False else (a, b, c, erros)

# C_out: recalcular com OUT (gold vazio + fora=1 conta como acerto)
# Como o loop acima ja trata gold vazio s/ fora como erro, e com_out=True
# conta gold vazio+fora como acerto, refazemos aqui para C_out.
a2, b2, c2, erros2 = avaliar(com_out=True)
# C_out usa C da avaliacao com OUT
c_out_val = c2

# ---------- 4. saida ----------
def fmt(x, n_total=100):
    lo, hi = wilson95(x, n_total)
    return "%d/%d=%.1f%% (IC [%.1f%%, %.1f%%])" % (x, n_total, x / n_total * 100, lo * 100, hi * 100)

print("\n[METRICA_100] n=100")
print("A     : %s" % fmt(a))
print("B     : %s" % fmt(b))
print("C     : %s" % fmt(c))
print("C_out : %s" % fmt(c2))

print("\n--- ERROS (C, sem OUT) ---")
for e in erros:
    print("  id=%s | pred=%s | gold=%s | %s" % e)

print("\nMETRICA_100 OK")