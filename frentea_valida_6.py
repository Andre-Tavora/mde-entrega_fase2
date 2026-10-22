# frentea_valida_6.py - valida os 6 pares pred x gold com o enunciado da folha. Zero LLM.
import sys, re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
P_FOL = BASE / "folha_verdade.csv"
P_Q   = BASE / "cache_offline" / "questoes_limpas.csv"
P_SAI = BASE / "saida_classificacao_v9.csv"

TEXTOS = {
    "0077": "genocídio - decreto 2889 podera ser declarado unico?",
    "0058": "vitimologia e criminologia atuais - fundador?",
    "0208": "penas dos participantes simultaneos de crime - divisao",
    "0279": "maus tratos - indulto ou reclusap?",
    "0316": "infanticidio e diferenca - forma privilegiada de homicidio",
    "0319": "pessoa que participa de crime com repo fracionada - coautoria?",
}

IDS = ["0058", "0077", "0208", "0279", "0316", "0319"]

def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s

def ler(p, cols):
    if not p.exists():
        raise SystemExit("FALHOU: arquivo ausente: %s" % p)
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for c in cols:
        if c not in df.columns:
            raise SystemExit("FALHOU: coluna ausente em %s: %s" % (p.name, c))
    return df

# ---------- folha ----------
fol = ler(P_FOL, ["id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"])
fol["id"] = fol["id"].map(nid)
fol["central"] = fol["tema_central_id"].map(nid)
fol = fol[(fol["fora_da_taxonomia"].map(nid) != "1") & (fol["central"] != "")].drop_duplicates("id", keep="first")
fmap = dict(zip(fol["id"], fol["central"]))
fcm  = dict(zip(fol["id"], fol["tema_central_caminho"]))
fex  = dict(zip(fol["id"], fol["extras_ids"]))

# ---------- pred ----------
cls = ler(P_SAI, ["id", "tema_id", "caminho_previsto", "m1", "m2", "status"])
cls["id"] = cls["id"].map(nid)
cls["tema_id"] = cls["tema_id"].map(nid)
pmap = dict(zip(cls["id"], cls["tema_id"]))
pcm  = dict(zip(cls["id"], cls["caminho_previsto"]))
pm1  = dict(zip(cls["id"], cls["m1"]))
pst  = dict(zip(cls["id"], cls["status"]))

# ---------- questoes ----------
q = ler(P_Q, ["id", "texto_limpo"])
q["id"] = q["id"].map(nid)
qmap = dict(zip(q["id"], q["texto_limpo"]))

# ---------- temas (para resolver nome dos temas) ----------
t = ler(BASE / "cache_offline" / "temas_limpos.csv", ["tema_id", "caminho"])
t["tema_id"] = t["tema_id"].map(nid)
tid2cam = dict(zip(t["tema_id"], t["caminho"]))
tid2nome = {k: v.split(" > ")[-1] for k, v in tid2cam.items()}

print("=" * 100)
print("FRENTE A - VALIDACAO DOS 6 PARES pred x gold (folha_verdade.csv)")
print("=" * 100)
for qid in IDS:
    print("\n" + "=" * 100)
    print("ID: %s" % qid)
    print("-" * 100)
    txt = qmap.get(qid, "")
    print("[ENUNCIADO da folha]")
    print("  " + re.sub(r"\s+", " ", txt)[:600] + ("..." if len(txt) > 600 else ""))
    if qid in TEXTOS and not txt.strip():
        print("  (enunciado vazio no CSV - verbete de busca anomalo)")
    print("")
    p  = pmap.get(qid, "??")
    g  = fmap.get(qid, "??")
    print("[PRED (modelo)]  tema=%s | m1=%s | status=%s" % (p, pm1.get(qid, "?"), pst.get(qid, "?")))
    print("  %s" % pcm.get(qid, ""))
    print("  -> %s" % tid2nome.get(p, "?"))
    print("[GOLD (folha)]   tema=%s" % g)
    print("  %s" % fcm.get(qid, ""))
    print("  -> %s" % tid2nome.get(g, "?"))
    print("[EXTRAS folha]   %s" % fex.get(qid, ""))
    # julgamento
    if p == g:
        print("[VEREDITO] PRED == GOLD (nada a fazer)")
    else:
        print("[VEREDITO] divergem - verificar se o enunciado aponta para PRED ou para GOLD")
print("\nFIM")