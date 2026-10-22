# -*- coding: utf-8 -*-
# verifica v3: limpa sufixo .0 no gold, compara string sem ele
import sys, re
sys.path.insert(0, r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
import pandas as pd
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub = pd.read_csv(BASE + "\\publicacao.csv", dtype={"id": str, "tema_central_id": str})
fol = pd.read_csv(BASE + "\\folha_verdade.csv", dtype={"id": str, "tema_central_id": str})
fol["id_n"] = fol["id"].str.extract(r"(\d+)")[0]
pub["id_n"] = pub["id"].str.extract(r"(\d+)")[0]
m = pub.merge(fol[["id_n", "tema_central_id"]], on="id_n", how="left", suffixes=("", "_gold"))
pub_ = m[m.decisao == "publicar"]
g = pub_.dropna(subset=["tema_central_id_gold"]).copy()
def limpa(v):
    v = str(v).strip()
    return v[:-2] if v.endswith(".0") else v
g["pred_s"] = g["tema_central_id"].map(limpa)
g["gold_s"] = g["tema_central_id_gold"].map(limpa)
acertos = (g.pred_s == g.gold_s).sum()
print("publicadas no ouro: %d" % len(g))
print("acertos: %d (%.1f%%)" % (acertos, 100 * acertos / len(g)))
print("exemplos:")
for _, r in g.head(7).iterrows():
    print("  %s pred=%s gold=%s %s" % (r.id_n, r.pred_s, r.gold_s, "V" if r.pred_s == r.gold_s else "X"))
# se quiser incluir pai/filho