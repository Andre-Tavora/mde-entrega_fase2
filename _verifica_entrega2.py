# -*- coding: utf-8 -*-
# verifica v2: cast para str antes de comparar
import sys
sys.path.insert(0, r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
import v10_comum as C
import pandas as pd
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub = pd.read_csv(BASE + "\\publicacao.csv")
fol = pd.read_csv(BASE + "\\folha_verdade.csv")
pub["id_n"] = pub["id"].map(C.nid)
fol["id_n"] = fol["id"].map(C.nid)
m = pub.merge(fol[["id_n", "tema_central_id"]], on="id_n", how="left", suffixes=("", "_gold"))
pub_ = m[m.decisao == "publicar"]
g = pub_.dropna(subset=["tema_central_id_gold"]).copy()
g["pred_s"] = g["tema_central_id"].astype(str)
g["gold_s"] = g["tema_central_id_gold"].astype(str)
acertos = (g.pred_s == g.gold_s).sum()
print("publicadas no ouro: %d" % len(g))
print("acertos (string igual): %d (%.1f%%)" % (acertos, 100 * acertos / len(g)))
print("exemplos (id, pred, gold, acerto):")
for _, r in g.head(8).iterrows():
    print("  %s %s %s %s" % (r.id_n, r.pred_s, r.gold_s, "OK" if r.pred_s == r.gold_s else "X"))