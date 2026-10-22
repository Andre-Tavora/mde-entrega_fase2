# -*- coding: utf-8 -*-
# verifica publicacao.csv x folha (mesmo normalizador C.nid nos dois lados)
import sys
sys.path.insert(0, r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
import v10_comum as C
import pandas as pd
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub = pd.read_csv(BASE + "\\publicacao.csv")
fol = pd.read_csv(BASE + "\\folha_verdade.csv")
pub["id_n"] = pub["id"].map(C.nid)
fol["id_n"] = fol["id"].map(C.nid)
fol["g_n"] = fol["tema_central_id"].map(C.nid)
m = pub.merge(fol[["id_n", "g_n", "fora_da_taxonomia"]], on="id_n", how="left")
pub_ = m[m.decisao == "publicar"]
g = pub_.dropna(subset=["g_n"])
print("publicadas: %d | delas com ouro: %d" % (len(pub_), len(g)))
if len(g):
    acertos = (g.g_n == g.tema_central_id).sum()
    print("acertos (tema igual): %d (%.1f%%)" % (acertos, 100 * acertos / len(g)))
    print("primeiras 5 publicadas no ouro:")
    print(g[["id", "tema_central_id", "g_n"]].head().to_string())
else:
    print("NENHUMA publicada casa com ouro pelo id normalizado - conferir formato dos ids")