# valida_v10.py - checagens finais
import json
import numpy as np
import pandas as pd
import v10_comum as C

t = C.carregar_temas()
E = np.load(C.P_EMB_TEMAS)
ids = json.load(open(C.P_EMB_TEMAS_IDS, encoding="utf-8"))
erros = []
if E.shape[0] != len(t) or E.shape[0] != 377:
    erros.append("shape embeddings %s" % (E.shape,))
if ids != t["tema_id"].tolist():
    erros.append("ordem dos IDs diverge de temas_limpos.csv")
nm = np.linalg.norm(E, axis=1)
if float(np.abs(nm - 1).max()) > 1e-4:
    erros.append("norma L2 fora de 1 (max desvio %.2e)" % float(np.abs(nm - 1).max()))
rel = pd.read_csv(C.BASE / "relatorio_v10.csv", encoding="utf-8-sig")
par = pd.read_csv(C.BASE / "pares_confundidos_v10.csv", encoding="utf-8-sig")
if len(rel) == 0:
    erros.append("relatorio vazio")
if len(par) and par.groupby("conjunto").size().max() > 30:
    erros.append("mais de 30 pares em algum conjunto")
print(rel[["conjunto", "n", "modo", "top1_estrito", "top1_grupo", "top3_grupo", "top3_central_ou_extra"]].to_string(index=False))
if erros:
    print("VALIDACAO FALHOU: " + "; ".join(erros))
    raise SystemExit(1)
print("OK n=%d" % len(t))