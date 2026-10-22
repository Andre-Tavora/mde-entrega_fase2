# -*- coding: utf-8 -*-
# Reexporta publicacao.csv com METRICA OFICIAL (gold, pai, filho ou extras_gold) e recalcula LI95
import pandas as pd, numpy as np, math
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub = pd.read_csv(BASE+"\\publicacao.csv", dtype=str)
fol = pd.read_csv(BASE+"\\folha_verdade.csv", dtype=str)
gold_map = fol.set_index("id").to_dict("index")
# carregar arvore de temas para resolver pai/filho
import pickle, os, sys
sys.path.insert(0, BASE)
import v10_comum as C
t = C.carregar_temas()
idx = {x:i for i,x in enumerate(t["tema_id"])}
cam = t["caminho"].tolist()
eq = np.asarray(C.grupos_equivalencia(t))
def eh_pai_filho(a,b):
    ia,ib = idx.get(str(a)), idx.get(str(b))
    if ia is None or ib is None: return False
    pa,pb = cam[ia].split(" > "), cam[ib].split(" > ")
    return (len(pb)==len(pa)+1 and pb[:len(pa)]==pa) or (len(pa)==len(pb)+1 and pa[:len(pb)]==pb)
def wilson_low(k,n,z=1.95996):
    if n==0: return 0.0
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    e=z*math.sqrt((p*(1-p)/n)+z*z/(4*n*n))/d
    return max(0.0,c-e)
publicadas = pub[pub.decisao=="publicar"].copy()
stats = {"acertos":0,"total":0,"por_gold":0,"pai_filho":0,"extras":0,"erros":[]}
for _,r in publicadas.iterrows():
    g = gold_map.get(r.id)
    if g is None: continue
    stats["total"] += 1
    gold = str(g.get("tema_central_id","")).strip()
    pred = str(r.tema_central_id).strip()
    ok = False
    if pred == gold or (eq[idx.get(pred,-1)] == eq[idx.get(gold,-1)] if idx.get(pred) is not None and idx.get(gold) is not None else False):
        ok=True; stats["por_gold"]+=1
    elif eh_pai_filho(pred,gold):
        ok=True; stats["pai_filho"]+=1
    else:
        ex = str(g.get("extras_ids","")).replace("nan","")
        if pred in ex.split(";"):
            ok=True; stats["extras"]+=1
    if ok: stats["acertos"]+=1
    else: stats["erros"].append(f"{r.id}: pred={pred} gold={gold}")
k,t = stats["acertos"], stats["total"]
print(f"METRICA OFICIAL (gold+equivalencia+pai/filho+extras):")
print(f"  TOTAL publicadas na folha: {t}")
print(f"  ACERTOS: {k}  =>  {100*k/t:.1f}%")
print(f"  LI95 = {wilson_low(k,t):.3f}")
print(f"  composicao: gold={stats['por_gold']} | pai/filho={stats['pai_filho']} | extras={stats['extras']}")
print(f"  erros restantes ({len(stats['erros'])}):")
for e in stats["erros"]: print("   -",e)