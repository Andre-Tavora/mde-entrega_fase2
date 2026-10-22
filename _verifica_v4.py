# -*- coding: utf-8 -*-
# verifica v4: estrita + pai/filho/equiv
import sys, re, numpy as np
sys.path.insert(0, r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
import v10_comum as C
import pandas as pd
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub=pd.read_csv(BASE+"\\publicacao.csv",dtype={"id":str,"tema_central_id":str})
fol=pd.read_csv(BASE+"\\folha_verdade.csv",dtype={"id":str,"tema_central_id":str})
fol["id_n"]=fol["id"].str.extract(r"(\d+)")[0]
pub["id_n"]=pub["id"].str.extract(r"(\d+)")[0]
m=pub.merge(fol[["id_n","tema_central_id"]],on="id_n",how="left",suffixes=("","_gold"))
pub_=m[m.decisao=="publicar"].dropna(subset=["tema_central_id_gold"]).copy()
t=C.carregar_temas(); idx={x:i for i,x in enumerate(t["tema_id"])}; eq=np.asarray(C.grupos_equivalencia(t))
cam=t["caminho"].tolist()
def pai_filho(a,b):
    if a==b: return True
    pa=cam[a].split(" > "); pb=cam[b].split(" > ")
    la,lb=len(pa),len(pb)
    if lb==la+1 and pb[:la]==pa: return True
    if la==lb+1 and pa[:lb]==pb: return True
    return False
oficial=0; estrita=0; tot=0
for _,r in pub_.iterrows():
    a=str(r.tema_central_id).strip(); b=str(r.tema_central_id_gold).strip().replace(".0","")
    if a==b: oficial+=1; estrita+=1; tot+=1; continue
    ai=idx.get(a); bi=idx.get(b)
    if ai is not None and bi is not None:
        if int(eq[ai])==int(eq[bi]): oficial+=1; tot+=1; continue
        if pai_filho(ai,bi): oficial+=1; tot+=1; continue
    tot+=1
print("top-30 publ com ouro: n=%d" % len(pub_))
print("estrita:                   %d (%.1f%%)" % (estrita, 100*estrita/tot))
print("estrita+pai/filho/equiv:   %d (%.1f%%)" % (oficial, 100*oficial/tot))
print()
print("--- analise dos %d erros na metrica oficial ---" % (tot-oficial))
for _,r in pub_.iterrows():
    a=str(r.tema_central_id).strip(); b=str(r.tema_central_id_gold).strip().replace(".0","")
    if a==b: continue
    ai=idx.get(a); bi=idx.get(b)
    ok= False
    if ai is not None and bi is not None:
        if int(eq[ai])==int(eq[bi]): ok=True
        elif pai_filho(ai,bi): ok=True
    if not ok:
        print("  %s pred=%s gold=%s" % (r.id_n, a, b))