# -*- coding: utf-8 -*-
# varredura m1 x smax v2 — bug da format string corrigido
import sys, math, os
sys.path.insert(0, r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
import v10_comum as C
import pandas as pd, numpy as np
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub=pd.read_csv(BASE+"\\publicacao.csv",dtype={"id":str,"tema_central_id":str})
fol=pd.read_csv(BASE+"\\folha_verdade.csv",dtype={"id":str,"tema_central_id":str})
fol["id_n"]=fol["id"].str.extract(r"(\d+)")[0]
pub["id_n"]=pub["id"].str.extract(r"(\d+)")[0]
m=pub.merge(fol[["id_n","tema_central_id"]],on="id_n",how="left",suffixes=("","_gold"))
pub_=m[m.decisao=="publicar"].dropna(subset=["tema_central_id_gold"]).copy()
t=C.carregar_temas(); idx={x:i for i,x in enumerate(t["tema_id"])}; eq=np.asarray(C.grupos_equivalencia(t))
cam=t["caminho"].tolist()
def pf(a,b):
    if a==b: return True
    pa=cam[a].split(" > "); pb=cam[b].split(" > ")
    la,lb=len(pa),len(pb)
    return (lb==la+1 and pb[:la]==pa) or (la==lb+1 and pa[:lb]==pb)
def ok_oficial(a,b):
    a=str(a).strip(); b=str(b).strip().replace(".0","")
    if a==b: return True
    ai=idx.get(a); bi=idx.get(b)
    if ai is None or bi is None: return False
    return int(eq[ai])==int(eq[bi]) or pf(ai,bi)
def wilson_low(k,n,z=1.95996):
    if n==0: return 0.0
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    e=z*math.sqrt((p*(1-p)/n)+z*z/(4*n*n))/d
    return max(0.0,c-e)
# verificar se ha coluna smax
pub_["confianca_f"]=pub_["confianca"].astype(float)
# varredura
melhores=[]
for m1c in [1.20,1.30,1.38,1.50,1.60,1.80,2.00,2.20,2.50]:
    for smc in [0.40,0.50,0.60,0.70,0.80,0.90]:
        sub2=pub_[(pub_.confianca_f>=m1c)&(pub_.confianca_f>=smc)]
        if len(sub2)==0: continue
        k=sum(1 for _,r in sub2.iterrows() if ok_oficial(r.tema_central_id, r.tema_central_id_gold))
        li=wilson_low(k,len(sub2))
        melhores.append((li,k,len(sub2),m1c,smc))
melhores.sort(reverse=True)
print("Top 10 combinacoes (limite inferior Wilson 95%):")
print("%-6s %-5s %-6s %-4s %-4s" % ("LI95","ac/to","n","m1","smax"))
for li,k,tot,m1c,smc in melhores[:10]:
    print("%.3f %d/%-3d %d %.2f %.2f" % (li,k,tot,m1c,smc))
# linha atual
cur=pub_[pub_.confianca_f>=1.38]
k=sum(1 for _,r in cur.iterrows() if ok_oficial(r.tema_central_id,r.tema_central_id_gold))
print("\n[corte atual m1>=1.38]: %d/%d = %.1f%% | LI95=%.3f (n=%d)" % (k,len(cur),100*k/len(cur),wilson_low(k,len(cur)),len(cur)))