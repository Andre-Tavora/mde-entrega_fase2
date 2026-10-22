# -*- coding: utf-8 -*-
# diagnostico por faixa de margem - precisao real (pai/filho/equivalencia)
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
BASE=Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE=BASE/"cache_offline"
sys.path.insert(0,str(BASE))
import v10_comum as C
q=C.ler_csv(CACHE/"questoes_limpas.csv"); q["id"]=q["id"].map(C.nid)
t=C.carregar_temas(); idx={x:i for i,x in enumerate(t["tema_id"])}
fol=C.ler_csv(BASE/"folha_verdade.csv"); fol["id"]=fol["id"].map(C.nid)
fora_col = fol["fora_da_taxonomia"].map(C.nid)=="1" if "fora_da_taxonomia" in fol else pd.Series(False,index=fol.index)
ok=(~fora_col)&fol["tema_central_id"].map(C.nid).isin(idx)&fol["id"].isin(set(q["id"]))
fol=fol[ok].reset_index(drop=True)
n=len(fol)
golds=[idx[x] for x in fol["tema_central_id"].map(C.nid)]
rows=[q["id"].tolist().index(x) for x in fol["id"]]
E=np.load(CACHE/"emb_659.npy").astype(np.float32); Et=np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(ngram_range=(1,2),max_features=20000,sublinear_tf=True,strip_accents="unicode",lowercase=True)
q0=q["texto_limpo"].fillna("").astype(str).tolist(); tf.fit(t["texto"].tolist()+q0)
Stf=(tf.transform(q0)@tf.transform(t["texto"].tolist()).T).toarray().astype(np.float32)
def z(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
S=0.6*z(E@Et.T)+0.4*z(Stf)
eq=np.asarray(C.grupos_equivalencia(t)); cam=t["caminho"].tolist()
def ordem(s,lim=5):
    out=[];vi=set()
    for j in np.argsort(-s,kind="stable"):
        if int(eq[j]) in vi: continue
        vi.add(int(eq[j])); out.append(int(j))
        if len(out)>=lim: break
    return out
def rel_oficial(g,p):
    if int(eq[g])==int(eq[p]): return True
    pa=cam[g].split(" > "); pb=cam[p].split(" > ")
    la,lb=len(pa),len(pb)
    return (lb==la+1 and pb[:la]==pa) or (la==lb+1 and pa[:lb]==pb)
def pai_direto(p,c):
    pa=cam[p].split(" > "); pc=cam[c].split(" > ")
    return len(pc)==len(pa)+1 and pc[:-1]==pa
prevs=[]
m1s=[]
for i in rows:
    s=S[i]; o=ordem(s)
    p=o[0]
    for k in (1,2):
        if k>=len(o): break
        if (s[o[0]]-s[o[k]])<0.5 and pai_direto(o[k],o[0]):
            p=o[k]; break
    prevs.append(p)
    eg=int(eq[p]); others=np.array([s[j] for j in range(len(t)) if int(eq[j])!=eg])
    m1s.append(float(s[p]-others.max()) if len(others) else 0.0)
prevs=np.array(prevs); m1s=np.array(m1s)
y=np.array([1 if rel_oficial(golds[i],prevs[i]) else 0 for i in range(n)])
df=pd.DataFrame(dict(m=m1s,y=y))
print("=== Precisao por faixa de margem m1 (precisao real, pai/filho/equiv) ===")
for lo,hi in [(0,0.5),(0.5,1),(1,1.5),(1.5,2),(2,3),(3,10)]:
    sub=df[(df.m>=lo)&(df.m<hi)]
    if len(sub)==0: continue
    prec=100*sub.y.mean()
    print("  m1 [%.1f,%.1f): n=%2d  acertos=%2d  precisao=%5.1f%%"%(lo,hi,len(sub),sub.y.sum(),prec))
print()
print("=== Acumulado de cima para baixo (publica os de maior margem) ===")
df2=df.sort_values("m",ascending=False).reset_index(drop=True)
cum=df2.y.cumsum(); cnt=np.arange(1,n+1); prec=100*cum/cnt
for idx_row in [5,10,15,20,25,30,40,50,60,70,80,92]:
    if idx_row>n: break
    i=idx_row-1
    print("  top-%2d: precisao=%5.1f%%  (corte m1>=%.3f)"%(idx_row,prec[i],df2.m[i]))
# quantos golds tem equiv alheio no top: diagnostico dos erros
eq_gold=[int(eq[g]) for g in golds]; eq_pred=[int(eq[p]) for p in prevs]
erros=[i for i in range(n) if not y[i]]
print()
print("=== Diagnostico dos %d erros (pai/filho): quao perto do gold? ==="%len(erros))
for i in erros[:15]:
    g=golds[i]; p=int(prevs[i])
    sg=S[rows[i]]
    o=ordem(sg)
    rank_gold = o.index(g) if g in o else -1
    print("  q=%s gold=%s(%d) pred=%s(%d) | top1m=%s rank_gold_no_top=%d m1=%.3f"%(
        str(fol["id"][i])[:12], str(fol["tema_central_id"][i])[:12], eq_gold[i],
        t["tema_id"][p], eq_pred[i], t["tema_id"][o[0]] if o else "?", rank_gold, m1s[i]))