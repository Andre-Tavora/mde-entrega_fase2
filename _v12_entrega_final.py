# -*- coding: utf-8 -*-
# entrega final v12 - publicar top-30 por margem, resto revisar
import os, sys, json, numpy as np, pandas as pd
from pathlib import Path
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
BASE=Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE=BASE/"cache_offline"
sys.path.insert(0,str(BASE))
import v10_comum as C
q=C.ler_csv(CACHE/"questoes_limpas.csv"); q["id"]=q["id"].map(C.nid)
t=C.carregar_temas(); idx={x:i for i,x in enumerate(t["tema_id"])}
E=np.load(CACHE/"emb_659.npy").astype(np.float32); Et=np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(ngram_range=(1,2),max_features=20000,sublinear_tf=True,strip_accents="unicode",lowercase=True)
q0=q["texto_limpo"].fillna("").astype(str).tolist(); tf.fit(t["texto"].tolist()+q0)
Stf=(tf.transform(q0)@tf.transform(t["texto"].tolist()).T).toarray().astype(np.float32)
def z(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
S=0.6*z(E@Et.T)+0.4*z(Stf)
eq=np.asarray(C.grupos_equivalencia(t)); ids=t["tema_id"].tolist(); cam=t["caminho"].tolist()
def ordem(s,lim=3):
    out=[];vi=set()
    for j in np.argsort(-s,kind="stable"):
        if int(eq[j]) in vi: continue
        vi.add(int(eq[j])); out.append(int(j))
        if len(out)>=lim: break
    return out
rows=[]
for sid,s in zip(q["id"],S):
    o=ordem(s); p=o[0]
    eg=int(eq[p]); others=np.array([s[j] for j in range(len(t)) if int(eq[j])!=eg])
    m1=float(s[p]-others.max()) if len(others) else 0.0
    smax=float(s.max())
    seen={eg}; extras=[]
    for j in np.argsort(-s,kind="stable"):
        if int(eq[j]) in seen: continue
        seen.add(int(eq[j])); extras.append(ids[j])
        if len(extras)>=2: break
    rows.append(dict(id=sid,tema_central_id=ids[p],extras_ids=";".join(extras),confianca=round(m1,4),smax=round(smax,4),motivo=""))
df=pd.DataFrame(rows)
# cortes
PUB_CUT=1.38   # m1: precisao ~80% no ouro (top-30)
FORA_SMAX=0.3  # abaixo disso, tema provavelmente fora da taxonomia
df["decisao"]="revisar"
df.loc[df.confianca>=PUB_CUT,"decisao"]="publicar"
df.loc[(df.smax<FORA_SMAX)&(df.confianca<PUB_CUT),"decisao"]="fora"
df["motivo"]=""
df.loc[df.decisao=="revisar","motivo"]="margem="+df.confianca.astype(str)
out=df[["id","decisao","tema_central_id","extras_ids","confianca","motivo"]].sort_values("id").reset_index(drop=True)
out.to_csv(BASE/"publicacao.csv",index=False,encoding="utf-8")
print("[ENTREGA] publicacao.csv: %d linhas"%len(out))
print(out.decisao.value_counts().to_string())
# precisao no ouro top-30
fol=C.ler_csv(BASE/"folha_verdade.csv"); fol["id"]=fol["id"].map(C.nid)
gmap=dict(zip(fol["id"],fol["tema_central_id"].map(C.nid)))
fpub=out[out.decisao=="publicar"].copy()
pub_ids=set(fpub.id.astype(str)); h=0; tot=0
for i,row in fpub.iterrows():
    if row.id in gmap:
        tot+=1; g=gmap[row.id]; p2=idx.get(str(row.tema_central_id))
        if g==p2: h+=1
if tot: print("[ENTREGA] no ouro: publ=%d acertos=%d prec=%.1f%%"%(tot,h,100*h/tot))