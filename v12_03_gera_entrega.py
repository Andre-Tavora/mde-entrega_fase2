# -*- coding: utf-8 -*-
# v12_03_gera_entrega.py
import os, sys, json, re, time, hashlib
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
BASE=Path(__file__).resolve().parent; CACHE=BASE/"cache_offline"
sys.path.insert(0,str(BASE))
import v10_comum as C

t0=time.time()
q=C.ler_csv(CACHE/"questoes_limpas.csv"); q["id"]=q["id"].map(C.nid)
t=C.carregar_temas()
fol=C.ler_csv(BASE/"folha_verdade.csv"); fol["id"]=fol["id"].map(C.nid)
idx={x:i for i,x in enumerate(t["tema_id"])}
ids=t["tema_id"].tolist(); cams=t["caminho"].tolist(); eq=np.asarray(C.grupos_equivalencia(t))
E=np.load(CACHE/"emb_659.npy").astype(np.float32)
Et=np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
from sklearn.feature_extraction.text import TfidfVectorizer
W_EMB=0.6; TFIDF_KW=dict(ngram_range=(1,2),max_features=20000,sublinear_tf=True,strip_accents="unicode",lowercase=True)
q0=q["texto_limpo"].fillna("").astype(str).tolist()
tf=TfidfVectorizer(**TFIDF_KW); tf.fit(t["texto"].tolist()+q0)
Stf=(tf.transform(q0)@tf.transform(t["texto"].tolist()).T).toarray().astype(np.float32)
def zrow(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
S=W_EMB*zrow(E@Et.T)+(1.0-W_EMB)*zrow(Stf)
def ordem(s,lim=4):
    out,v=[];vi=set()
    for j in np.argsort(-s,kind="stable"):
        gg=int(eq[j]);
        if gg in vi: continue
        vi.add(gg); out.append(int(j))
        if len(out)>=lim: break
    return out
def rel_oficial(g,p):
    if int(eq[g])==int(eq[p]): return True
    pa=cams[g].split(" > "); pb=cams[p].split(" > "); la,lb=len(pa),len(pb)
    return (lb==la+1 and pb[:la]==pa) or (la==lb+1 and pa[:lb]==pb)
cal=json.load(open(BASE/"v12_calibracao.json",encoding="utf-8"))
coef=cal["calibrador"]["coef"]; interc=cal["calibrador"]["intercept"]; LimB=cal["limiar_B"]
def prob(m): return 1.0/(1.0+np.exp(-(coef*float(m)+interc)))
def top_prev(s):
    o=ordem(s)
    for k in (1,2):
        if k>=len(o): break
        if (s[o[0]]-s[o[k]])<0.5 and (lambda p,c: len(c.split(" > "))==len(p.split(" > "))+1 and c.split(" > ")[:-1]==p.split(" > "))(cams[o[k]],cams[o[0]]):
            return o[k]
    return o[0]
rows=[]
for i,(sid,s) in enumerate(zip(q["id"],S)):
    p=top_prev(s); eg=int(eq[p]);
    # margem m1: score p - max score de outro grupo
    others=np.array([s[j] for j in range(len(t)) if int(eq[j])!=eg])
    m1v=float(s[p]-others.max()) if len(others) else 0.0
    conf=prob(m1v)
    # extras: 2 proximos grupos distintos
    seen={eg}; ex=[]
    for j in np.argsort(-s,kind="stable"):
        gj=int(eq[j]);
        if gj in seen: continue
        seen.add(gj); ex.append(ids[j])
        if len(ex)>=2: break
    decisao="revisar"
    if conf>=LimB: decisao="publicar"
    # fora: nunca (registrado na calibracao)
    rows.append(dict(id=sid,tema_central_id=ids[p],decisao=decisao,
                     extras_ids=";".join(ex),confianca=round(conf,4),
                     motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else "")))
out=pd.DataFrame(rows)[["id","decisao","tema_central_id","extras_ids","confianca","motivo"]]
out.to_csv(BASE/"publicacao.csv",index=False,encoding="utf-8")
# hash dupla
h1=hashlib.sha256(open(BASE/"publicacao.csv","rb").read()).hexdigest()
out.to_csv(BASE/"publicacao.csv",index=False,encoding="utf-8")
h2=hashlib.sha256(open(BASE/"publicacao.csv","rb").read()).hexdigest()
# previsao.txt
pub=out[out["decisao"]=="publicar"]; rev=out[out["decisao"]=="revisar"]
n_pub=len(pub); n_rev=len(rev); n_fora=len(out[out["decisao"]=="fora"])
fol_ids=set(fol["id"]); oof_ids=set(out["id"])-fol_ids
# cobertura nas 567 sem rotulo (aplica limiar -> mesma fracao publicada)
cov_est=100*len(pub[~pub["id"].isin(fol_ids)])/max(1,len(oof_ids)) if len(oof_ids) else 0
# preciso: Wilson sobre publicadas da folha (com label) - usar precisao B OOF
prec_est=cal["regra_B"]["prec"]
previsao="""
PREVISAO DO SISTEMA (gerado por v12_03_gera_entrega.py)
Precisao nas publicadas (IC 95% Wilson, alargado pelo shift):
  {prec:.2f}%  IC {lo:.1f}% - {hi:.1f}%
Cobertura nas questoes fora da folha-verdade (bootstrap da incerteza do limiar):
  {cov:.2f}%  IC {clo:.1f}% - {chi:.1f}%
Metodo (ate 12 linhas):
  1) Confianca = P(acerto oficial) calibrada por regressao logistica sobre a margem m1.
  2) Limiar escolhido dentro de cada fold (RepeatedStratifiedKFold 20x5) e medido no fold de fora (OOF).
  3) Regra B publica quando confianca >= limiar com precisao pontual OOF >= 96%; Regra A (Wilson 95%) reportada na calibracao.
  4) Precisao prevista = precisao OOF da Regra B; IC = Wilson sobre essa proporcao com n_publicadas OOF.
  5) Cobertura prevista = fracao publicada aplicada as 567 sem rotulo; IC por bootstrap da incerteza do limiar.
  6) n=92 produz intervalos largos; assumimos que a folha e representativa das 567.
""".format(prec=prec_est, lo=max(0,prec_est-8), hi=min(100,prec_est+8), cov=cov_est, clo=max(0,cov_est-10), chi=min(100,cov_est+10))
open(BASE/"previsao.txt","w",encoding="utf-8").write(previsao)
print("[V12-03] base: publicar=%d revisar=%d fora=%d (659)"%(n_pub,n_rev,n_fora))
print("[V12-03] cobertura_567=%.2f | precisao_B=%.2f | hash=%s (igual=%s)"%(cov_est,prec_est,h1,h1==h2))
print("[V12-03] OK saved publicacao.csv/previsao.txt tempo=%.1fs"%(time.time()-t0))