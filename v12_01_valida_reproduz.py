# -*- coding: utf-8 -*-
# v12_01_valida_reproduz.py
import os, sys, json, re
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
BASE = Path(__file__).resolve().parent
CACHE = BASE/"cache_offline"
sys.path.insert(0, str(BASE))

import v10_comum as C

def fail(m): print("[V12-01] FALHA: "+m); sys.exit(1)

# 1) entradas
q = C.ler_csv(CACHE/"questoes_limpas.csv"); q["id"]=q["id"].map(C.nid)
t = C.carregar_temas()  # 377
fol = C.ler_csv(BASE/"folha_verdade.csv"); fol["id"]=fol["id"].map(C.nid)
if len(q)!=659: fail("questoes=%d (esperado 659)"%len(q))
if len(t)!=377: fail("temas=%d (esperado 377)"%len(t))
fora = fol["fora_da_taxonomia"].map(C.nid)=="1" if "fora_da_taxonomia" in fol else pd.Series(False,index=fol.index)
idx = {x:i for i,x in enumerate(t["tema_id"])}
ok = (~fora) & fol["tema_central_id"].map(C.nid).isin(idx) & fol["id"].isin(set(q["id"]))
n = int(ok.sum())
if n!=92: fail("avaliaveis=%d (esperado 92)"%n)

E = np.load(CACHE/"emb_659.npy").astype(np.float32)
with open(CACHE/"emb_659_ids.json",encoding="utf-8") as f: ids_q = [C.nid(x) for x in json.load(f)]
if ids_q != q["id"].tolist(): fail("ordem emb_659 != questoes_limpas")
if E.shape != (659,1024): fail("shape emb_659 "+str(E.shape))

Et = np.load(CACHE/"emb_temas_v10.npy").astype(np.float32)
with open(CACHE/"emb_temas_v10_ids.json",encoding="utf-8") as f: ids_t = [C.nid(x) for x in json.load(f)]
if ids_t != t["tema_id"].tolist(): fail("ordem emb_temas_v10 != temas_limpos")
if Et.shape != (377,1024): fail("shape emb_temas_v10 "+str(Et.shape))
print("[V12-01] status=OK entradas 659/377/92")

# 2) score hibrido + regra (V0/M0/R1b T=0.5) - fiel ao v11
from sklearn.feature_extraction.text import TfidfVectorizer
W_EMB=0.6
TFIDF_KW=dict(ngram_range=(1,2),max_features=20000,sublinear_tf=True,strip_accents="unicode",lowercase=True)
SEP=" > "
q0 = q["texto_limpo"].fillna("").astype(str).tolist()
tf = TfidfVectorizer(**TFIDF_KW); tf.fit(t["texto"].tolist()+q0)
Stf = (tf.transform(q0) @ tf.transform(t["texto"].tolist()).T).toarray().astype(np.float32)
def zrow(S): return (S-S.mean(axis=1,keepdims=True))/(S.std(axis=1,keepdims=True)+1e-9)
S = W_EMB*zrow(E@Et.T) + (1.0-W_EMB)*zrow(Stf)

eq = np.asarray(C.grupos_equivalencia(t))
class EQI:
    def __init__(self,eq): self.eq=eq; self.mem={}
    def pai_dir(self,p,c,exc):
        ep,ec=int(eq[p]),int(eq[c])
        if (ep,ec) in exc: return exc[(ep,ec)]
        ok=False
        for a in np.where(eq==ep)[0]:
            for b in np.where(eq==ec)[0]:
                ca=t["caminho"][a].split(SEP); cb=t["caminho"][b].split(SEP)
                if len(cb)==len(ca)+1 and cb[:-1]==ca: ok=True; break
            if ok: break
        exc[(ep,ec)]=ok; return ok
# grupo unico por linha (ordem dos temas = indice)
rows = [q["id"].tolist().index(x) for x in fol[ok]["id"]]
golds = [idx[x] for x in fol[ok]["tema_central_id"].map(C.nid)]
def ordem(s,lim=5):
    out,v=[];vi=set()
    for j in np.argsort(-s,kind="stable"):
        g=int(eq[j]);
        if g in vi: continue
        vi.add(g); out.append(int(j))
        if len(out)>=lim: break
    return out
def aplicar(o,s,T):
    if T<=0 or len(o)<2: return o
    for k in (1,2):
        if k>=len(o): break
        if (s[o[0]]-s[o[k]])<T and EQI.pai_dir if False else False: pass
    return o
# caminho simples: usar a regra do v11 (membros do grupo, pai direto)
class EI:
    def __init__(self,eq): self.eq=eq; self.uniq={}
    def pai_dir(self,p,c):
        ep,ec=int(self.eq[p]),int(self.eq[c])
        if (ep,ec) in self.uniq: return self.uniq[(ep,ec)]
        ok=False
        pa=np.where(self.eq==ep)[0]; pc=np.where(self.eq==ec)[0]
        for a in pa:
            for b in pc:
                ca=t["caminho"][a].split(SEP); cb=t["caminho"][b].split(SEP)
                if len(cb)==len(ca)+1 and cb[:-1]==ca: ok=True; break
            if ok: break
        self.uniq[(ep,ec)]=ok; return ok
ei=EI(eq)
def aplica(o,s,T):
    if T<=0 or len(o)<2: return o
    kmax=min(2,len(o)-1)
    for k in range(1,kmax+1):
        if (s[o[0]]-s[o[k]])<T and ei.pai_dir(o[k],o[0]): return [o[k]]+o[:k]+o[k+1:]
    return o
def rel_oficial(g,p):  # acerto oficial: grupo igual ou pai/filho DIRETO (1 nivel)
    if int(eq[g])==int(eq[p]): return True
    pa=t["caminho"][g].split(SEP); pb=t["caminho"][p].split(SEP)
    la,lb=len(pa),len(pb)
    return (lb==la+1 and pb[:la]==pa) or (la==lb+1 and pa[:lb]==pb)
Sg=S[rows]
topm=[aplica(ordem(s),s,0.0) for s in Sg]   # sem regra
topr=[aplica(ordem(s),s,0.5) for s in Sg]   # R1b T=0.5
def rel_hier(g,p):
    if int(eq[g])==int(eq[p]): return True
    pa=t["caminho"][g].split(SEP); pb=t["caminho"][p].split(SEP)
    la,lb=len(pa),len(pb)
    return (lb>la and pb[:la]==pa) or (la>lb and pa[:lb]==pb)
base_h=100*sum(rel_hier(g,o[0]) for g,o in zip(golds,topm))/n
r1b_h=100*sum(rel_hier(g,o[0]) for g,o in zip(golds,topr))/n
base_o=100*sum(rel_oficial(g,o[0]) for g,o in zip(golds,topm))/n
r1b_o=100*sum(rel_oficial(g,o[0]) for g,o in zip(golds,topr))/n
print("[V12-01] hier: sem_regra=%.2f R1bT0.5=%.2f (esperado 61.96/65.22)"%(base_h,r1b_h))
print("[V12-01] oficial: sem_regra=%.2f R1bT0.5=%.2f"%(base_o,r1b_o))
if abs(base_h-61.96)>1.0 or abs(r1b_h-65.22)>1.0:
    fail("baseline nao reproduzido (sairam %.2f/%.2f)"%(base_h,r1b_h))
# extras
ex_col = fol[ok]["extras_ids"] if "extras_ids" in fol else pd.Series("",index=fol[ok].index)
exr=[]
for v in ex_col:
    exr.append({idx[x] for x in re.split(r"[;,|\s]+",str(v)) if x and C.nid(x) in idx})
def rel_oficial_ex(g,p,ex): return rel_oficial(g,p) or any(int(eq[g])==int(eq[e]) for e in ex)
r1b_oe=100*sum(rel_oficial_ex(g,o[0],ex) for g,o,ex in zip(golds,topr,exr))/n
print("[V12-01] oficial+extras R1bT0.5=%.2f"%(r1b_oe))
json.dump(dict(baseline_hier=base_h,r1b_hier=r1b_h,baseline_oficial=base_o,r1b_oficial=r1b_o,r1b_oficial_extras=r1b_oe,n=n),
          open(BASE/"v12_baseline.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("[V12-01] OK saved v12_baseline.json")