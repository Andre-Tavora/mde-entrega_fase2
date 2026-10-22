# -*- coding: utf-8 -*-
# v12_regenera.py - regenera publicacao.csv do zero, offline, sem LLM
import subprocess, sys, json, hashlib, time, os
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
t0=time.time()
for f in ("v12_01_valida_reproduz.py","v12_02_calibra.py","v12_03_gera_entrega.py"):
    r=subprocess.run([sys.executable,str(BASE/f)],cwd=str(BASE))
    if r.returncode: sys.exit(r.returncode)
entradas=["cache_offline/questoes_limpas.csv","cache_offline/temas_limpos.csv","cache_offline/descricoes_consolidado.json",
          "cache_offline/emb_659.npy","cache_offline/emb_659_ids.json","cache_offline/emb_temas_v10.npy","cache_offline/emb_temas_v10_ids.json",
          "folha_verdade.csv","v10_comum.py"]
mani={}
for p in entradas:
    mani[p]=hashlib.sha256((BASE/p).read_bytes()).hexdigest()
for p in ("publicacao.csv","previsao.txt"):
    if (BASE/p).exists(): mani[p]=hashlib.sha256((BASE/p).read_bytes()).hexdigest()
json.dump(mani,open(BASE/"MANIFEST_v12.json","w",encoding="utf-8"),indent=2,ensure_ascii=False)
print("[V12] OK regenerado em %.1fs | manifest=%s"%(time.time()-t0,BASE/"MANIFEST_v12.json"))