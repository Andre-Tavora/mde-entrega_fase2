import os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
cache = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
if not cache.exists():
    print("SEM CACHE LOCAL - NADA PARA TESTAR")
    sys.exit(0)

candidatos = []
for p in cache.iterdir():
    if p.is_dir() and p.name.startswith("models--"):
        nome = p.name.replace("models--", "").replace("--", "/")
        snaps = list((p / "snapshots").glob("*"))
        if not snaps:
            continue
        sn = max(snaps, key=lambda x: x.stat().st_mtime)
        refs = p / "refs" / "main"
        if refs.exists():
            alvo = (refs.read_text().strip())
            if (p / "snapshots" / alvo).exists():
                sn = p / "snapshots" / alvo
        candidatos.append((nome, sn))

if not candidatos:
    print("CACHE EXISTE MAS SEM MODELOS VÁLIDOS")
    sys.exit(0)

print(f"CANDIDATOS: {len(candidatos)}")
for nome, sn in candidatos:
    print(f"  {nome} -> {sn}")

# Questões da folha (92)
q = pd.read_csv(BASE / "cache_offline" / "questoes_limpas.csv", encoding="utf-8-sig", dtype=str)
g = pd.read_csv(BASE / "folha_verdade.csv", encoding="utf-8-sig", dtype=str)
def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s
q["id"] = q["id"].map(nid)
g["id"] = g["id"].map(nid)
g = g[g["fora_da_taxonomia"].map(nid) != "1"]
df = q.merge(g[["id", "tema_central_id"]], on="id", how="inner").reset_index(drop=True)
tema_ids_esp = set(g["tema_central_id"].map(nid))

# Para teste rápido das 92: vamos verificar apenas que o encoder carrega e o shape das embeddings
testes = [nome for nome, _ in candidatos][:5]
for nome, sn in candidatos[:5]:
    try:
        print(f"\n=== TESTE {nome} ===")
        model = SentenceTransformer(str(sn), local_files_only=True)
        model.max_seq_length = 512
        E = model.encode(df["texto_limpo"].head(8).tolist(), normalize_embeddings=True, batch_size=8)
        print(f"  OK shape={E.shape}")
    except Exception as e:
        print(f"  FALHOU: {type(e).__name__}: {str(e)[:150]}")