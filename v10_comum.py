# v10_comum.py - utilitarios compartilhados (offline, sem LLM)
import os
import re
import sys
import json
import difflib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache_offline"
ENC_LOCAL = BASE / "saida_pipeline_v4" / "artefatos_v9_final" / "encoder_local"
ENC_NOME = "BAAI/bge-m3"          # mesmo modelo do V9
ENC_MAXLEN = 512
GRAFIA_MIN = 0.95                 # similaridade minima de nomes para tratar como variante de grafia

P_TEMAS = CACHE / "temas_limpos.csv"
P_DESC = CACHE / "descricoes_consolidado.json"
P_QUEST = CACHE / "questoes_limpas.csv"
P_EMB_TEMAS = CACHE / "emb_temas_v10.npy"
P_EMB_TEMAS_IDS = CACHE / "emb_temas_v10_ids.json"
P_EMB_Q = CACHE / "emb_659.npy"
P_EMB_Q_IDS = CACHE / "emb_659_ids.json"

if ENC_LOCAL.is_dir():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")


def norm_txt(s):
    s = sem_acento(s).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def nome_base(s):
    s = sem_acento(s).lower()
    s = re.sub(r"\(\s*parte\s*\d+\s*\)", " ", s)
    return norm_txt(s)


def ler_csv(p):
    ultimo = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc, dtype=str, keep_default_na=False)
            df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
            return df
        except UnicodeDecodeError as e:
            ultimo = e
    raise ultimo


def carregar_temas():
    """Ordem = temas_limpos.csv. texto = descricao + ' ' + palavras_chave (SEM nao_confundir)."""
    for p in (P_TEMAS, P_DESC):
        if not p.exists():
            raise SystemExit("arquivo ausente: " + str(p))
    t = ler_csv(P_TEMAS)
    for c in ("tema_id", "caminho"):
        if c not in t.columns:
            raise SystemExit("coluna ausente em temas_limpos.csv: " + c)
    t = t[["tema_id", "caminho"]].copy()
    t["tema_id"] = t["tema_id"].map(nid)
    if t["tema_id"].duplicated().any():
        raise SystemExit("tema_id duplicado em temas_limpos.csv")
    t["caminho"] = t["caminho"].map(lambda x: " > ".join(p.strip() for p in str(x).split(">") if p.strip()))
    t["nome"] = t["caminho"].map(lambda x: x.split(" > ")[-1] if x else "")

    with open(P_DESC, encoding="utf-8-sig") as f:
        dados = json.load(f)
    dm = {nid(o["tema_id"]): o for o in dados}
    faltam = [x for x in t["tema_id"] if x not in dm]
    if faltam:
        raise SystemExit("temas sem descricao no consolidado: " + ", ".join(faltam[:20]))

    def pk(o):
        v = o.get("palavras_chave", "")
        return "; ".join(map(str, v)) if isinstance(v, list) else str(v or "")

    t["descricao"] = [str(dm[x].get("descricao", "") or "").strip() for x in t["tema_id"]]
    t["palavras_chave"] = [pk(dm[x]).strip() for x in t["tema_id"]]
    t["nao_confundir"] = [str(dm[x].get("nao_confundir", "") or "").strip() for x in t["tema_id"]]
    t["texto"] = [(d + " " + k).strip() for d, k in zip(t["descricao"], t["palavras_chave"])]
    return t.reset_index(drop=True)


def grupos_equivalencia(t):
    """Classes de equivalencia: ultimo nivel igual (sem '(Parte N)'), JEC, LEP e variantes de grafia."""
    m = len(t)
    uf = list(range(m))

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    nomes = [nome_base(n) for n in t["nome"]]
    by = {}
    for j, k in enumerate(nomes):
        if k:
            by.setdefault(k, []).append(j)
    for L in by.values():
        for j in L[1:]:
            union(L[0], j)
    jec = [j for j, k in enumerate(nomes) if "juizado" in k and "crimin" in k]
    for j in jec[1:]:
        union(jec[0], j)
    lep = [j for j, k in enumerate(nomes) if k == "lei de execucao penal"]
    for j in lep[1:]:
        union(lep[0], j)
    chaves = sorted(by)
    for a in range(len(chaves)):
        for b in range(a + 1, len(chaves)):
            x, y = chaves[a], chaves[b]
            if min(len(x), len(y)) >= 8 and abs(len(x) - len(y)) <= 3:
                if difflib.SequenceMatcher(None, x, y).ratio() >= GRAFIA_MIN:
                    union(by[x][0], by[y][0])
    return pd.factorize(pd.Series([find(j) for j in range(m)]))[0]


def carregar_encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if ENC_LOCAL.is_dir():
        fonte = str(ENC_LOCAL)
    else:
        fonte = ENC_NOME
        print("[AVISO] encoder_local ausente; usando " + ENC_NOME + " (exige cache HF local)")
    print("[ENCODER] " + fonte + " | device=" + dev)
    model = SentenceTransformer(fonte, device=dev)
    model.max_seq_length = ENC_MAXLEN
    return model


def codificar(model, textos, batch=16):
    E = model.encode(list(textos), batch_size=batch, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    E = np.asarray(E, dtype=np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)