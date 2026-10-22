# inferir_v9.py - inferencia offline V9 congelada (sem LLM generativo)
import os
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artefatos_v9_final")
_P = None

def _zrow(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)

def _carregar():
    global _P
    if _P is None:
        P = joblib.load(os.path.join(ART, "pipeline_v9_congelado.joblib"))
        P["vec_ch"] = joblib.load(os.path.join(ART, "vec_ch.joblib"))
        _P = P
    return _P

def _codificar(textos, P):
    nome = P.get("encoder_name")
    if not nome:
        raise RuntimeError("Encoder de embeddings nao identificado/validado: passe embeddings pre-calculados "
                           "(L2-normalizados, mesma dimensao de Et) em classificar(textos, embeddings=...).")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError("sentence-transformers nao instalado: passe embeddings pre-calculados.")
    local = os.path.join(ART, "encoder_local")
    fonte = local if os.path.isdir(local) else nome
    enc = SentenceTransformer(fonte, local_files_only=True)
    enc.max_seq_length = int(P.get("encoder_max_seq_length", 512))
    return enc.encode(list(textos), batch_size=16, normalize_embeddings=True, show_progress_bar=False)

def classificar(textos, embeddings=None, ids=None):
    P = _carregar()
    if isinstance(textos, str):
        textos = [textos]
    textos = ["" if x is None else str(x) for x in textos]
    n = len(textos)
    ids = [str(i) for i in range(n)] if ids is None else [str(i) for i in ids]
    E = np.asarray(embeddings) if embeddings is not None else _codificar(textos, P)
    if E.shape != (n, P["Et"].shape[1]):
        raise ValueError("embeddings com shape %s, esperado %s" % (E.shape, (n, P["Et"].shape[1])))
    Zt = _zrow((P["tfidf"].transform(textos) @ P["Xt"].T).toarray())
    Ze = _zrow(E @ P["Et"].T)
    Zch = _zrow((P["vec_ch"].transform(textos) @ P["Xch_t"].T).toarray())
    S = P["w_tfidf"] * Zt + P["w_embed"] * Ze + P["w_char"] * Zch
    parent, depth, eq, Down1 = P["parent"], P["depth"], P["eq"], P["Down1"]
    Cm = eq[:, None] == eq[None, :]
    tau = P["tau"]
    lim = P["limiar"]
    lim = np.inf if lim is None else float(lim)
    srt = np.sort(S, axis=1)
    m2 = srt[:, -1] - srt[:, -2]
    rows = []
    for i in range(n):
        j = int(S[i].argmax()); guard = 0
        while parent[j] != -1 and depth[parent[j]] >= 2 and S[i, j] - S[i, parent[j]] < tau and guard < 50:
            j = int(parent[j]); guard += 1
        excl = Cm[j] | Down1[j] | Down1[:, j]
        outros = ~excl
        m1 = float(S[i, j] - S[i][outros].max()) if outros.any() else 99.0
        rows.append(dict(id=ids[i], tema_id=P["tema_ids"][j], caminho_previsto=P["caminhos"][j],
                         depth=int(depth[j]), m1=m1, m2=float(m2[i]),
                         status="PUBLICAR" if m1 >= lim else "REVISAR"))
    return pd.DataFrame(rows, columns=["id", "tema_id", "caminho_previsto", "depth", "m1", "m2", "status"])
