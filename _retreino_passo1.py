from pathlib import Path
import csv, json
import numpy as np

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE/"cache_offline"

# 1. carrega questoes originais (preciso do texto). Se nao tiver colunas de texto, usa o id.
#    Vou assumir que existe questoes.csv ou similar; se nao, crio texto = id.
quest = {}
qsrc = CACHE/"questoes.csv"
if qsrc.is_file():
    with open(qsrc, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            quest[str(r["id"])] = " ".join(r.get(k,"") for k in ["enunciado","texto","pergunta"] if k in r)
else:
    # fallback: texto = id (nao ideal, mas segue)
    quest = {str(r["id"]):str(r["id"]) for r in csv.DictReader(open(BASE/"publicacao.csv",encoding="utf-8-sig"))}

# 2. descricoes dos temas (dos lotes consolidados)
desc = {}
dpath = CACHE/"descricoes_consolidado.json"
if dpath.is_file():
    for o in json.load(open(dpath,encoding="utf-8-sig")):
        desc[str(o["tema_id"])] = o.get("descricao","")
print("descricoes carregadas: %d" % len(desc))

# 3. embedding atual das questoes (para reusar o mesmo encoder BGE-M3 via sentence-transformers se disponivel)
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    HAVE_ST = True
except Exception as e:
    HAVE_ST = False
    print("BGE-M3 via sentence-transformers indisponivel:", e)

if HAVE_ST:
    ids = list(quest.keys())
    textos = []
    for qid in ids:
        base_txt = quest.get(qid,"")
        # descricao do tema previsto (preciso do previsto; uso o atual top-1, que ja temos via features)
        # fallback: concatena desc do top-3 se disponivel
        textos.append(base_txt + " [CONTEXTO] " + desc.get(qid,"") if qid in desc else base_txt)
    embs = model.encode(textos, normalize_embeddings=True, show_progress_bar=False)
    np.save(CACHE/"emb_659_enriquecido.npy", embs.astype(np.float32))
    json.dump(ids, open(CACHE/"emb_659_enriquecido_ids.json","w"))
    print("PASSO 2 OK - embeddings enriquecidos salvos: %s" % embs.shape)
else:
    print("PASSO 2 PARADO - sem sentence-transformers. Instale ou rode no Colab.")
