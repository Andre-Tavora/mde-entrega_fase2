# gera_emb_temas_v10.py - embeddings dos temas (descricao + palavras_chave), L2, offline
import json
import numpy as np
import v10_comum as C

t = C.carregar_temas()
n = len(t)
print("[DADOS] temas =", n)
if n != 377:
    print("[AVISO] esperado 377 temas")

model = C.carregar_encoder()
E = C.codificar(model, t["texto"].tolist())
np.save(C.P_EMB_TEMAS, E)
with open(C.P_EMB_TEMAS_IDS, "w", encoding="utf-8") as f:
    json.dump(t["tema_id"].tolist(), f, ensure_ascii=False)

nm = np.linalg.norm(E, axis=1)
print("[EMB] shape =", E.shape, "| norma min/max = %.6f / %.6f" % (nm.min(), nm.max()))
print("[SAIDA]", C.P_EMB_TEMAS)
print("[SAIDA]", C.P_EMB_TEMAS_IDS)
print("OK n=%d" % n)