import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1], device="cpu")
v = m.encode(["teste de carga offline"], normalize_embeddings=True)
print("OK | dim:", v.shape[1], "| norma:", round(float((v**2).sum()**0.5), 4))
