import numpy as np, json, re
from pathlib import Path
p = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2\cache_offline")
for f in ["emb_temas_v10.npy","emb_temas_v10b.npy","emb_folha.npy"]:
    a = np.load(str(p/f), allow_pickle=True)
    print(f, a.shape, a.dtype)
print("--- bumps id compatibility ---")
ids_t = json.loads((p/"emb_temas_v10_ids.json").read_text(encoding="utf-8"))
ids_f = json.loads((p/"emb_folha_ids.json").read_text(encoding="utf-8"))
print("temas ids[0..2]:", [str(x) for x in ids_t[:3]])
print("folha ids[0..2]:", [str(x) for x in ids_f[:3]])
