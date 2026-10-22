import sys, pandas as pd
from pathlib import Path
B = Path(sys.argv[1]); sys.path.insert(0, str(B / "saida_pipeline_v4"))
from inferir_v9 import _carregar
lim = float(_carregar()["limiar"])
d = pd.read_csv(B / "saida_regressao_v9.csv", encoding="utf-8-sig")
d["delta"] = d["m1"] - lim
print("limiar =", repr(lim))
print(d.reindex(d["delta"].abs().sort_values().index).head(6)[["id","m1","delta","status","aceito_A7"]].to_string(index=False))
