import pandas as pd
from pathlib import Path
B = Path(__file__).resolve().parent
f = pd.read_csv(B / "folha_verdade.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
r = pd.read_csv(B / "saida_regressao_v9.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
print("folha_verdade:", f.shape, list(f.columns))
print(f.head(4).to_string(index=False))
print("\nregressao aceito_*: valores por coluna")
for c in [c for c in r.columns if c.startswith("aceito_")] + ["tema_central_id", "extras_ids"]:
    print(c, r[c].value_counts().head(4).to_dict())
print("\nids em comum folha x regressao:", len(set(f.iloc[:, 0].str.replace(r"\.0$", "", regex=True)) & set(r["id"].str.replace(r"\.0$", "", regex=True))))
