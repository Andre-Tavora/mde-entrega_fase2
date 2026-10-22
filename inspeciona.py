import sys, json, joblib, pandas as pd
b = sys.argv[1]
a = b + r"\saida_pipeline_v4\artefatos_v9_final"
c = b + r"\cache_offline"
print("== config.json =="); print(open(a + r"\config.json", encoding="utf-8").read()[:1500])
for n in ["pipeline_v9_congelado.joblib", "vec_ch.joblib"]:
    o = joblib.load(a + "\\" + n)
    print("==", n, "==", type(o).__name__)
    if isinstance(o, dict):
        for k, v in o.items():
            print("  ", k, "->", type(v).__name__, getattr(v, "shape", ""))
for n in ["questoes_limpas.csv", "temas_limpos.csv"]:
    d = pd.read_csv(c + "\\" + n, encoding="utf-8")
    print("==", n, d.shape, "=="); print(list(d.columns)); print(d.head(2).to_string()[:700])
r = json.load(open(c + r"\regras_leis.json", encoding="utf-8"))
print("== regras_leis.json ==", type(r).__name__, len(r)); print(str(r)[:500])
f = pd.read_csv(b + r"\folha_verdade.csv", encoding="utf-8")
print("== folha_verdade ==", f.shape, "| fora_da_taxonomia:", int(f.fora_da_taxonomia.sum()))
