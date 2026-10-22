# -*- coding: utf-8 -*-
# Frente A v2 — pred REAL do publicacao.csv x gold da folha (Q-prefixado)
import pandas as pd
BASE = r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2"
pub = pd.read_csv(BASE+"\\publicacao.csv", dtype=str)
fol = pd.read_csv(BASE+"\\folha_verdade.csv", dtype=str)
ids_alvo = ["Q0058","Q0077","Q0208","Q0279","Q0316","Q0319"]
pub_alvo = pub[pub.id.isin(ids_alvo)][["id","decisao","tema_central_id","confianca","motivo"]]
gold_map = fol.set_index("id").to_dict("index")
print("="*72)
print("FRENTE A v2 — pred(publicacao) x gold(folha)  [ids Q-prefixados]")
print("="*72)
for _,r in pub_alvo.iterrows():
    g = gold_map.get(r.id, {})
    print(f"\nID {r.id}  | decisao={r.decisao}  | m1/conf={r.confianca}  | motivo={r.motivo}")
    print(f"  PRED: {r.tema_central_id}")
    print(f"  GOLD: {g.get('tema_central_id','NA')}  | caminho: {g.get('tema_central_caminho','?')}")
    print(f"  extras_gold: {g.get('extras_ids','')}  | fora: {g.get('fora_da_taxonomia','')}")
print("\nFIM")