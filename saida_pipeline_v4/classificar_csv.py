#!/usr/bin/env python
"""Classifica textos de um CSV (colunas id, texto) com o pipeline V9 congelado."""
import argparse
import os
import sys

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
COLS = ["id", "tema_id", "caminho_previsto", "depth", "m1", "m2", "status"]


def main():
    ap = argparse.ArgumentParser(description="Classificacao V9 (offline, sem LLM generativo)")
    ap.add_argument("--entrada", required=True, help="CSV com colunas id e texto (utf-8-sig)")
    ap.add_argument("--saida", required=True, help="CSV de saida")
    a = ap.parse_args()

    df = pd.read_csv(a.entrada, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    faltam = [c for c in ("id", "texto") if c not in df.columns]
    if faltam:
        sys.exit("ERRO: colunas ausentes na entrada: " + ", ".join(faltam) + " (esperado: id, texto)")

    ids = df["id"].astype(str).str.strip().tolist()
    textos = df["texto"].astype(str).tolist()
    vazio = [not t.strip() for t in textos]
    idx_ok = [i for i, v in enumerate(vazio) if not v]

    res = None
    if idx_ok:
        from inferir_v9 import classificar
        res = classificar([textos[i] for i in idx_ok], ids=[ids[i] for i in idx_ok]).reset_index(drop=True)

    linhas, k = [], 0
    for i in range(len(df)):
        if vazio[i]:
            linhas.append({"id": ids[i], "tema_id": "", "caminho_previsto": "", "depth": "",
                           "m1": "", "m2": "", "status": "REVISAR"})
        else:
            r = res.iloc[k]
            k += 1
            lin = {c: r[c] for c in COLS}
            lin["id"] = ids[i]
            linhas.append(lin)

    pd.DataFrame(linhas, columns=COLS).to_csv(a.saida, index=False, encoding="utf-8-sig")
    print(f"linhas={len(df)} | classificadas={len(idx_ok)} | vazias(REVISAR)={len(df) - len(idx_ok)} | saida={a.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
