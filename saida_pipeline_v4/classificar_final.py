# classificar_final.py - wrapper de inferir_v9 com tolerancia numerica no limiar
from inferir_v9 import classificar as _classificar_base, _carregar, _codificar

# Ruido numerico CPU vs GPU (~6e-6) pode jogar uma questao na borda do limiar para REVISAR.
# O gap seguinte ao limiar e ~0.08, entao 1e-4 nao altera nenhuma outra decisao.
EPS_M1 = 1e-4

__all__ = ["classificar", "_carregar", "_codificar", "EPS_M1"]


def classificar(textos, embeddings=None, ids=None):
    df = _classificar_base(textos, embeddings=embeddings, ids=ids)
    lim = _carregar()["limiar"]
    lim = float("inf") if lim is None else float(lim)
    df["status"] = ["PUBLICAR" if float(m) >= lim - EPS_M1 else "REVISAR" for m in df["m1"]]
    return df