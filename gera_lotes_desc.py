import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "saida_pipeline_v4"))
from inferir_v9 import _carregar

SEP = " > "
TAM_LOTE = 25
MAX_LISTA = 15
N_ESPERADO = 377
P_TEMAS = BASE / "cache_offline" / "temas_limpos.csv"
DIR_OUT = BASE / "cache_offline" / "lotes_desc"
COLS = ["tema_id", "caminho", "depth", "pai_caminho", "irmaos", "filhos", "equivalentes"]


def nid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def ncam(s):
    return SEP.join([x.strip() for x in str(s).split(">") if x.strip()])


def falhar(msg):
    print("LOTES FALHOU: " + msg)
    sys.exit(1)


# ---------- 1. leitura e validacao ----------
if not P_TEMAS.exists():
    falhar("arquivo ausente: " + str(P_TEMAS))
t = pd.read_csv(P_TEMAS, encoding="utf-8-sig", dtype=str, keep_default_na=False)
t.columns = [c.strip().lstrip("\ufeff") for c in t.columns]
for c in ("tema_id", "caminho"):
    if c not in t.columns:
        falhar("coluna ausente em temas_limpos.csv: " + c)
t["tema_id"] = t["tema_id"].map(nid)

P = _carregar()
ids = [nid(x) for x in P["tema_ids"]]
cams = [ncam(x) for x in P["caminhos"]]
parent = [int(x) for x in P["parent"]]
depth = [int(x) for x in P["depth"]]
eq = [int(x) for x in P["eq"]]
m = len(ids)

set_csv = set(t["tema_id"])
set_p = set(ids)
if set_csv != set_p:
    print("tema_id no CSV e ausentes em P (%d): %s" % (len(set_csv - set_p), sorted(set_csv - set_p)[:30]))
    print("tema_id em P e ausentes no CSV (%d): %s" % (len(set_p - set_csv), sorted(set_p - set_csv)[:30]))
    falhar("conjuntos de tema_id diferem")
if len(set_p) != m:
    falhar("tema_id duplicado em P")
if m != N_ESPERADO or len(t) != N_ESPERADO:
    falhar("esperado %d nos; P=%d, CSV=%d" % (N_ESPERADO, m, len(t)))
print("[DADOS] nos = %d | conjuntos de tema_id identicos" % m)

csv_cam = dict(zip(t["tema_id"], t["caminho"].map(ncam)))
dif_cam = sum(1 for i in range(m) if csv_cam[ids[i]] != cams[i])
if dif_cam:
    print("[AVISO] %d nos com caminho diferente entre CSV e P (usando P)" % dif_cam)

# ---------- 2. estrutura ----------
filhos_de = {}
for j in range(m):
    filhos_de.setdefault(parent[j], []).append(j)
grupo_eq = {}
for j in range(m):
    grupo_eq.setdefault(eq[j], []).append(j)


def ultimo(j):
    p = cams[j].split(SEP)
    return p[-1] if p else ""


def lista(js):
    nomes = [ultimo(j) for j in sorted(js, key=lambda k: cams[k])]
    return " | ".join(nomes[:MAX_LISTA])


rows = []
for j in range(m):
    partes = cams[j].split(SEP)
    pai_cam = SEP.join(partes[:-1]) if len(partes) > 1 else ""
    irm = [k for k in filhos_de.get(parent[j], []) if k != j]
    fil = filhos_de.get(j, [])
    equiv = [k for k in grupo_eq[eq[j]] if k != j]
    rows.append(dict(
        tema_id=ids[j],
        caminho=cams[j],
        depth=depth[j],
        pai_caminho=pai_cam,
        irmaos=lista(irm),
        filhos=lista(fil),
        equivalentes=" | ".join(cams[k] for k in sorted(equiv, key=lambda k: cams[k])),
    ))

df = pd.DataFrame(rows, columns=COLS).sort_values(["caminho", "tema_id"], kind="stable").reset_index(drop=True)

# ---------- 3. lotes ----------
DIR_OUT.mkdir(parents=True, exist_ok=True)
nlotes = (len(df) + TAM_LOTE - 1) // TAM_LOTE
tamanhos = []
total = 0
for b in range(nlotes):
    blk = df.iloc[b * TAM_LOTE:(b + 1) * TAM_LOTE]
    p = DIR_OUT / ("lote_%02d.csv" % (b + 1))
    blk.to_csv(p, index=False, encoding="utf-8-sig")
    tamanhos.append(len(blk))
    total += len(blk)

# ---------- 4. resumo ----------
print("[SAIDA] pasta: " + str(DIR_OUT))
print("lotes: %d" % nlotes)
print("nos por lote: " + ", ".join("lote_%02d=%d" % (i + 1, n) for i, n in enumerate(tamanhos)))
print("total gravado: %d" % total)
if total != m:
    falhar("total gravado (%d) difere de %d" % (total, m))
print("\nLOTES OK")