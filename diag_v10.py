# diag_v10.py - diagnostico (ablacao, hierarquia, desempate, rank, variante B). Zero LLM.
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import v10_comum as C

# ===================== CONFIG (espelha avalia_v10.py) =====================
W_EMB_PADRAO = 0.6
MARGEM = 0.30
KW_MIN_LEN = 4
NOME_MIN_LEN = 6
TOPK = 3
SEP = " > "
TFIDF_KW = dict(ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
                strip_accents="unicode", lowercase=True)
P_FOL = C.BASE / "folha_verdade.csv"
P_ABL = C.BASE / "ablacao_v10.csv"
P_ERR = C.BASE / "diag_erros_v10.csv"
P_EMB_B = C.CACHE / "emb_temas_v10b.npy"
AVISO = "AVISO: n=92 -> margem de erro ~+-9 pp; so considere diferencas > 5 pp."
# ==========================================================================

# ---------- dados ----------
t = C.carregar_temas()
m = len(t)
ids = t["tema_id"].tolist()
idx = {x: i for i, x in enumerate(ids)}
if not (C.P_EMB_TEMAS.exists() and C.P_EMB_TEMAS_IDS.exists()):
    raise SystemExit("rode antes: gera_emb_temas_v10.py")
Et = np.load(C.P_EMB_TEMAS).astype(np.float32)
with open(C.P_EMB_TEMAS_IDS, encoding="utf-8") as f:
    if json.load(f) != ids:
        raise SystemExit("ordem dos IDs em emb_temas_v10_ids.json difere de temas_limpos.csv")
if Et.shape[0] != m:
    raise SystemExit("emb_temas_v10.npy com shape incompativel")

eq = C.grupos_equivalencia(t)
membros = {}
for j in range(m):
    membros.setdefault(int(eq[j]), []).append(j)
partes = [[p for p in c.split(SEP) if p] for c in t["caminho"]]
cams = t["caminho"].tolist()
nomes_t = t["nome"].tolist()

q = C.ler_csv(C.P_QUEST)
q["id"] = q["id"].map(C.nid)
q = q.drop_duplicates("id").reset_index(drop=True)
qtexts = q["texto_limpo"].fillna("").astype(str).tolist()
qpos = {x: i for i, x in enumerate(q["id"])}
QN = [" " + C.norm_txt(x) + " " for x in qtexts]

_enc = {"model": None}


def get_encoder():
    if _enc["model"] is None:
        _enc["model"] = C.carregar_encoder()
    return _enc["model"]


def emb_questoes():
    qids = q["id"].tolist()
    if C.P_EMB_Q.exists() and C.P_EMB_Q_IDS.exists():
        with open(C.P_EMB_Q_IDS, encoding="utf-8") as f:
            ic = [C.nid(x) for x in json.load(f)]
        E = np.load(C.P_EMB_Q)
        if ic == qids and E.shape == (len(qids), Et.shape[1]):
            print("[CACHE] embeddings das questoes reutilizados", E.shape)
            return E.astype(np.float32)
    print("[EMB] gerando embeddings das questoes (cache ausente/incompativel)")
    E = C.codificar(get_encoder(), qtexts)
    np.save(C.P_EMB_Q, E)
    with open(C.P_EMB_Q_IDS, "w", encoding="utf-8") as f:
        json.dump(qids, f)
    return E


Eq = emb_questoes()

# ---------- gabarito ----------
if not P_FOL.exists():
    raise SystemExit("ausente: " + str(P_FOL))
fol = C.ler_csv(P_FOL)
for c in ("id", "tema_central_id"):
    if c not in fol.columns:
        raise SystemExit("coluna ausente em folha_verdade.csv: " + c)
fol["id"] = fol["id"].map(C.nid)
if "fora_da_taxonomia" in fol.columns:
    fora = fol["fora_da_taxonomia"].map(C.nid) == "1"
else:
    fora = pd.Series(False, index=fol.index)
fol["central"] = fol["tema_central_id"].map(C.nid)
n0 = len(fol)
ok = (~fora) & fol["central"].isin(idx) & fol["id"].isin(qpos)
fol = fol[ok].drop_duplicates("id").reset_index(drop=True)
exs = fol["extras_ids"] if "extras_ids" in fol.columns else pd.Series("", index=fol.index)
extras = [{idx[C.nid(x)] for x in re.split(r"[;,|\s]+", str(v)) if x and C.nid(x) in idx} for v in exs]
golds = [idx[x] for x in fol["central"]]
gids = fol["id"].tolist()
rows = [qpos[x] for x in gids]
n = len(golds)
print("[GABARITO] linhas=%d | avaliadas=%d" % (n0, n))
if n == 0:
    raise SystemExit("nenhuma questao avaliavel")

# ---------- scores base ----------
def zrow(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


def tfidf_scores(textos_tema):
    tf = TfidfVectorizer(**TFIDF_KW)
    tf.fit(list(textos_tema) + qtexts)
    Xt = tf.transform(list(textos_tema))
    Xq = tf.transform(qtexts)
    return (Xq[rows] @ Xt.T).toarray()


Semb = Eq[rows] @ Et.T
Stf = tfidf_scores(t["texto"].tolist())
Zemb, Ztf = zrow(Semb), zrow(Stf)


def hibrido(Se, St, w, z=True):
    if z:
        Se, St = zrow(Se), zrow(St)
    return w * Se + (1.0 - w) * St


S0 = hibrido(Semb, Stf, W_EMB_PADRAO, True)

# ---------- utilitarios ----------
def ordem_grupos(s):
    out, vistos = [], set()
    for j in np.argsort(-s, kind="stable"):
        g = int(eq[j])
        if g in vistos:
            continue
        vistos.add(g)
        out.append(int(j))
    return out


PRI = {"acerto": 0, "pai_filho": 1, "filho_pai": 1, "distante": 2, "irmao": 3, "outro_ramo": 4}


def rel_par(a, b):
    """a = gold, b = previsto. pai_filho: gold e pai direto do previsto."""
    pa, pb = partes[a], partes[b]
    if pa == pb:
        return "acerto"
    la, lb = len(pa), len(pb)
    if lb > la and pb[:la] == pa:
        return "pai_filho" if lb - la == 1 else "distante"
    if la > lb and pa[:lb] == pb:
        return "filho_pai" if la - lb == 1 else "distante"
    if la == lb and la >= 2 and pa[:-1] == pb[:-1]:
        return "irmao"
    return "outro_ramo"


def relacao(g, p):
    if eq[g] == eq[p]:
        return "acerto"
    best = None
    for a in membros[int(eq[g])]:
        for b in membros[int(eq[p])]:
            r = rel_par(a, b)
            if best is None or PRI[r] < PRI[best]:
                best = r
    return best


HIER_OK = ("acerto", "pai_filho", "filho_pai", "distante")


def avaliar(S, with_hier=False):
    a = b = c = d = e = h = 0
    for s, g, ex in zip(S, golds, extras):
        ordem = ordem_grupos(s)
        top = ordem[:TOPK]
        gs = {int(eq[x]) for x in top}
        a += int(top[0] == g)
        b += int(eq[top[0]] == eq[g])
        c += int(eq[g] in gs)
        d += int((eq[g] in gs) or any(eq[x] in gs for x in ex))
        h += int(relacao(g, top[0]) in HIER_OK)
    f = lambda v: round(v / len(golds) * 100, 2)
    return dict(top1_estrito=f(a), top1_grupo=f(b), top3_grupo=f(c), top3_central_ou_extra=f(d), top1_hier=f(h))


# ================= 1. ABLACAO =================
print("\n" + "=" * 70 + "\n1. ABLACAO (sem desempate; w = peso do embedding)\n" + "=" * 70)
abl = []
for z in (True, False):
    for w in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        S = hibrido(Semb, Stf, float(w), z)
        mt = avaliar(S)
        tipo = "so_tfidf" if w == 0.0 else ("so_embedding" if w == 1.0 else "hibrido")
        abl.append(dict(zscore=z, W_EMB=float(w), W_TFIDF=round(1.0 - float(w), 1), tipo=tipo, **mt))
abl = pd.DataFrame(abl)
ib = abl.sort_values(["top1_grupo", "top3_central_ou_extra", "top3_grupo"], ascending=False).index[0]
abl["melhor"] = ""
abl.loc[ib, "melhor"] = "<== MELHOR"
abl.to_csv(P_ABL, index=False, encoding="utf-8-sig")
cols = ["zscore", "W_EMB", "tipo", "top1_grupo", "top3_grupo", "top3_central_ou_extra", "top1_hier", "melhor"]
print(abl[cols].to_string(index=False))
ref = abl[(abl["zscore"]) & (abl["W_EMB"] == W_EMB_PADRAO)].iloc[0]
print("\nreferencia atual (z=True, W_EMB=0.6): top1_grupo=%.2f top3_grupo=%.2f top3_ce=%.2f" % (
    ref["top1_grupo"], ref["top3_grupo"], ref["top3_central_ou_extra"]))
print("melhor: z=%s W_EMB=%.1f top1_grupo=%.2f (delta vs atual: %+.2f pp)" % (
    abl.loc[ib, "zscore"], abl.loc[ib, "W_EMB"], abl.loc[ib, "top1_grupo"], abl.loc[ib, "top1_grupo"] - ref["top1_grupo"]))
print(AVISO)
print("[SAIDA]", P_ABL)

# ================= 2. HIERARQUIA DOS ERROS TOP-1 =================
print("\n" + "=" * 70 + "\n2. RELACAO HIERARQUICA DO TOP-1 (config padrao W_EMB=0.6, z=True)\n" + "=" * 70)
ordens = [ordem_grupos(s) for s in S0]
rels = [relacao(g, o[0]) for g, o in zip(golds, ordens)]
cr = Counter(rels)
nome_rel = [("acerto", "acerto (mesmo grupo)"),
            ("pai_filho", "gold=PAI -> previsto=FILHO direto"),
            ("filho_pai", "gold=FILHO -> previsto=PAI direto"),
            ("distante", "ancestral/descendente distante"),
            ("irmao", "irmao (mesmo pai)"),
            ("outro_ramo", "outro ramo")]
for k, d in nome_rel:
    print("  %-38s %3d  (%.1f%%)" % (d, cr.get(k, 0), cr.get(k, 0) / n * 100))
n_err = n - cr.get("acerto", 0)
n_hier = sum(cr.get(k, 0) for k in HIER_OK)
mt0 = avaliar(S0)
print("erros top-1 (grupo): %d" % n_err)
print("top1_grupo=%.2f (existente, inalterado) | top1_hier=%.2f (acerto + ancestral/descendente)" % (
    mt0["top1_grupo"], mt0["top1_hier"]))
if n_err:
    print("dos erros, %d (%.1f%%) sao relacao ancestral/descendente" % (n_hier - cr.get("acerto", 0), (n_hier - cr.get("acerto", 0)) / n_err * 100))

# ================= 3. DIAGNOSTICO DO DESEMPATE =================
print("\n" + "=" * 70 + "\n3. DESEMPATE (nao_confundir) EM QUESTOES AMBIGUAS (margem < %.2f)\n" % MARGEM + "=" * 70)
nomes_b = [C.nome_base(x) for x in nomes_t]
pats = [re.compile(r"(?<![a-z0-9])" + re.escape(x) + r"(?![a-z0-9])") if len(x) >= NOME_MIN_LEN else None for x in nomes_b]
nc = [C.norm_txt(x) for x in t["nao_confundir"]]
kws = []
for s in t["palavras_chave"]:
    termos = [C.norm_txt(x) for x in re.split(r"[;,]", s)]
    kws.append([x for x in dict.fromkeys(termos) if len(x) >= KW_MIN_LEN])


def kw_hits(j, qn):
    return sum(1 for k in kws[j] if (" " + k + " ") in qn)


def motivo_lado(a, b, qn):
    """None se 'a' e penalizavel por nao_confundir de 'b'; senao o motivo."""
    if pats[a] is None:
        return "nome_curto"
    if not nc[b]:
        return "nc_vazio"
    if not pats[a].search(nc[b]):
        return "nome_ausente_em_nc"
    if kw_hits(a, qn) > 0:
        return "kw_presente"
    return None


motivos, lados = Counter(), Counter()
n_amb = n_top2_gold = n_aplica = 0
amb_info = {}
for k in range(n):
    s = S0[k]
    qn = QN[rows[k]]
    od = ordem_grupos(s)
    if len(od) < 2:
        continue
    c1, c2 = od[0], od[1]
    marg = float(s[c1] - s[c2])
    if marg >= MARGEM:
        continue
    n_amb += 1
    m1_, m2_ = motivo_lado(c1, c2, qn), motivo_lado(c2, c1, qn)
    in1 = bool(pats[c2] and nc[c1] and pats[c2].search(nc[c1]))   # nome do top2 em nc do top1
    in2 = bool(pats[c1] and nc[c2] and pats[c1].search(nc[c2]))   # nome do top1 em nc do top2
    h1, h2 = kw_hits(c1, qn), kw_hits(c2, qn)
    if m1_ is None and m2_ is None:
        mot = "ambos_penalizaveis (regra nao aplica)"
    elif (m1_ is None) != (m2_ is None):
        mot = "APLICA"
        n_aplica += 1
    else:
        mot = "nenhum_penalizavel: top1=%s | top2=%s" % (m1_, m2_)
    motivos[mot] += 1
    for lado, mm in (("top1", m1_), ("top2", m2_)):
        lados["%s:%s" % (lado, mm if mm else "penalizavel")] += 1
    gold_top1 = eq[c1] == eq[golds[k]]
    gold_top2 = eq[c2] == eq[golds[k]]
    n_top2_gold += int((not gold_top1) and gold_top2)
    amb_info[k] = dict(marg=marg, mot=mot)
    print("id=%s marg=%.3f gold_top1=%s gold_top2=%s | top2_em_nc_top1=%s top1_em_nc_top2=%s | kw top1=%d top2=%d" % (
        gids[k], marg, gold_top1, gold_top2, in1, in2, h1, h2))
    print("   top1: %s" % cams[c1][:80])
    print("   top2: %s" % cams[c2][:80])
    print("   motivo: %s" % mot)
print("\nambiguas=%d de %d | regra aplicaria em %d | gold no top2 (e nao no top1) em %d" % (n_amb, n, n_aplica, n_top2_gold))
print("contagem por motivo:")
for k_, v in motivos.most_common():
    print("  %3d  %s" % (v, k_))
print("contagem por lado:")
for k_, v in sorted(lados.items()):
    print("  %3d  %s" % (v, k_))

# ================= 4. RANK DO GOLD =================
print("\n" + "=" * 70 + "\n4. DISTRIBUICAO DO RANK DO GOLD (rank de GRUPO de equivalencia)\n" + "=" * 70)
ranks = []
for g, od in zip(golds, ordens):
    grp = [int(eq[x]) for x in od]
    ranks.append(grp.index(int(eq[g])) + 1)
ranks = np.array(ranks)
buckets = [("1", lambda r: r == 1), ("2", lambda r: r == 2), ("3", lambda r: r == 3),
           ("4-5", lambda r: 4 <= r <= 5), ("6-10", lambda r: 6 <= r <= 10), (">10", lambda r: r > 10)]


def imprime_rank(nome, mask):
    r = ranks[mask]
    if len(r) == 0:
        print("  %-12s n=0" % nome)
        return
    cont = "  ".join("%s:%d" % (b, int(sum(1 for x in r if f(x)))) for b, f in buckets)
    print("  %-12s n=%3d | %s | mediana=%.1f" % (nome, len(r), cont, float(np.median(r))))


imprime_rank("TODAS", np.ones(n, dtype=bool))
rels_a = np.array(rels)
for k, _ in nome_rel:
    imprime_rank(k, rels_a == k)

# ================= 5. VARIANTE B =================
print("\n" + "=" * 70 + "\n5. VARIANTE B: caminho + descricao + palavras_chave (sem nao_confundir)\n" + "=" * 70)
textoB = [(c + " " + d + " " + k).strip() for c, d, k in zip(t["caminho"], t["descricao"], t["palavras_chave"])]
EtB = None
if P_EMB_B.exists():
    tmp = np.load(P_EMB_B)
    if tmp.shape == Et.shape:
        EtB = tmp.astype(np.float32)
        print("[CACHE] emb_temas_v10b.npy reutilizado", EtB.shape)
if EtB is None:
    EtB = C.codificar(get_encoder(), textoB)
    np.save(P_EMB_B, EtB)
    print("[EMB] emb_temas_v10b.npy gerado", EtB.shape)
nmB = np.linalg.norm(EtB, axis=1)
print("norma L2 min/max = %.6f / %.6f" % (nmB.min(), nmB.max()))

SembB = Eq[rows] @ EtB.T
StfB = tfidf_scores(textoB)
cfgs = [("original (emb+tfidf)", hibrido(Semb, Stf, W_EMB_PADRAO)),
        ("B emb + tfidf original", hibrido(SembB, Stf, W_EMB_PADRAO)),
        ("B emb + B tfidf", hibrido(SembB, StfB, W_EMB_PADRAO)),
        ("so embedding original", hibrido(Semb, Stf, 1.0)),
        ("so embedding B", hibrido(SembB, Stf, 1.0))]
res_b = []
for nome, S in cfgs:
    res_b.append(dict(config=nome, **avaliar(S)))
res_b = pd.DataFrame(res_b)
base_row = res_b.iloc[0]
for c in ("top1_grupo", "top3_grupo", "top3_central_ou_extra"):
    res_b["d_" + c] = (res_b[c] - base_row[c]).round(2)
print(res_b[["config", "top1_estrito", "top1_grupo", "top3_grupo", "top3_central_ou_extra", "top1_hier",
             "d_top1_grupo", "d_top3_grupo", "d_top3_central_ou_extra"]].to_string(index=False))
dmax = float(res_b.iloc[1:3][["d_top1_grupo", "d_top3_grupo", "d_top3_central_ou_extra"]].abs().max().max())
print("maior |delta| (variante B no hibrido): %.2f pp -> %s" % (dmax, "acima de 5 pp" if dmax > 5 else "dentro do ruido (<= 5 pp)"))
print(AVISO)

# ================= 6. diag_erros_v10.csv =================
out = []
for k in range(n):
    od = ordens[k]
    top = od[:TOPK]
    g = golds[k]
    r = dict(id=gids[k], caminho_gold=cams[g], rank_gold_grupo=int(ranks[k]), relacao=rels[k])
    for i in range(TOPK):
        r["top%d" % (i + 1)] = ("%s | score=%.4f" % (cams[top[i]], S0[k][top[i]])) if i < len(top) else ""
    r["ambigua"] = k in amb_info
    r["motivo_desempate"] = amb_info[k]["mot"] if k in amb_info else ""
    r["texto_200"] = re.sub(r"\s+", " ", qtexts[rows[k]])[:200]
    out.append(r)
pd.DataFrame(out).to_csv(P_ERR, index=False, encoding="utf-8-sig")
print("\n[SAIDA]", P_ERR, "(%d linhas)" % len(out))

print("OK n_gold=%d ablacao_linhas=%d erros_top1_grupo=%d ambiguas=%d desempate_aplicaria=%d top1_hier=%.2f" % (
    n, len(abl), n_err, n_amb, n_aplica, mt0["top1_hier"]))