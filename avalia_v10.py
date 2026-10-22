# avalia_v10.py - score hibrido (embedding + TF-IDF), grupos de equivalencia,
# desempate por nao_confundir, avaliacao top-1/top-3 e pares mais confundidos. Zero LLM.
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import v10_comum as C

# ===================== CONFIG =====================
W_EMB = 0.6              # peso do score de embedding
W_TFIDF = 0.4            # peso do score TF-IDF
USAR_ZSCORE = True       # z-score por linha antes da media ponderada (mesma escala)
MARGEM = 0.30            # desempate so se (score top1 - score top2) < MARGEM (em unidades do score combinado)
PENALIDADE = 0.40        # subtraida do grupo penalizado (> MARGEM para poder inverter)
KW_MIN_LEN = 4           # tamanho minimo de palavra-chave considerada
NOME_MIN_LEN = 6         # nomes de tema menores que isso nao entram na regra
TOPK = 3
N_PARES = 30
TFIDF_KW = dict(ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
                strip_accents="unicode", lowercase=True)
P_REL = C.BASE / "relatorio_v10.csv"
P_PARES = C.BASE / "pares_confundidos_v10.csv"
# ==================================================

# ---------- dados ----------
t = C.carregar_temas()
m = len(t)
ids = t["tema_id"].tolist()
idx = {x: i for i, x in enumerate(ids)}
if not (C.P_EMB_TEMAS.exists() and C.P_EMB_TEMAS_IDS.exists()):
    raise SystemExit("rode antes: gera_emb_temas_v10.py")
Et = np.load(C.P_EMB_TEMAS)
with open(C.P_EMB_TEMAS_IDS, encoding="utf-8") as f:
    if json.load(f) != ids:
        raise SystemExit("ordem dos IDs em emb_temas_v10_ids.json difere de temas_limpos.csv; rode gera_emb_temas_v10.py")
if Et.shape[0] != m:
    raise SystemExit("emb_temas_v10.npy com shape incompativel")

eq = C.grupos_equivalencia(t)
csize = np.bincount(eq)
print("[GRUPOS] grupos>1: %d | temas em grupos>1: %d" % (int((csize > 1).sum()), int(csize[eq][csize[eq] > 1].size)))
for k, cid in enumerate(np.where(csize > 1)[0][:30]):
    print("  grupo %d: " % (k + 1) + " | ".join(t["tema_id"][j] + ":" + t["nome"][j] for j in np.where(eq == cid)[0]))

q = C.ler_csv(C.P_QUEST)
q["id"] = q["id"].map(C.nid)
q = q.drop_duplicates("id").reset_index(drop=True)
qtexts = q["texto_limpo"].fillna("").astype(str).tolist()
qpos = {x: i for i, x in enumerate(q["id"])}
QN = [" " + C.norm_txt(x) + " " for x in qtexts]


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
    model = C.carregar_encoder()
    E = C.codificar(model, qtexts)
    np.save(C.P_EMB_Q, E)
    with open(C.P_EMB_Q_IDS, "w", encoding="utf-8") as f:
        json.dump(qids, f)
    return E


Eq = emb_questoes()

# ---------- score hibrido ----------
tf = TfidfVectorizer(**TFIDF_KW)
tf.fit(t["texto"].tolist() + qtexts)
Xt = tf.transform(t["texto"].tolist())
Xq = tf.transform(qtexts)


def zrow(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


S_emb = Eq @ Et.T
S_tf = (Xq @ Xt.T).toarray()
if USAR_ZSCORE:
    S_emb, S_tf = zrow(S_emb), zrow(S_tf)
S = (W_EMB * S_emb + W_TFIDF * S_tf) / (W_EMB + W_TFIDF)
print("[SCORE] W_EMB=%.2f W_TFIDF=%.2f zscore=%s | S %s" % (W_EMB, W_TFIDF, USAR_ZSCORE, S.shape))

# ---------- regra de desempate (nao_confundir) ----------
nomes = [C.nome_base(n) for n in t["nome"]]
pats = [re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])") if len(n) >= NOME_MIN_LEN else None for n in nomes]
nc = [C.norm_txt(x) for x in t["nao_confundir"]]
kws = []
for s in t["palavras_chave"]:
    termos = [C.norm_txt(x) for x in re.split(r"[;,]", s)]
    kws.append([x for x in dict.fromkeys(termos) if len(x) >= KW_MIN_LEN])


def kw_hits(j, qn):
    return sum(1 for k in kws[j] if (" " + k + " ") in qn)


def penaliza(a, b, qn):
    """a e penalizado se seu nome aparece em nao_confundir de b e nenhuma palavra-chave de a aparece na questao."""
    p = pats[a]
    return bool(p and nc[b] and p.search(nc[b]) and kw_hits(a, qn) == 0)


def top_grupos(s, k):
    out, vistos = [], set()
    for j in np.argsort(-s):
        g = eq[j]
        if g in vistos:
            continue
        vistos.add(g)
        out.append(int(j))
        if len(out) == k:
            break
    return out


def decidir(s, qn):
    """Retorna (top-k representantes de grupos distintos, info)."""
    antes = top_grupos(s, TOPK + 1)
    info = {"amb": False, "aplicou": False}
    if len(antes) >= 2:
        c1, c2 = antes[0], antes[1]          # c2 = melhor candidato de OUTRO grupo
        if s[c1] - s[c2] < MARGEM:
            info["amb"] = True
            p1, p2 = penaliza(c1, c2, qn), penaliza(c2, c1, qn)
            if p1 != p2:
                s2 = s.copy()
                alvo = c1 if p1 else c2
                s2[eq == eq[alvo]] -= PENALIDADE
                info["aplicou"] = True
                return top_grupos(s2, TOPK), info
    return antes[:TOPK], info


# ---------- conjuntos rotulados ----------
def achar_conjuntos():
    vistos, out = set(), []
    for d in (C.BASE, C.CACHE):
        for pad in ("folha_verdade*.csv", "gabarito*.csv"):
            for p in sorted(d.glob(pad)):
                rp = p.resolve()
                if rp in vistos:
                    continue
                vistos.add(rp)
                df = C.ler_csv(p)
                if "id" not in df.columns or "tema_central_id" not in df.columns:
                    print("[PULADO] %s: sem colunas id/tema_central_id" % p.name)
                    continue
                out.append((p.name, df))
    return out


def preparar(df):
    df = df.copy()
    df["id"] = df["id"].map(C.nid)
    if "fora_da_taxonomia" in df.columns:
        fora = df["fora_da_taxonomia"].map(C.nid) == "1"
    else:
        fora = pd.Series(False, index=df.index)
    df["central"] = df["tema_central_id"].map(C.nid)
    n0 = len(df)
    ok = (~fora) & df["central"].isin(idx) & df["id"].isin(qpos)
    df = df[ok].drop_duplicates("id")
    ex = df["extras_ids"] if "extras_ids" in df.columns else pd.Series("", index=df.index)
    extras = [{idx[x] for x in re.split(r"[;,|\s]+", str(v)) if C.nid(x) in idx and x} for v in ex]
    extras = [{idx[C.nid(x)] for x in re.split(r"[;,|\s]+", str(v)) if x and C.nid(x) in idx} for v in ex]
    return df.reset_index(drop=True), extras, n0


def metr(preds, golds, extras):
    n = len(golds)
    t1e = t1g = t3g = t3ce = 0
    for p, g, ex in zip(preds, golds, extras):
        gs = {int(eq[x]) for x in p}
        t1e += int(p[0] == g)
        t1g += int(eq[p[0]] == eq[g])
        t3g += int(eq[g] in gs)
        t3ce += int((eq[g] in gs) or any(eq[e] in gs for e in ex))
    f = lambda v: round(v / n * 100, 2) if n else float("nan")
    return dict(top1_estrito=f(t1e), top1_grupo=f(t1g), top3_grupo=f(t3g), top3_central_ou_extra=f(t3ce))


conjuntos = achar_conjuntos()
if not conjuntos:
    raise SystemExit("nenhum conjunto rotulado encontrado (folha_verdade*.csv / gabarito*.csv)")

rel, pares_out = [], []
cfg_txt = "emb=%.2f;tfidf=%.2f;z=%s;margem=%.2f;pen=%.2f" % (W_EMB, W_TFIDF, USAR_ZSCORE, MARGEM, PENALIDADE)
rep_cam = {}
for j in range(m):
    rep_cam.setdefault(int(eq[j]), (ids[j], t["caminho"][j]))

for nome, df0 in conjuntos:
    df, extras, n0 = preparar(df0)
    n = len(df)
    print("\n=== CONJUNTO %s: %d linhas | avaliadas %d ===" % (nome, n0, n))
    if n == 0:
        continue
    golds = [idx[x] for x in df["central"]]
    sem, com, infos = [], [], []
    for qid in df["id"]:
        qi = qpos[qid]
        s = S[qi]
        sem.append(top_grupos(s, TOPK))
        c, info = decidir(s, QN[qi])
        com.append(c)
        infos.append(info)
    m_sem, m_com = metr(sem, golds, extras), metr(com, golds, extras)
    n_amb = sum(i["amb"] for i in infos)
    n_apl = sum(i["aplicou"] for i in infos)
    ganhos = sum(1 for a, b, g in zip(sem, com, golds) if eq[a[0]] != eq[g] and eq[b[0]] == eq[g])
    perdas = sum(1 for a, b, g in zip(sem, com, golds) if eq[a[0]] == eq[g] and eq[b[0]] != eq[g])
    mesmo12 = 0
    for qid in df["id"]:
        o = np.argsort(-S[qpos[qid]])[:2]
        mesmo12 += int(eq[o[0]] == eq[o[1]])
    print("%-16s %12s %10s %10s %14s" % ("modo", "top1_estrito", "top1_grupo", "top3_grupo", "top3_cent/extra"))
    for modo, mm in (("sem_desempate", m_sem), ("com_desempate", m_com)):
        print("%-16s %12.2f %10.2f %10.2f %14.2f" % (modo, mm["top1_estrito"], mm["top1_grupo"], mm["top3_grupo"], mm["top3_central_ou_extra"]))
        rel.append(dict(conjunto=nome, n=n, modo=modo, **mm, ambiguas=n_amb, desempates_aplicados=n_apl,
                        ganhos=ganhos, perdas=perdas, top1_top2_mesmo_grupo=mesmo12, config=cfg_txt))
    print("ambiguas=%d | desempates aplicados=%d | ganhos=%d | perdas=%d | top1/top2 do mesmo grupo (nao ambiguas)=%d" % (n_amb, n_apl, ganhos, perdas, mesmo12))

    cont = Counter()
    for c, g in zip(com, golds):
        if eq[c[0]] != eq[g]:
            cont[(int(eq[g]), int(eq[c[0]]))] += 1
    total_err = sum(cont.values())
    for r, ((gg, pp), cnt) in enumerate(cont.most_common(N_PARES), start=1):
        pares_out.append(dict(conjunto=nome, rank=r, n_erros=cnt,
                              tema_correto_id=rep_cam[gg][0], caminho_correto=rep_cam[gg][1],
                              tema_previsto_id=rep_cam[pp][0], caminho_previsto=rep_cam[pp][1],
                              total_erros_conjunto=total_err))
    print("top 5 pares (correto -> previsto):")
    for (gg, pp), cnt in cont.most_common(5):
        print("  %dx | %s -> %s" % (cnt, rep_cam[gg][1][:60], rep_cam[pp][1][:60]))

pd.DataFrame(rel).to_csv(P_REL, index=False, encoding="utf-8-sig")
pd.DataFrame(pares_out).to_csv(P_PARES, index=False, encoding="utf-8-sig")
print("\n[SAIDA]", P_REL)
print("[SAIDA]", P_PARES)
print("AVISO: n pequeno; so aceite mudanca de pesos/margem com ganho robusto e valide em lote novo.")
print("OK n=%d conjuntos=%d" % (m, len(rel) // 2))