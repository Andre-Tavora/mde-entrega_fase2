# v11_experimentos.py - variantes de texto x fusao de grupos x regra hierarquica. Zero LLM.
import re
import sys
import json
import difflib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import v10_comum as C

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

# ===================== CONFIG =====================
W_EMB = 0.6
TFIDF_KW = dict(ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
                strip_accents="unicode", lowercase=True)
SEP = " > "
TOPK = 3
TS = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
P_FOL = C.BASE / "folha_verdade.csv"
P_RES = C.BASE / "v11_resultados.csv"
P_AUD = C.BASE / "auditoria_gold_v11.csv"
IDS_AUD = ["Q0005", "Q0058", "Q0077", "Q0106", "Q0138", "Q0160", "Q0202", "Q0208", "Q0259", "Q0274", "Q0279", "Q0286"]
AVISO = "n=92 -> margem de erro ~+-9 pp; so considere diferencas > 5 pp ou ganhos consistentes em varios T vizinhos (plato)"
M1_PARES = [
    ("Processo Penal > Recursos (Parte 03) - Espécies", "Processo Penal > Recursos em Espécies"),
    ("Penal > Estatuto do Idoso - Crimes", "Direito Penal - Legislação Especial > Estaturo de Idoso - Crimes"),
]
M2_PARES = [
    ("Processo Penal > Sentença Penal > Efeitos Decorrentes da Sentença Penal Condenatória",
     "Penal > Teoria da Pena > Efeitos da Condenação Penais e Extrapenais"),
    ("Direito Penal - Legislação Especial > Juizado Especial Criminal", "Processo Penal > Procedimento Sumaríssimo"),
]
FUS_M2 = "M2_conceitual_infla_metrica"
# ==================================================

# ---------- temas ----------
t = C.carregar_temas()
m = len(t)
ids = t["tema_id"].tolist()
idx = {x: i for i, x in enumerate(ids)}
cams = t["caminho"].tolist()
partes = [[p for p in c.split(SEP) if p] for c in cams]
if not (C.P_EMB_TEMAS.exists() and C.P_EMB_TEMAS_IDS.exists()):
    raise SystemExit("rode antes: gera_emb_temas_v10.py")
Et = np.load(C.P_EMB_TEMAS).astype(np.float32)
with open(C.P_EMB_TEMAS_IDS, encoding="utf-8") as f:
    if json.load(f) != ids:
        raise SystemExit("ordem dos IDs em emb_temas_v10_ids.json difere de temas_limpos.csv")
if Et.shape[0] != m:
    raise SystemExit("emb_temas_v10.npy com shape incompativel")

cam2idx = defaultdict(list)
for j, c in enumerate(cams):
    cam2idx[c].append(j)
faltam = []
for a, b in M1_PARES + M2_PARES:
    for p in (a, b):
        if p not in cam2idx:
            sug = difflib.get_close_matches(p, list(cam2idx), n=3, cutoff=0.6)
            faltam.append((p, sug))
if faltam:
    print("ERRO: caminhos de fusao inexistentes em temas_limpos.csv:")
    for p, sug in faltam:
        print("  - " + p)
        print("    parecidos: " + " || ".join(sug))
    raise SystemExit("corrija os caminhos de fusao e rode novamente")

eq0 = np.asarray(C.grupos_equivalencia(t))


def fundir(eq, pares):
    labels = [int(x) for x in eq]
    uf = {l: l for l in set(labels)}

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for a, b in pares:
        ls = [labels[j] for j in cam2idx[a] + cam2idx[b]]
        for l in ls[1:]:
            ra, rb = find(ls[0]), find(l)
            if ra != rb:
                uf[rb] = ra
    return np.asarray(pd.factorize(pd.Series([find(l) for l in labels]))[0])


PRI = {"acerto": 0, "pai_filho": 1, "filho_pai": 1, "distante": 2, "irmao": 3, "outro_ramo": 4}
HIER_OK = ("acerto", "pai_filho", "filho_pai", "distante")


def rel_par(a, b):
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


class EqInfo:
    def __init__(self, eq):
        self.eq = np.asarray(eq)
        self.membros = {}
        for j in range(m):
            self.membros.setdefault(int(self.eq[j]), []).append(j)
        self._pai = {}
        self._rel = {}

    def pai_dir(self, p, c):
        """True se algum membro do grupo de p e pai DIRETO de algum membro do grupo de c."""
        key = (int(self.eq[p]), int(self.eq[c]))
        if key not in self._pai:
            ok = False
            for a in self.membros[key[0]]:
                for b in self.membros[key[1]]:
                    if len(partes[b]) == len(partes[a]) + 1 and partes[b][:-1] == partes[a]:
                        ok = True
                        break
                if ok:
                    break
            self._pai[key] = ok
        return self._pai[key]

    def relacao(self, g, p):
        if self.eq[g] == self.eq[p]:
            return "acerto"
        key = (int(self.eq[g]), int(self.eq[p]))
        if key not in self._rel:
            best = None
            for a in self.membros[key[0]]:
                for b in self.membros[key[1]]:
                    r = rel_par(a, b)
                    if best is None or PRI[r] < PRI[best]:
                        best = r
            self._rel[key] = best
        return self._rel[key]


FUS = {
    "M0": EqInfo(eq0),
    "M1": EqInfo(fundir(eq0, M1_PARES)),
    FUS_M2: EqInfo(fundir(eq0, M1_PARES + M2_PARES)),
}
for k, ei in FUS.items():
    print("[FUSAO] %-30s grupos=%d (M0=%d)" % (k, len(ei.membros), len(FUS["M0"].membros)))

# ---------- questoes e variantes ----------
q = C.ler_csv(C.P_QUEST)
q["id"] = q["id"].map(C.nid)
q = q.drop_duplicates("id").reset_index(drop=True)
q0 = q["texto_limpo"].fillna("").astype(str).tolist()
qids = q["id"].tolist()
qpos = {x: i for i, x in enumerate(qids)}
nq = len(q0)

RE_ENUN = re.compile(r"Enunciado\s*:", re.I)
RE_ASS = re.compile(r"Assunto\s*:[^|]*(?:\|\s*)?", re.I)
RE_DIS = re.compile(r"Disciplina\s*:[^|]*(?:\|\s*)?", re.I)


def partir(s):
    mm = RE_ENUN.search(s)
    if mm:
        return s[:mm.start()], s[mm.start():], True
    return s, "", False


def var1(s):
    h, tl, ok = partir(s)
    return (RE_ASS.sub("", h, count=1) + tl).strip() if ok else s


def var2(s):
    h, tl, ok = partir(s)
    if not ok:
        return s
    return (RE_DIS.sub("", RE_ASS.sub("", h, count=1), count=1) + tl).strip()


def var3(s):
    h, tl, ok = partir(s)
    return tl.strip() if ok else s


QV = {"V0": q0, "V1": [var1(s) for s in q0], "V2": [var2(s) for s in q0], "V3": [var3(s) for s in q0]}
if QV["V3"] == QV["V2"]:
    del QV["V3"]
    print("[VARIANTE] V3 identica a V2 -> omitida")

hs = [partir(s) for s in q0]
n_enun = sum(1 for h in hs if h[2])
n_dis = sum(1 for h in hs if RE_DIS.search(h[0]))
n_ass = sum(1 for h in hs if RE_ASS.search(h[0]))
print("\n[PADROES] questoes=%d | Enunciado: %d (%.1f%%) | Disciplina: %d (%.1f%%) | Assunto: %d (%.1f%%)" % (
    nq, n_enun, n_enun / nq * 100, n_dis, n_dis / nq * 100, n_ass, n_ass / nq * 100))
for nome, v in (("Enunciado", n_enun), ("Disciplina", n_dis), ("Assunto", n_ass)):
    if v / nq < 0.95:
        print("[AVISO] padrao '%s' encontrado em apenas %.1f%% (< 95%%); confira as variantes" % (nome, v / nq * 100))
exs = [i for i in range(nq) if hs[i][2] and RE_ASS.search(hs[i][0]) and RE_DIS.search(hs[i][0])][:3] or list(range(min(3, nq)))
for i in exs:
    print("\n--- exemplo id=%s ---" % qids[i])
    for vn in QV:
        print("%s: %s" % (vn, re.sub(r"\s+", " ", QV[vn][i])[:220]))

# ---------- embeddings ----------
_enc = {"m": None}


def encoder():
    if _enc["m"] is None:
        _enc["m"] = C.carregar_encoder()
    return _enc["m"]


def emb_variante(vn):
    dim = Et.shape[1]
    if vn == "V0" and C.P_EMB_Q.exists() and C.P_EMB_Q_IDS.exists():
        with open(C.P_EMB_Q_IDS, encoding="utf-8") as f:
            ic = [C.nid(x) for x in json.load(f)]
        E = np.load(C.P_EMB_Q)
        if ic == qids and E.shape == (nq, dim):
            print("[CACHE] V0: emb_659.npy reutilizado", E.shape)
            return E.astype(np.float32)
    p = C.CACHE / ("emb_659_v11_%s.npy" % vn)
    pj = C.CACHE / ("emb_659_v11_%s_ids.json" % vn)
    if p.exists():
        E = np.load(p)
        ok_ids = True
        if pj.exists():
            with open(pj, encoding="utf-8") as f:
                ok_ids = [C.nid(x) for x in json.load(f)] == qids
        if E.shape == (nq, dim) and ok_ids:
            print("[CACHE] %s: %s reutilizado %s" % (vn, p.name, E.shape))
            return E.astype(np.float32)
    print("[EMB] %s: gerando embeddings (%d textos, CPU pode levar alguns minutos)" % (vn, nq))
    E = C.codificar(encoder(), QV[vn])
    np.save(p, E)
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(qids, f)
    print("[EMB] %s salvo em %s %s" % (vn, p.name, E.shape))
    return E


def zrow(S):
    return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-9)


S_ALL = {}
for vn in QV:
    Eq = emb_variante(vn)
    tf = TfidfVectorizer(**TFIDF_KW)
    tf.fit(t["texto"].tolist() + QV[vn])
    Stf = (tf.transform(QV[vn]) @ tf.transform(t["texto"].tolist()).T).toarray()
    S_ALL[vn] = W_EMB * zrow(Eq @ Et.T) + (1.0 - W_EMB) * zrow(Stf)
    print("[SCORE] %s shape=%s" % (vn, S_ALL[vn].shape))

# ---------- gabarito ----------
fol_all = C.ler_csv(P_FOL)
for c in ("id", "tema_central_id"):
    if c not in fol_all.columns:
        raise SystemExit("coluna ausente em folha_verdade.csv: " + c)
fol_all["id"] = fol_all["id"].map(C.nid)
fol = fol_all.copy()
if "fora_da_taxonomia" in fol.columns:
    fora = fol["fora_da_taxonomia"].map(C.nid) == "1"
else:
    fora = pd.Series(False, index=fol.index)
fol["central"] = fol["tema_central_id"].map(C.nid)
n0 = len(fol)
ok = (~fora) & fol["central"].isin(idx) & fol["id"].isin(qpos)
fol = fol[ok].drop_duplicates("id").reset_index(drop=True)
exs_col = fol["extras_ids"] if "extras_ids" in fol.columns else pd.Series("", index=fol.index)
extras = [{idx[C.nid(x)] for x in re.split(r"[;,|\s]+", str(v)) if x and C.nid(x) in idx} for v in exs_col]
golds = [idx[x] for x in fol["central"]]
gids = fol["id"].tolist()
rows = [qpos[x] for x in gids]
n = len(golds)
print("\n[GABARITO] linhas=%d | avaliadas=%d" % (n0, n))
if n == 0:
    raise SystemExit("nenhuma questao avaliavel")


# ---------- ordenacao e regras ----------
def ordem(s, eq, lim=5):
    out, vistos = [], set()
    for j in np.argsort(-s, kind="stable"):
        g = int(eq[j])
        if g in vistos:
            continue
        vistos.add(g)
        out.append(int(j))
        if len(out) >= lim:
            break
    return out


def aplicar(o, s, ei, regra, T):
    if regra == "none" or T <= 0 or len(o) < 2:
        return o
    kmax = 1 if regra == "R1" else min(2, len(o) - 1)
    for k in range(1, kmax + 1):
        if (s[o[0]] - s[o[k]]) < T and ei.pai_dir(o[k], o[0]):
            return [o[k]] + o[:k] + o[k + 1:]
    return o


def avaliar(ords, ei):
    a = b = c = d = h = 0
    flag = np.zeros(n, dtype=bool)
    for k, (o, g, ex) in enumerate(zip(ords, golds, extras)):
        top = o[:TOPK]
        gs = {int(ei.eq[x]) for x in top}
        eg = int(ei.eq[g])
        a += int(top[0] == g)
        okg = int(ei.eq[top[0]]) == eg
        b += int(okg)
        flag[k] = okg
        c += int(eg in gs)
        d += int((eg in gs) or any(int(ei.eq[x]) in gs for x in ex))
        h += int(ei.relacao(g, top[0]) in HIER_OK)
    f = lambda v: round(v / n * 100, 2)
    return dict(top1_estrito=f(a), top1_grupo=f(b), top3_grupo=f(c), top3_central_ou_extra=f(d), top1_hier=f(h)), flag


CFG_REGRAS = [("none", 0.0)] + [(r, T) for r in ("R1", "R1b") for T in TS]
res = []
for vi, vn in enumerate(QV):
    Sg = S_ALL[vn][rows]
    for fi, (fn, ei) in enumerate(FUS.items()):
        base_ord = [ordem(s, ei.eq) for s in Sg]
        for regra, T in CFG_REGRAS:
            ords = [aplicar(o, s, ei, regra, T) for o, s in zip(base_ord, Sg)]
            mt, flag = avaliar(ords, ei)
            res.append(dict(variante=vn, fusao=fn, regra=regra, T=T, **mt, _flag=flag, _vi=vi, _fi=fi))

bi = [i for i, r in enumerate(res) if r["variante"] == "V0" and r["fusao"] == "M0" and r["regra"] == "none"][0]
bm, bflag = res[bi], res[bi]["_flag"]
print("\n[BASE] V0/M0/sem regra: top1_estrito=%.2f top1_grupo=%.2f top3_grupo=%.2f top3_ce=%.2f hier=%.2f (V10 esperado top1_grupo=34.78)" % (
    bm["top1_estrito"], bm["top1_grupo"], bm["top3_grupo"], bm["top3_central_ou_extra"], bm["top1_hier"]))
for i, r in enumerate(res):
    fl = r.pop("_flag")
    gan = [gids[k] for k in range(n) if (not bflag[k]) and fl[k]]
    per = [gids[k] for k in range(n) if bflag[k] and (not fl[k])]
    r["base"] = "BASE" if i == bi else ""
    for c in ("top1_estrito", "top1_grupo", "top3_grupo", "top3_central_ou_extra", "top1_hier"):
        r["d_" + c] = round(r[c] - bm[c], 2)
    r["ganhos"], r["perdas"], r["saldo"] = len(gan), len(per), len(gan) - len(per)
    r["ids_ganhos"], r["ids_perdas"] = ";".join(gan), ";".join(per)
df = pd.DataFrame(res).sort_values(["_vi", "_fi", "regra", "T"], kind="stable")
df["regra"] = df["regra"].replace({"none": "sem_regra"})
df = df.drop(columns=["_vi", "_fi"])
cols = ["base", "variante", "fusao", "regra", "T", "top1_estrito", "top1_grupo", "top3_grupo", "top3_central_ou_extra",
        "top1_hier", "d_top1_grupo", "d_top3_grupo", "d_top3_central_ou_extra", "d_top1_hier",
        "ganhos", "perdas", "saldo", "ids_ganhos", "ids_perdas"]
df = df[cols]
df.to_csv(P_RES, index=False, encoding="utf-8-sig")

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)
pc = ["base", "variante", "fusao", "regra", "T", "top1_estrito", "top1_grupo", "top3_grupo",
      "top3_central_ou_extra", "top1_hier", "d_top1_grupo", "ganhos", "perdas", "saldo"]
disp = df[pc].copy()
disp["fusao"] = disp["fusao"].replace({FUS_M2: "M2*"})
print("\n" + "=" * 100 + "\nTABELA COMPLETA (M2* = " + FUS_M2 + ")\n" + "=" * 100)
print(disp.to_string(index=False))

print("\n" + "=" * 100 + "\nTOP 15 por top1_grupo (desempate: top3_ce, sem regra primeiro)\n" + "=" * 100)
rk = df.assign(_nr=(df["regra"] != "sem_regra").astype(int)).sort_values(
    ["top1_grupo", "top3_central_ou_extra", "_nr", "saldo"], ascending=[False, False, True, False], kind="stable")
rd = rk[pc].head(15).copy()
rd["fusao"] = rd["fusao"].replace({FUS_M2: "M2*"})
print(rd.to_string(index=False))

print("\nPLATO R1/R1b em V0/M0 (T vizinhos):")
pl = df[(df["variante"] == "V0") & (df["fusao"] == "M0")][["regra", "T", "top1_grupo", "d_top1_grupo", "ganhos", "perdas", "saldo"]]
print(pl.to_string(index=False))
print("\n" + AVISO)

# ---------- sanidade nas 659 ----------
def alterado_659(vn, fn, regra, T):
    ei, S = FUS[fn], S_ALL[vn]
    ch = 0
    for s in S:
        o = ordem(s, ei.eq)
        o2 = aplicar(o, s, ei, regra, T)
        ch += int(o2[0] != o[0])
    return ch, ch / len(S) * 100


best = rk.iloc[0]
best_nm2 = rk[rk["fusao"] != FUS_M2].iloc[0]
rb = "none" if best["regra"] == "sem_regra" else best["regra"]
rn = "none" if best_nm2["regra"] == "sem_regra" else best_nm2["regra"]
print("\n" + "=" * 100 + "\nSANIDADE NAS %d QUESTOES (sem rotulo)\n" % nq + "=" * 100)
ch_b, p_b = alterado_659(best["variante"], best["fusao"], rb, float(best["T"]))
ch_n, p_n = alterado_659(best_nm2["variante"], best_nm2["fusao"], rn, float(best_nm2["T"]))
ch_r, p_r = alterado_659("V0", "M0", "R1", 0.5)
print("melhor config geral : %s/%s/%s T=%s top1_grupo=%.2f -> top-1 alterado em %d/%d (%.2f%%)" % (
    best["variante"], best["fusao"], best["regra"], best["T"], best["top1_grupo"], ch_b, nq, p_b))
print("melhor sem fusao M2 : %s/%s/%s T=%s top1_grupo=%.2f -> top-1 alterado em %d/%d (%.2f%%)" % (
    best_nm2["variante"], best_nm2["fusao"], best_nm2["regra"], best_nm2["T"], best_nm2["top1_grupo"], ch_n, nq, p_n))
print("R1 T=0.5 em V0/M0   : top-1 alterado em %d/%d (%.2f%%)" % (ch_r, nq, p_r))
if max(p_b, p_n, p_r) > 15:
    print("ALERTA: a regra pode estar trocando demais (> 15%% das questoes).")
if best["fusao"] == FUS_M2:
    print("NOTA: a melhor config usa M2 (fusao conceitual) -> ganho pode ser metrica inflada, nao acerto real.")
if best["regra"] == "sem_regra":
    print("NOTA: a melhor config nao usa regra; % alterado pela regra = 0 por definicao.")

# ---------- auditoria de gold ----------
b_eq = FUS["M0"]
S0 = S_ALL["V0"]
fa = fol_all.drop_duplicates("id").set_index("id")
aud = []
for i in IDS_AUD:
    if i not in qpos:
        print("[AVISO] id %s ausente em questoes_limpas.csv" % i)
        aud.append(dict(id=i, caminho_gold="", top1_previsto="", texto_limpo=""))
        continue
    k = qpos[i]
    gold_cam = ""
    if i in fa.index:
        cid = C.nid(fa.loc[i, "tema_central_id"])
        if cid in idx:
            gold_cam = cams[idx[cid]]
        elif "tema_central_caminho" in fa.columns:
            gold_cam = str(fa.loc[i, "tema_central_caminho"])
    o = ordem(S0[k], b_eq.eq, 1)
    aud.append(dict(id=i, caminho_gold=gold_cam, top1_previsto=cams[o[0]], texto_limpo=q0[k][:1500]))
pd.DataFrame(aud).to_csv(P_AUD, index=False, encoding="utf-8-sig")
print("\n[SAIDA]", P_RES, "(%d linhas)" % len(df))
print("[SAIDA]", P_AUD, "(%d linhas)" % len(aud))

print("\nOK linhas_csv=%d | melhor=%s/%s/%s/T=%s top1_grupo=%.2f | alterado_659_melhor=%.2f%% | alterado_659_R1_T0.5_V0M0=%.2f%% | n_gold=%d" % (
    len(df), best["variante"], best["fusao"], best["regra"], best["T"], best["top1_grupo"], p_b, p_r, n))