# _diag_texto.py - Diagnostico de texto: folha (consulta) vs base de temas. Nao roda kNN, nao re-embeda.
import re, sys, json
from pathlib import Path
import csv

BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")
CACHE = BASE / "cache_offline"

P_FOL_IDS = CACHE / "emb_folha_ids.json"
P_TEM_IDS = CACHE / "emb_temas_v10_ids.json"
P_TEM_CSV = CACHE / "temas_limpos.csv"

def nid(v):
    if v is None:
        return ""
    return re.sub(r"\.0$", "", str(v).strip())

def falhar(msg):
    print("DIAG_TEXTO FALHOU: " + msg)
    sys.exit(1)

def ler_txt(p):
    for enc in ("utf-8-sig", "latin-1", "utf-8"):
        try:
            with open(p, encoding=enc, newline="") as f:
                return f.read(), enc
        except Exception:
            pass
    return None, None

def detectar_delim(txt):
    try:
        return csv.Sniffer().sniff(txt[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","

def achar_csv_texto(base):
    """Procura CSV com coluna de texto e coluna de ids Q... na raiz ou cache."""
    for pasta in (base, CACHE):
        for p in sorted(pasta.glob("*.csv")):
            txt, enc = ler_txt(p)
            if txt is None:
                continue
            delim = detectar_delim(txt)
            rd = csv.DictReader(txt.splitlines(), delimiter=delim)
            cols = [c.strip().lstrip("\ufeff") for c in (rd.fieldnames or [])]

            # detecta coluna de id com valores Q...
            id_col = None
            for row in rd:
                for c in cols:
                    v = str(row.get(c, "")).strip()
                    if re.match(r"^Q\d+", v):
                        id_col = c
                        break
                if id_col:
                    break

            if not id_col:
                continue

            # detecta coluna de texto longo
            rd2 = csv.DictReader(txt.splitlines(), delimiter=delim)
            cols2 = [c.strip().lstrip("\ufeff") for c in (rd2.fieldnames or [])]
            text_col = None
            for row in rd2:
                for c in cols2:
                    if len(str(row.get(c, "")).strip()) > 100:
                        text_col = c
                        break
                if text_col:
                    break

            if not text_col:
                continue
            return p, id_col, text_col
    return None, None, None


# ---------- 1. arquivos obrigatorios ----------
for p in (P_FOL_IDS, P_TEM_IDS, P_TEM_CSV):
    if not p.is_file():
        falhar("arquivo ausente: " + str(p))

# ---------- 2. ids da folha e temas ----------
with open(P_FOL_IDS, encoding="utf-8-sig") as f:
    folha_ids = [nid(x) for x in json.load(f)]
with open(P_TEM_IDS, encoding="utf-8-sig") as f:
    tema_ids = [nid(x) for x in json.load(f)]
print("[IDS] folha=%d | temas=%d" % (len(folha_ids), len(tema_ids)))

# ---------- 3. achar CSV com texto ----------
csv_path, id_col, text_col = achar_csv_texto(BASE)

if csv_path is None:
    print("[TEXTO] NAO_ENCONTRADO — nenhum CSV com texto e ids Q... na raiz/cache")
    print("DIAG_TEXTO OK")
    sys.exit(0)

print("[CSV] %s | id_col=%s | text_col=%s" % (csv_path, id_col, text_col))

# ---------- 4. gold_map (folha_verdade) ----------
folha_csv = BASE / "folha_verdade.csv"
gold_map = {}
if folha_csv.is_file():
    txt, _ = ler_txt(folha_csv)
    if txt is not None:
        rd = csv.DictReader(txt.splitlines(), delimiter=detectar_delim(txt))
        for row in rd:
            fid = nid(row.get("id", ""))
            if fid:
                gold_map[fid] = nid(row.get("tema_central_id", ""))

# ---------- 5. mapa texto -> id ----------
text_map = {}
txt2, _ = ler_txt(csv_path)
if txt2 is not None:
    rd = csv.DictReader(txt2.splitlines(), delimiter=detectar_delim(txt2))
    for row in rd:
        vid = nid(row.get(id_col, ""))
        vt = str(row.get(text_col, "") or "")
        if vid:
            text_map[vid] = vt

# ---------- 6. iterar folha_ids ----------
HTML_RX = re.compile(r"<[^>]+>|&lt;|&gt;|&amp;|&quot;|&#?\w+;")
RUIDO_RX = re.compile(r"\t|  +")
html_count = ruido_count = vazio_count = com_texto = sem_texto = 0

for qid in folha_ids:
    texto = text_map.get(qid, "")
    if not texto.strip():
        sem_texto += 1
        print("id=%s | len=0 | gold=%s | texto_vazio=SIM" % (qid, gold_map.get(qid, "")))
        continue

    com_texto += 1
    hl = "SIM" if HTML_RX.search(texto) else "NAO"
    if hl == "SIM": html_count += 1
    rl = "SIM" if RUIDO_RX.search(texto) else "NAO"
    if rl == "SIM": ruido_count += 1
    vl = "SIM" if not texto.strip() else "NAO"
    if vl == "SIM": vazio_count += 1

    s60 = texto.strip()[:60].replace("\n", "\\n").replace("\r", "\\r")
    e60 = texto.strip()[-60:].replace("\n", "\\n").replace("\r", "\\r") if len(texto.strip()) >= 60 else texto.strip().replace("\n", "\\n").replace("\r", "\\r")
    print("id=%s | len=%d | gold=%s | html=%s | ruido=%s | vazio=%s | ini=%s | fim=%s" % (
        qid, len(texto), gold_map.get(qid, ""), hl, rl, vl, s60, e60))

# ---------- 7. temas com nome/caminho curto ----------
temas_def = 0
try:
    txt_t, _ = ler_txt(P_TEM_CSV)
    if txt_t is not None:
        rd_t = csv.DictReader(txt_t.splitlines(), delimiter=detectar_delim(txt_t))
        cols_t = [c.strip().lstrip("\ufeff") for c in (rd_t.fieldnames or [])]
        tid_col = next((c for c in cols_t if c.lower() in ("tema_id", "id")), None)
        cam_col = next((c for c in cols_t if c.lower() == "caminho"), None)
        nome_col = next((c for c in cols_t if c.lower() in ("nome", "nome_tema", "titulo")), None)
        if not nome_col and cam_col:
            nome_col = cam_col
        tema_map = {}
        for row in rd_t:
            tid = nid(row.get(tid_col, ""))
            nome = str(row.get(nome_col, "") or "").strip() if nome_col else ""
            cam = str(row.get(cam_col, "") or "").strip() if cam_col else (nome or "")
            if tid:
                tema_map[tid] = (nome, cam)

        for tid in tema_ids:
            nome, cam = tema_map.get(tid, ("", ""))
            if not nome and not cam:
                temas_def += 1
                print("tema=%s | NOME E CAMINHO VAZIOS" % tid)
            elif len(nome) < 10 and not cam:
                temas_def += 1
                print("tema=%s | nome_tamanho=%d nome=%s" % (tid, len(nome), nome))
except Exception as e:
    temas_def = -1
    print("[AVISO] erro ao ler temas_limpos.csv: %s" % e)

# ---------- 8. resumo ----------
print("")
print("[TEXTO] total_folha_ids=%d | com_texto=%d | sem_texto=%d | html_presente=%d | ruido=%d | texto_vazio=%d | temas_deficientes=%d" % (
    len(folha_ids), com_texto, sem_texto, html_count, ruido_count, vazio_count, temas_def if temas_def >= 0 else -1))
print("DIAG_TEXTO OK")