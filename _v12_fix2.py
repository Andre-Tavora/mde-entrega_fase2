# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("v12_03_gera_entrega.py")
s = p.read_text(encoding="utf-8")
old = '''rows.append(dict(id=sid,tema_central_id=ids[p],decisao=decisao,
                     extras_ids=";".join(ex),confianca=round(conf,4),
                     motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else "")))'''
new = '''rows.append({
    "id": sid,
    "tema_central_id": ids[p],
    "decisao": decisao,
    "extras_ids": ";".join(ex),
    "confianca": round(conf, 4),
    "motivo": ("margem=" + str(round(m1v, 3))) if decisao == "revisar" else ""
})'''
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("[FIX2] rows.append corrigido")
else:
    print("[FIX2] trecho nao encontrado exato; imprimindo linhas 60-72:")
    for i, l in enumerate(s.splitlines()[59:72], start=60):
        print("%3d: %s" % (i, l))