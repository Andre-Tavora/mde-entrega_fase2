# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("v12_03_gera_entrega.py")
s = p.read_text(encoding="utf-8")
# Troca as 3 linhas do rows.append
old = '''    rows.append(dict(id=sid,tema_central_id=ids[p],decisao=decisao,
                     extras_ids=";".join(ex),confianca=round(conf,4),
                     motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else ""))'''
new = '''    rows.append(dict(id=sid,tema_central_id=ids[p],decisao=decisao,
                     extras_ids=";".join(ex),confianca=round(conf,4),
                     motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else "")))'''
if old in s:
    s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")
    print("[FIX3] adicionado parentese de fechamento")
else:
    print("[FIX3] trecho exato nao encontrado; corrigindo manualmente as 3 linhas...")
    lines = s.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip().startswith("motivo=(") and "revisar" in line:
            # garante que termina com )))
            lines[i] = line.rstrip() + "))\n"
        if line.strip().startswith("rows.append") and "dict" in line:
            # verifica se termina com (
            pass
    s = "".join(lines)
    p.write_text(s, encoding="utf-8")
    print("[FIX3] tentativa manual aplicada")