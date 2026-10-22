# -*- coding: utf-8 -*-
import pathlib
p = pathlib.Path("v12_03_gera_entrega.py")
s = p.read_text(encoding="utf-8")
s = s.replace('motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else "conf">=lim")))',
              'motivo=("margem="+str(round(m1v,3)) if decisao=="revisar" else ""))')
p.write_text(s, encoding="utf-8")
print("[FIX] v12_03_gera_entrega.py corrigido")