from pathlib import Path
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")

print("=== v12_calibracao.json ===")
print((BASE/"v12_calibracao.json").read_text(encoding="utf-8"))

print("\n=== _v12_entrega_final.py (trechos-chave) ===")
txt = (BASE/"_v12_entrega_final.py").read_text(encoding="utf-8")
for linha in txt.splitlines():
    if any(k in linha.lower() for k in ["def ", "regra", "tau", "t_fora", "wilson", "publicar", "fora", "csv", "json", "np.load", "texto_limpo", "questoes_limpas"]):
        print(linha)

print("\n=== v12_03_gera_entrega.py (caminhos e saidas) ===")
txt = (BASE/"v12_03_gera_entrega.py").read_text(encoding="utf-8")
for linha in txt.splitlines():
    if any(k in linha.lower() for k in ["def ", "csv", "json", "np.load", "saida", "publicar", "fora", "tau", "wilson"]):
        print(linha)
