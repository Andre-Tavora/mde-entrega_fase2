from pathlib import Path
BASE = Path(r"C:\Users\André Tavora\Documents\CASE2_COLAB\CASE_2-20261021T010007Z-1-001\CASE_2")

print("=== TODOS OS ARQUIVOS (raiz) ===")
for p in sorted(BASE.iterdir()):
    if p.is_file():
        print(f"{p.name}  ({p.stat().st_size} bytes)")

print("\n=== cache_offline ===")
cache = BASE / "cache_offline"
if cache.is_dir():
    for p in sorted(cache.iterdir()):
        print(f"{p.name}  ({p.stat().st_size} bytes)")
else:
    print("(cache_offline não existe)")

print("\n=== Qualquer arquivo .jsonl em qualquer subpasta (distância 2) ===")
for d in [BASE] + [x for x in BASE.iterdir() if x.is_dir()]:
    for p in d.glob("*.jsonl"):
        print(p)
