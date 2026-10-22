# Classificador V9 (Metodo VDE) - inferencia offline

## Objetivo
Classificar questoes de Direito na arvore de temas (377 temas) com similaridade vetorial:
TF-IDF + TF-IDF de caracteres + embeddings BAAI/bge-m3, subida restrita na hierarquia (tau=2).
Zero LLM generativo na inferencia.

## Estrutura
```
saida_pipeline_v4/
  inferir_v9.py            # funcao classificar(textos, ids=None, embeddings=None)
  classificar_csv.py       # CLI
  requirements.txt
  README.md
  MANIFEST.sha256.json
  predicoes_v9_final.csv   # predicoes de referencia nas 92 questoes
  artefatos_v9_final/
    pipeline_v9_congelado.joblib
    vec_ch.joblib
    encoder_local/         # BAAI/bge-m3 (~2,2 GB)
    config.json
```

## Como rodar
CLI (entrada com colunas `id` e `texto`, utf-8-sig):
```
pip install -r requirements.txt
python classificar_csv.py --entrada entrada.csv --saida saida.csv
```
Import:
```
from inferir_v9 import classificar
df = classificar(textos, ids=ids)
```
Saida: id, tema_id, caminho_previsto, depth, m1, m2, status. Texto vazio sai com status REVISAR.

## Regra de status
- `PUBLICAR` se m1 >= 2.2525
- `REVISAR` caso contrario (revisao humana)

m1 = score do tema previsto menos o melhor score fora da classe de equivalencia e da linhagem do previsto.

## Metricas (92 questoes do gabarito)
- Exato: 44.6
- F1 macro: 42.7
- Aceito restrito (exato, equivalente ou linhagem com ancestral de depth >= 2): 66.3

## Limitacoes
- O limiar foi escolhido no proprio gabarito, com n=92, sem certificacao. Revisar com novas questoes rotuladas.
- `inferir_v9.py` e `artefatos_v9_final/` devem ser copiados juntos.
- O `vec_ch` foi ajustado nas 92 questoes do gabarito (deterministico; muda se for reajustado).
- Encoder BAAI/bge-m3 salvo localmente (~2,2 GB); a inferencia nao acessa a internet.
- Sem regras de leis, sem Disciplina, sem Assunto.
- Zero LLM generativo.
