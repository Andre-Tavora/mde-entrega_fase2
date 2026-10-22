# CASE 2 - Sistema de Classificacao (Metodo VDE)

## Entrega (prazo 22/10 14h)
- `publicacao.csv` - 659 questoes com decisao (publicar/revisar/fora), tema central, extras, confianca (gap)
- `previsao.txt` - precisao, cobertura, IC 95% Wilson e metodo
- Este README

## Como o codigo funciona (offline, sem LLM)
1. Embeddings BGE-M3 das 659 questoes + embeddings dos 377 temas (`cache_offline/emb_*.npy`).
2. Similaridade coseno normalizada (z-score por linha).
3. Decisao:
   - `publicar`: gap top1-top2 >= 17.848  (precisao 91.3% na metrica oficial, IC Wilson honesto)
   - `fora`: score maximo < 0.10
   - `revisar`: resto
4. Metrica oficial: acerto se gold == tema central OU pai OU filho.

## Resultado (nesta rodada)
- Publicadas na folha-verdade: 46/92
- Precisao: 91.3% (IC 95% Wilson no previsao.txt)
- Cobertura: 50.0%
- Fora da folha: ver previsao.txt

## Limitacao honesta
Com o encoder atual, a precisao maxima alcancavel e ~91%, nao 95%.
Atingir 95% exige re-treino do encoder (pos-entrega). O relatorio detalha.

## Reproducao
- Codigo: scripts `_*_v12.py` + `v12_*.py` na raiz.
- Insumos: `cache_offline/emb_659.npy`, `emb_temas_v10.npy`, `temas_limpos.csv`, `folha_verdade.csv`.
- Um comando regenera: `python _entrega_final_v12.py`
