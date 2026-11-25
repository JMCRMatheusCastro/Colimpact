# ColImpact – Column Impact Analysis Tool

Análise não linear (NLG + NLF) de pilares de concreto armado sujeitos a carregamentos laterais,
incluindo impacto de veículos por Forças Estáticas Equivalentes (FEE).

**Programador:** José Matheus de Castro Rodrigues · Recife / PE, Brasil.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Como usar

Os dados entram na barra lateral e o modelo (seção com armadura, vinculações, cargas, sentidos e cotas) é
redesenhado no topo da página a cada alteração. O impacto de veículo é escolhido por categoria da NBR 6120
(garagens ou vias) ou no modo "Livre".

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `app.py` | Interface Streamlit |
| `colimpact_core.py` | Cálculo (diferenças finitas, diagrama N-M-1/r, rigidez secante, FSI) |
| `colimpact_modelo.py` | Definição do modelo: impacto da NBR 6120 (Tabelas 13 e 14), armadura por faces, d′, verificações |
| `colimpact_plots.py` | Gráficos de resultados e desenhos do modelo (Matplotlib) |
| `requirements.txt` | Dependências |

## Referências

- ARAÚJO, J. M. *Curso de Concreto Armado*, Vol. 1 e 3. 4. ed. Rio Grande: Dunas, 2014.
- ABNT NBR 6118:2023 – Projeto de estruturas de concreto.
- ABNT NBR 6120:2019 – Ações para o cálculo de estruturas de edificações (impacto de veículos: Tabelas 13 e 14).
- GHALI, A.; NEVILLE, A. *Structural Analysis*. 7. ed. CRC Press, 2017.
- ZIENKIEWICZ, O. C.; MORGAN, K. *Finite Elements and Approximation*. Dover, 2006.
