# O que a Diana fez — resumo para o grupo

## O que implementei

### Análise de deposição de energia (`analysis/energy_deposition.py`)

Script completo de análise com PyROOT. Cobre:
- Distribuições de Landau por espécie (π, K, p) e por detetor, com ajuste
- MPV (Most Probable Value) vs número de detetor, com ajuste linear
- Composição do feixe por espécie (PDG codes)
- Verificação de consistência do MPV entre os 4 runs

Análises extra além do guião:
- **Overlay das três espécies** no mesmo gráfico — comparação directa de π, K, p
- **Curvas teóricas de Bethe-Bloch sobrepostas** ao dE/dx vs momento — validação do modelo físico
- **Consistência entre runs** — confirmar que os 4 ficheiros ROOT dão resultados coerentes

### Classificador de aprendizagem automática (`analysis/particle_id_ml.py`)

Classificador Gradient Boosting (HistGradientBoostingClassifier) para identificação π/K/p. Inclui:
- Treino com validação cruzada e balanceamento de classes
- Curvas ROC e AUC por espécie
- Matriz de confusão
- Importância das variáveis (permutation importance)
- AUC por bin de momento — para ver em que gama o classificador funciona melhor
- Curva eficiência-pureza para π vs p

### Relatório (`report/report.tex`)

Escrevi as secções: Introdução, Dados e Metodologia, 3.1 (deposição de energia), 3.6 (ML/PID), 3.7 (perspectivas), e a Conclusão (com espaço para as secções dos colegas).

Já integrei também a secção 3.5 (Distribuição do Momento) no documento.

---

## Para os colegas

Para contribuírem para o relatório:
1. `git fetch origin` + `git merge origin/Diana` na vossa branch
2. Editam `report/report.tex` na secção correspondente (marcada com comentários `% INSTRUÇÕES`)
3. Guardam as figuras em `plots/` numa subpasta com o nome da análise
4. Abrem um PR para a branch `Diana` (não para main)

Para dúvidas sobre a estrutura do LaTeX, olhem para as secções 3.1 ou 3.5 — é o mesmo padrão.

**Secções que ainda precisam de ser preenchidas:**
- 3.2 — Distribuição de Hits
- 3.3 — Vértices Hadrónicos
- 3.4 — Distribuição Temporal

---

## Ficheiros que peço que não alterem sem falar comigo

- `analysis/energy_deposition.py`
- `analysis/particle_id_ml.py`
- `plots/energy/*.png`
- Secções 3.1, 3.6, 3.7 do `report/report.tex`
