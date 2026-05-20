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

Já integrei também a secção 3.5 do José no documento.

---

## Para o Tiago

A tua parte é a **secção 3.2 (Distribuição de Hits)** e/ou **3.3 (Vértices Hadrónicos)** — confirma qual.

Para contribuíres para o relatório:
1. Faz `git fetch origin` + `git merge origin/Diana` na tua branch
2. Edita `report/report.tex` na secção correspondente (está marcada com comentários `% INSTRUÇÕES`)
3. Guarda as tuas figuras em `plots/` numa subpasta com o nome da tua análise
4. Commita e abre um PR para a branch `Diana` (não para main)

Se tiveres dúvidas sobre a estrutura do LaTeX, olha para como estão feitas as secções 3.1 ou 3.5 — é o mesmo padrão.

---

## Para o Rafael

A tua parte é a **secção 3.4 (Distribuição Temporal)** — confirma com o grupo.

Mesmo processo que o Tiago:
1. `git fetch origin` + `git merge origin/Diana`
2. Edita a tua secção em `report/report.tex`
3. Figuras em `plots/<nome-da-tua-análise>/`
4. PR para a branch `Diana`

---

## Para o José

A tua secção 3.5 (Distribuição do Momento) já está integrada no documento. Obrigada!

Uma coisa a confirmar: os caminhos das figuras no relatório ficaram `plots/momentum/...` (sem `../`). Se compilares localmente, as figuras precisam de estar em `plots/momentum/` a partir da raiz do projecto, não dentro de `report/`. Diz-me se precisas de ajustar.

---

## Ficheiros que peço que não alterem sem falar comigo

- `analysis/energy_deposition.py`
- `analysis/particle_id_ml.py`
- `plots/energy/*.png`
- Secções 3.1, 3.6, 3.7 do `report/report.tex`
