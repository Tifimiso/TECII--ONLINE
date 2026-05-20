# TECII — ONLINE | Trabalho de Grupo 2024/2025

Repositório do trabalho de grupo da unidade curricular **Tópicos em Engenharia Computacional II (42186)**, Departamento de Física, Universidade de Aveiro.

## Descrição

Análise de dados de simulação Geant4 do detector **AMBER Target**.
O objetivo é processar os ficheiros de saída da simulação (formato ROOT) e produzir histogramas e distribuições das grandezas físicas pedidas no guia do trabalho prático.

## Membros do Grupo

| Nome   | Branch Git |
|--------|------------|
| Diana  | `Diana`    |
| Tiago  | `Tiago`    |
| José   | `jose_branch`   |
| Rafael | `rafinha_branch` |

## Estrutura do Repositório

Ver [STRUCTURE.md](STRUCTURE.md) para a descrição completa de todas as pastas e ficheiros.

## Regras de Trabalho em Git

> **Cada membro do grupo deve trabalhar na sua própria branch.**
> Ninguém deve fazer commits diretamente na branch `main`.
> As contribuições são integradas em `main` via Pull Request, depois de revisão.

### Convenção de Commits

Os commits devem seguir o formato abaixo para que o histórico seja claro.
O professor avalia a quantidade **e qualidade** dos commits — mensagens vagas como
`"update"` ou `"fix"` não contam.

**Formato:**
```
<tipo>: <descrição curta e clara do que foi feito>

<explicação opcional: porquê, o que mudou, decisões tomadas>
```

**Tipos disponíveis:**

| Tipo | Quando usar |
|------|-------------|
| `feat` | Adicionaste uma nova funcionalidade ou análise |
| `fix` | Corrigiste um erro num script ou ficheiro |
| `refactor` | Reorganizaste código sem mudar o comportamento |
| `docs` | Alteraste documentação (README, comentários, relatório) |
| `chore` | Tarefas de manutenção (gitignore, dependências, estrutura) |

**Exemplo de um bom commit:**
```
feat: implementar histograma de deposição de energia por detetor

Adicionada a função plot_energy_deposition_per_detector() em
analysis/energy_deposition.py. Lê os dados do ficheiro ROOT com
uproot, filtra por detetor e gera um histograma por cada um.
Os plots são guardados em plots/energy/.
```

**Exemplo de um commit fraco (a evitar):**
```
update analysis   ❌
fix stuff         ❌
trabalho          ❌
```

## Dados de Simulação

Os ficheiros de dados (`.root`) **não estão no repositório git** — são demasiado grandes.
Ver as instruções em [`data/README.md`](data/README.md) para obter os ficheiros e colocá-los na pasta correta.

## Pré-requisitos de Reprodutibilidade

### Dependências Python

```bash
pip install -r requirements.txt
```

As versões mínimas estão especificadas em `requirements.txt`. Se houver problemas
de compatibilidade, confirmar as versões instaladas com `pip list`.

### ROOT (para `energy_deposition.py`)

O script `energy_deposition.py` usa a API C++ do ROOT via PyROOT.  
Requer ROOT 6.x instalado e acessível no PATH. Confirmar com:

```bash
root --version
python -c "import ROOT; print(ROOT.__version__)"
```

A instalação recomendada é via `conda`:
```bash
conda install -c conda-forge root
```

### Ficheiros de dados ROOT

Os ficheiros `.root` **não estão no repositório** (demasiado grandes).  
Devem ser colocados na pasta `data/` com os nomes exactos:

```
data/AmberTarget_Run_0.root
data/AmberTarget_Run_1.root
data/AmberTarget_Run_2.root
data/AmberTarget_Run_3.root
```

Ver [`data/README.md`](data/README.md) para instruções de obtenção.

> Os scripts verificam automaticamente a existência dos ficheiros e terminam com
> mensagem de erro clara se algum estiver em falta.

## Como Executar a Análise

### Confirmar que os dados estão presentes

```bash
ls -lh data/*.root
```

Devem aparecer os 4 ficheiros `AmberTarget_Run_*.root`.

### Opção 1 — Pipeline completo (recomendado)

```bash
python run_all.py
```

Executa todos os scripts pela ordem correcta. Os scripts ainda não implementados
pelos colegas são saltados automaticamente sem errar.

### Opção 2 — Só a análise de energia + ML (secção da Diana)

```bash
python run_all.py --energy   # deposição de energia (ROOT/PyROOT)
python run_all.py --ml       # classificador ML (uproot + scikit-learn)
```

### Opção 3 — Script individual

```bash
python analysis/energy_deposition.py   # deposição de energia
python analysis/particle_id_ml.py      # identificação por ML
```

Os gráficos são guardados automaticamente em `plots/energy/`.

### Compilar o relatório LaTeX

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # segunda execução para referências cruzadas
```

Ou usar um editor LaTeX (Overleaf, TeXstudio, VS Code + LaTeX Workshop).  
Para Overleaf: ver as instruções na raiz do repositório sobre a estrutura do ZIP.

---

## Análises Implementadas

As análises seguem o guia do trabalho prático. Os scripts estão em `analysis/`:

| Script | Secção | Responsável | Estado |
|--------|--------|-------------|--------|
| `energy_deposition.py` | 3.1 — Deposição de energia | Diana | ✅ Completo |
| `particle_id_ml.py` | 3.6 — Identificação por ML | Diana | ✅ Completo |
| `hit_distribution.py` | 3.2 — Distribuição de hits | (colega) | ⏳ Por implementar |
| `hadronic_vertex.py` | 3.3 — Vértices hadrónicos | (colega) | ⏳ Por implementar |
| `temporal_distribution.py` | 3.4 — Distribuição temporal | (colega) | ⏳ Por implementar |
| `momentum_distribution.py` | 3.5 — Distribuição do momento | José | ⏳ Por implementar |
