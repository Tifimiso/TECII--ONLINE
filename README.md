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
| _(a preencher)_ | _(a preencher)_ |
| _(a preencher)_ | _(a preencher)_ |

## Estrutura do Repositório

Ver [STRUCTURE.md](STRUCTURE.md) para a descrição completa de todas as pastas e ficheiros.

## Regras de Trabalho em Git

> **Cada membro do grupo deve trabalhar na sua própria branch.**
> Ninguém deve fazer commits diretamente na branch `main`.
> As contribuições são integradas em `main` via Pull Request, depois de revisão.

### Convenção de Commits

Os commits devem seguir o formato abaixo para que o histórico seja claro e profissional.
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

## Instalação de Dependências

```bash
pip install -r requirements.txt
```

## Análises Implementadas

As análises seguem o guia do trabalho prático. Os scripts estão em `analysis/`:

| Script | Análise |
|--------|---------|
| `energy_deposition.py` | Deposição de energia por detetor e por partícula |
| `hit_distribution.py` | Distribuição de hits em X e Y por detetor e carga |
| `hadronic_vertex.py` | Vértices hadrónicos primários e secundários em Z |
| `temporal_distribution.py` | Distribuição temporal dos hits por detetor |
| `momentum_distribution.py` | Momento em Z para muões, piões e piões primários/secundários |
