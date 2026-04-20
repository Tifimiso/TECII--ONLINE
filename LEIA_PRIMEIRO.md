# LEIA ISTO ANTES DE QUALQUER COISA

Olá a todos.

Este ficheiro serve para explicar o que é o trabalho, o que cada um tem de fazer, e como devemos trabalhar juntos.

---

## O que é o trabalho?

Temos dados de uma simulação de física (ficheiros `.root`) e o nosso trabalho é **analisar esses dados e produzir gráficos** — histogramas, distribuições, etc. — e depois escrever um relatório com os resultados.

**Não é um trabalho complexo.** Os scripts já estão criados e estruturados, só precisam de ser preenchidos com o código de análise. Somos 4 pessoas e o trabalho está dividido em partes iguais. Não há desculpa para não estar feito rapidamente.

---

## OBRIGATÓRIO — Ler antes de tocar em qualquer ficheiro

Antes de fazer seja o que for, **leiam com atenção**:

1. [`README.md`](README.md) — descrição do projeto, regras de trabalho em git e como fazer commits
2. [`STRUCTURE.md`](STRUCTURE.md) — onde está cada ficheiro e para que serve cada pasta

Não saltem estes ficheiros. Estão escritos de forma simples e explicam tudo o que precisam de saber.

---

## Divisão do Trabalho

Dividi o trabalho da seguinte forma. Se alguém preferir fazer uma parte diferente, é só dizer e trocamos — mas avisem com antecedência.

| Pessoa | Script a implementar | Secção do relatório |
|--------|----------------------|---------------------|
| **Diana** | `analysis/energy_deposition.py` | Introdução, Conclusão e secção de deposição de energia |
| **Tiago** | `analysis/hadronic_vertex.py` | Secção de vértices hadrónicos |
| **Rafael** | `analysis/hit_distribution.py` + `analysis/temporal_distribution.py` | Secções de hits e distribuição temporal |
| **José** | `analysis/momentum_distribution.py` | Secção de momento |

---

## O que cada um tem de fazer concretamente

### 1. Implementar o script de análise
O script já está criado em `analysis/`. Só está vazio (as funções têm `pass`).
Têm de preencher as funções com código Python que:
- Leia os dados do ficheiro `.root` que está em `data/`
- Produza os gráficos pedidos
- Guarde os gráficos na pasta `plots/` correspondente

### 2. Escrever a secção do relatório
Em `report/report.tex` está o ficheiro LaTeX do relatório.
Cada um escreve a secção correspondente à sua análise — com os gráficos gerados e uma breve explicação dos resultados.

### 3. Fazer commits à medida que trabalham
Não façam tudo de uma vez no final. Façam commits regulares enquanto trabalham.
**Leiam a secção "Convenção de Commits" no README** — o professor avalia a qualidade dos commits individualmente.

---

## Como trabalhar no git

1. Trabalhem sempre na vossa própria branch (ex: `Tiago`, `Rafael`, `Jose`)
2. **Nunca façam commits diretamente no `main`**
3. Quando acabarem a vossa parte, criem um Pull Request para o `main`

Se não souberem como criar uma branch ou um Pull Request, perguntem.

---

## Dados de simulação

Os ficheiros `.root` com os dados **não estão no git** (são demasiado grandes).
Leiam [`data/README.md`](data/README.md) para saber como obter os ficheiros e onde os colocar.

---

Qualquer dúvida, falem.
