# Guia de Git — Passo a Passo

Este ficheiro explica exatamente o que tens de fazer no git, desde o início até entregar o teu trabalho. Segue os passos por ordem.

---

## PASSO 1 — Copiar o repositório para o teu computador (só na primeira vez)

Abre o terminal e escreve:

```bash
git clone https://github.com/Tifimiso/TECII--ONLINE.git
```

Depois entra na pasta:

```bash
cd TECII--ONLINE
```

---

## PASSO 2 — Entrar na tua branch (só na primeira vez)

Cada pessoa tem a sua própria branch. Substitui `NomeAqui` pelo teu nome:

```bash
git checkout -b NomeAqui
```

Por exemplo:
- Tiago → `git checkout -b Tiago`
- Rafael → `git checkout -b Rafael`
- José → `git checkout -b Jose`

Se a branch já existir no GitHub, usa:

```bash
git checkout NomeAqui
```

---

## PASSO 3 — Verificar em que branch estás (sempre que abrires o terminal)

```bash
git branch
```

A branch onde estás aparece com um `*` à frente. **Nunca trabalhes se o `*` estiver no `main`.**

---

## PASSO 4 — Atualizar o teu repositório com as últimas alterações

Antes de começar a trabalhar, busca sempre as últimas alterações:

```bash
git pull origin main
```

---

## PASSO 5 — Trabalhar nos ficheiros

Abre os ficheiros que são da tua responsabilidade (ver divisão de tarefas na mensagem do grupo) e faz as alterações.

Os scripts estão em `analysis/` e o relatório está em `report/report.tex`.

---

## PASSO 6 — Guardar o trabalho (fazer um commit)

Depois de fazeres alterações, guarda-as no git. Faz isto regularmente — **não esperes ter tudo feito para fazer o primeiro commit.**

**Ver o que alteraste:**
```bash
git status
```

**Adicionar os ficheiros que queres guardar:**
```bash
git add nome_do_ficheiro.py
```

Por exemplo:
```bash
git add analysis/hadronic_vertex.py
```

**Criar o commit com uma mensagem descritiva:**
```bash
git commit -m "feat: implementar histograma de vértices hadrónicos"
```

> Atenção: a mensagem do commit tem de ser clara e descritiva. Lê a secção "Convenção de Commits" no README.md — o professor avalia isto individualmente.

---

## PASSO 7 — Enviar o trabalho para o GitHub

```bash
git push origin NomeAqui
```

Por exemplo:
```bash
git push origin Tiago
```

Faz isto sempre que fizeres um ou mais commits, para o trabalho ficar guardado online.

---

## PASSO 8 — Criar um Pull Request (quando acabares a tua parte)

Quando tiveres tudo feito e quereres juntar o teu trabalho ao `main`:

1. Vai ao repositório no GitHub: [https://github.com/Tifimiso/TECII--ONLINE](https://github.com/Tifimiso/TECII--ONLINE)
2. Clica no separador **"Pull requests"**
3. Clica em **"New pull request"**
4. Em **"compare"** escolhe a tua branch (ex: `Tiago`)
5. Clica em **"Create pull request"**
6. Escreve um título e uma descrição do que fizeste
7. Clica em **"Create pull request"** novamente

Depois avisa no grupo para alguém fazer o merge.

---

## Resumo rápido do dia-a-dia

```bash
git branch                        # verificar em que branch estás
git pull origin main              # atualizar
# ... faz as tuas alterações ...
git add analysis/meu_script.py    # adicionar ficheiro
git commit -m "feat: descrição"   # guardar com mensagem
git push origin NomeAqui          # enviar para o GitHub
```

---

## Dúvidas?

Qualquer problema que tenham com o git ou com o código, falem no grupo. Estou disponível para ajudar.
