# Estrutura do Repositório

```
projeto/
│
├── data/                        # Dados de simulação (ficheiros ROOT, CSV, etc.)
│   └── .gitkeep                 # Pasta vazia mantida no git (dados podem ser grandes)
│
├── analysis/                    # Scripts de análise em Python
│   ├── energy_deposition.py     # Histogramas de deposição de energia por detetor e partícula
│   ├── hit_distribution.py      # Distribuição de hits em X e Y por detetor e tipo de partícula
│   ├── hadronic_vertex.py       # Vértices hadrónicos primários e secundários em função de Z
│   ├── temporal_distribution.py # Distribuição temporal dos hits por detetor
│   └── momentum_distribution.py # Distribuição do momento em Z para muões, piões e secundários
│
├── plots/                       # Gráficos gerados pelos scripts de análise
│   ├── energy/                  # Gráficos de deposição de energia
│   ├── hits/                    # Gráficos de distribuição de hits
│   ├── vertex/                  # Gráficos de vértices hadrónicos
│   └── momentum/                # Gráficos de distribuição do momento
│
├── report/                      # Relatório do trabalho
│   ├── report.tex               # Fonte LaTeX do relatório
│   └── report.pdf               # Relatório compilado (gerado a partir do .tex)
│
├── notebooks/                   # Jupyter Notebooks para exploração e prototipagem
│
├── requirements.txt             # Dependências Python (numpy, matplotlib, uproot, etc.)
├── STRUCTURE.md                 # Este ficheiro — descrição da estrutura do repositório
└── README.md                    # Descrição geral do projeto
```

## Descrição das pastas

### `data/`
Contém os ficheiros de dados da simulação. Devido ao tamanho potencialmente grande,
os dados **não devem ser adicionados ao git**. Apenas o `.gitkeep` é versionado.

### `analysis/`
Scripts Python com as análises pedidas no guia do trabalho prático:
- **energy_deposition.py** — deposição de energia nos detetores, por partícula e total.
- **hit_distribution.py** — distribuição espacial (X, Y) dos hits por detetor e carga.
- **hadronic_vertex.py** — posição em Z dos vértices hadrónicos primários e secundários.
- **temporal_distribution.py** — distribuição temporal dos hits em cada detetor.
- **momentum_distribution.py** — momento em Z para muões, piões e piões primários/secundários.

### `plots/`
Gráficos gerados automaticamente pelos scripts de análise, organizados por tema.

### `report/`
Relatório final em LaTeX. O ficheiro `report.pdf` é gerado ao compilar o `report.tex`.

### `notebooks/`
Jupyter Notebooks para exploração inicial dos dados e prototipagem de análises.

### `requirements.txt`
Lista de dependências Python necessárias para correr os scripts de análise.
Instalar com: `pip install -r requirements.txt`
