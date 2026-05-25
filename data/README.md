# Dados de Simulação

Os ficheiros de dados da simulação Geant4 (formato `.root`) **não estão no repositório git**
porque são demasiado grandes para serem versionados.

## Como obter os dados

Os ficheiros de dados encontram-se no computador de laboratório:

```
ComputerLab12: /media/scratch/CarlosScratch/AMBER/GEANT4/detectorsWithHole-CorrectSize
```

Podem ser copiados diretamente a partir daí (pen drive, `scp`, etc.).

## Ficheiros necessários

Coloca os seguintes ficheiros nesta pasta (`data/`) antes de correr os scripts de análise:

| Ficheiro | Descrição |
|---|---|
| `AmberTarget_Run_0.root` | Run 0 da simulação AMBER Target |
| `AmberTarget_Run_1.root` | Run 1 da simulação AMBER Target |
| `AmberTarget_Run_2.root` | Run 2 da simulação AMBER Target |
| `AmberTarget_Run_3.root` | Run 3 da simulação AMBER Target |

O ficheiro `Analise.root` é gerado automaticamente pelos macros ROOT e pelos scripts de análise — não precisa de ser copiado.

## TTrees disponíveis

Os ficheiros ROOT contêm pelo menos dois TTrees principais:

- `edep_Per_Event` — deposição de energia por evento, com branches `detector0`, `detector1`, etc.
- `tracksData` — dados por track, com branches como `EdepDet0_keV`, `particlePDG`, `momentum_GeV`, etc.
