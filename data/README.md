# Dados de Simulação

Os ficheiros de dados da simulação Geant4 (formato `.root`) **não estão no repositório git**
porque são demasiado grandes para serem versionados.

## Como obter os dados

Os ficheiros de dados foram produzidos pelo Tiago e devem ser partilhados diretamente
entre os membros do grupo (por exemplo, via pen drive, Google Drive, ou outro meio acordado).

## Ficheiros necessários

Coloca os seguintes ficheiros nesta pasta (`data/`) antes de correr os scripts de análise:

- `AmberTarget_Run_0.root` — ficheiro principal de saída da simulação

## Nota para o grupo

Depois de todos terem os ficheiros `.root` nas suas máquinas, abrir o ficheiro `.gitignore`
na raiz do projeto e seguir as instruções nos comentários para ativar a regra que ignora
os ficheiros `Zone.Identifier` (artefactos criados pelo Windows).
