# Template para projetos do Observatório FIESC

Este repositório serve como um template padrão para iniciar novos projetos de Data Science. O objetivo é fornecer uma estrutura de diretórios organizada que promova a reprodutibilidade e facilite a colaboração.

## Estrutura de diretórios

A estrutura de pastas foi pensada para separar de forma lógica os diferentes componentes de um projeto de dados:

```
Título do Projeto
├── Dados/
│   ├── Processados   -> Armazene aqui os dados brutos e processados
│   └── brutos
├── Relatórios/
│   ├──              -> Guarde imagens e gráficos gerados/utilizados pelo projeto
│   └──  
├── Modelos/
│   ├──              -> Salve os modelos treinados
│   └──  
├── Notebooks/
│   ├──              -> Para prototipação e análises exploratórias
│   └── 
├── Scripts/
│   ├──              -> Scripts para ETL, treinamento, inferência, etc
│   └──
├── Referências/
│   ├──              -> Salve referências e documentações utilizadas no projeto
│   └──   
├── requirements.txt -> Lista de dependências do projeto
└── README.md        -> Este arquivo
```

*   **`Dados/`**: Contém os datasets. É uma boa prática ter subpastas como `Brutos/` para dados brutos e `Processados/` para dados limpos e transformados.
*   **`Relatórios/`**: Ideal para salvar visualizações, como gráficos e plots, que podem ser usados em relatórios ou apresentações.
*   **`Modelos/`**: Local para armazenar os artefatos de modelos treinados (ex: arquivos `.pkl`, `.h5`, `.joblib`).
*   **`Notebooks/`**: Jupyter Notebooks para experimentação. Tente manter os notebooks focados em etapas específicas (ex: `01-exploracao-dados.ipynb`).
*   **`Scripts/`**: Código Python modularizado. Funções de pré-processamento, classes de modelos e pipelines de execução devem ser colocados aqui para serem reutilizados.
*   **`Referências/`**: Local para armazenar referências e documentações utilizadas na elaboração do projeot.

## Como começar

Siga os passos abaixo para configurar o ambiente de desenvolvimento para um novo projeto usando este template.

O processo consiste em primeiro criar um novo repositório a partir deste template e depois configurar seu ambiente de desenvolvimento local.

### 1. Crie um novo repositório a partir deste template
Navegue até a página principal deste repositório no GitHub.

Clique no botão verde "Use this template" e selecione "Create a new repository".

Na página seguinte, escolha um nome para o seu novo repositório (ex: analise-de-mercado-2025), adicione uma descrição e decida se ele será público ou privado.

Clique em "Create repository from template". O GitHub criará um novo repositório na sua conta com toda a estrutura e arquivos deste template.

### 2. Clone o seu novo repositório
Agora, clone o repositório que você acabou de criar para a sua máquina local. Abra seu terminal, navegue até o diretório onde deseja salvar o projeto e execute o comando abaixo, substituindo a URL pela URL do seu novo repositório:

**No Windows (PowerShell/CMD/terminal do VSCode):**
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_SEU_NOVO_REPOSITORIO.git
```
Depois, entre na pasta do projeto:
```bash
cd NOME_DO_SEU_NOVO_REPOSITORIO
```

### 3. Configure o arquivo .gitignore
Antes de criar qualquer outro arquivo ou ambiente, é fundamental instruir o Git sobre o que ele não deve rastrear. Isso evita que arquivos desnecessários, como o ambiente virtual, chaves de api ou dados brutos pesados, sejam enviados para o GitHub.


### 4. Crie o ambiente virtual

É uma prática essencial criar um ambiente virtual para isolar as dependências do seu projeto. Abra o terminal na raiz do projeto e execute:

```bash
# Cria um ambiente virtual na pasta venv
py -m venv venv
```

### 5. Ative o ambiente virtual

**No Windows (PowerShell/CMD):**
```bash
.\venv\Scripts\activate
```

Após a ativação, o nome do ambiente (`.venv`) aparecerá no início da linha do seu terminal.
Exemplo:
```bash
(venv) PS C:\Users\gustavo.migliorini\Documents\data_science\dlab-SC>
```

### 6. Instale as dependências

Com o ambiente ativado, instale as bibliotecas necessárias para o projeto:

```bash
# Exemplo de instalação
py -m pip install pandas
```
Opcionalmente, pode-se criar um arquivo .txt (bloco de notas) com as bibliotecas de interesse e instalá-las assim: 

Por padrão esse Template já contém algumas dependências básicas, como pandas, numpy, matplotlib, scikit-learn, etc.

```bash
py -m pip install -r requirements.txt
```

### 7. Atualizando o repositório no GitHub
Depois de fazer alterações no seu projeto, como adicionar novos scripts, notebooks ou atualizar a lista de dependências, é importante enviar essas mudanças para o seu repositório no GitHub. Isso garante que seu trabalho esteja salvo na nuvem e acessível para colaboração.

O VSCode tem uma integração nativa com o Git que facilita muito esse processo através da sua interface gráfica na aba "Source Control" (Ctrl+Shift+G). No entanto, é fundamental entender os comandos que rodam por trás da interface.

Abra o terminal integrado do VS Code (Ctrl+') e siga os passos abaixo:

#### a. Verifique o status das suas alterações

Antes de tudo, veja quais arquivos foram modificados, adicionados ou excluídos desde o seu último commit.

```bash
git status
```

Este comando listará os arquivos em vermelho (não rastreados ou modificados, mas não preparados) e em verde (preparados para o commit).

#### b. Adicione os arquivos para o próximo "commit"

Para incluir as alterações desejadas no seu próximo pacote de atualizações (commit), use o comando git add. Para adicionar todos os arquivos modificados de uma vez, o comando é:

```bash
git add .
```

Ao executar git status novamente, você verá que os arquivos agora estão na área de "staging" (preparados para o commit) e listados em verde.

#### c. Crie um "commit"

O "commit" é como um ponto de salvamento na história do seu projeto. Ele agrupa as alterações que você preparou no passo anterior com uma mensagem descritiva.

```bash
git commit -m "Adiciona análise exploratória inicial dos dados"
```

#### d. Envie as alterações para o GitHub

Por fim, envie o seu commit do repositório local para o repositório remoto no GitHub.

```bash
git push
```

### Pré-requisitos

-   Python 3.10+ instalado
-   Git instalado
-   Github login