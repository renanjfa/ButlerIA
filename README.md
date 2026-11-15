# ButlerIA

Reconhecedor experimental de textos humanos ou gerados por IA, usando TF-IDF com Naive Bayes e XGBoost.

## Abrir a interface

Nesta máquina, dê dois cliques em `iniciar.bat` na pasta do projeto:

```text
C:\Users\Pichau\Documents\ChatGPT\Pet\ButlerIA
```

Mantenha o terminal aberto e acesse http://127.0.0.1:8000. Cole um texto, selecione um modelo ou compare os dois e clique em **Analisar texto**. Para encerrar, pressione Ctrl+C no terminal.

A aplicação usa os modelos já treinados em `models/`, processa o texto localmente e não o salva. Prefira textos em inglês com pelo menos 50 palavras. As pontuações não são probabilidades calibradas e não comprovam autoria.

## Configurar em outra máquina

Execute os comandos na raiz deste repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-models.txt
```

Baixe a [base do Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text), extraia o CSV e coloque-o em `data/ai_human.csv`. As colunas esperadas são `text` e `generated`, com 0 para humano e 1 para IA. Os links de referência também estão em `links.txt`.

```powershell
.\.venv\Scripts\python train_models.py --data data/ai_human.csv --output models
.\.venv\Scripts\python report_results.py
.\.venv\Scripts\python app.py
```

O treinamento pode demorar e exige memória disponível. A interface requer `models/naive_bayes.joblib` e `models/xgboost_text.joblib`. Esses modelos estão presentes localmente, mas não são versionados. Carregue apenas arquivos joblib de origem confiável. Para usar outra porta: `python app.py --port 8001`.

## Estrutura

```text
ButlerIA/
  app.py                       Servidor local e inferência
  frontend/index.html          Interface no navegador
  iniciar.bat / iniciar.ps1    Inicialização no Windows
  train_models.py              Treinamento conjunto dos dois modelos
  report_results.py            Geração do relatório
  verify_training.py           Verificação com dados sintéticos temporários
  comparar_modelos.ipynb       Treinamento e comparação
  naive_bayes.ipynb            Experimento individual Naive Bayes
  xgboost.ipynb                Experimento individual XGBoost
  logistic_regression.ipynb    Experimento original
  eda.ipynb                   Análise exploratória
  requirements-models.txt     Dependências do treinamento e da interface
  requirements-xgboost.txt    Dependências do experimento XGBoost
  requirements.txt            Ambiente original do projeto
  dataset_metadata.json       Origem e SHA-256 da base usada
  RESULTADOS.md               Métricas reais e limitações
  data/                       CSV e ZIP locais, ignorados pelo Git
  models/                     Modelos e métricas gerados, ignorados pelo Git
  .deps/                      Dependências locais, ignoradas pelo Git
```

## Metodologia e resultados

O treinamento conjunto remove textos vazios, duplicatas normalizadas e conflitos de rótulos. Usa a mesma divisão estratificada de 60% treino, 20% validação e 20% teste para os dois modelos, com semente 42.

Para limitar a memória, o vocabulário de até 10.000 n-gramas de 1 a 3 palavras é selecionado em até 10.000 textos exclusivamente do treino. O IDF é ajustado em todo o treino. Os notebooks individuais selecionam o vocabulário em todo o treino; use `train_models.py` para reproduzir a comparação registrada.

O alpha do Naive Bayes é escolhido na validação. O XGBoost usa até 200 árvores, profundidade 4, `max_bin=64`, 8 threads e parada antecipada na validação. O teste permanece reservado para a avaliação final.

Os resultados completos estão em [RESULTADOS.md](RESULTADOS.md). A execução salva modelos, métricas, parâmetros e índices das divisões em `models/`. O log local fica em `training.log`.

O teste usa uma divisão da mesma base e não comprova generalização para outros temas, fontes, geradores ou idiomas. Nenhuma métrica do teste sintético representa desempenho na base real.

## Preparar o commit

Faça o commit dentro de `ButlerIA`, que é o repositório conectado ao GitHub. O `.gitignore` exclui datasets, modelos gerados, ambientes, caches, logs e a captura local da interface.

```powershell
cd "C:\Users\Pichau\Documents\ChatGPT\Pet\ButlerIA"
git status
git add .
git diff --cached --stat
git commit -m "Adiciona Naive Bayes, XGBoost e interface local"
```
