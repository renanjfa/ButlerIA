# Resultados: Naive Bayes e XGBoost

Base: [AI vs Human Text, Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text).

| Modelo | Acurácia | Precisão IA | Recall IA | F1 IA | F1 macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 95.8124% | 98.1789% | 90.8712% | 94.3838% | 95.5227% | 99.3629% |
| XGBoost | 98.6321% | 98.4385% | 98.0224% | 98.2300% | 98.5577% | 99.8841% |

## Divisão de dados

{'train': 278533, 'validation': 92845, 'test': 92845, 'seed': 42}

Textos após limpeza: 464,223.
Alpha do Naive Bayes: 0.1.
Melhor rodada do XGBoost (contagem iniciada em 1): 200.

Vocabulário de até 10.000 n-gramas escolhido em até 10.000 textos do treino, com IDF ajustado em todo o treino. Ambos os modelos usam os mesmos dados, características e limiar 0,5. A seleção de alpha e a parada antecipada usam somente a validação.

## Matrizes de confusão

Linhas = classe real; colunas = previsão. Ordem: humano, IA.

### naive_bayes

| Real / Previsto | Humano | IA |
|---|---:|---:|
| Humano | 56287 | 606 |
| IA | 3282 | 32670 |

### xgboost

| Real / Previsto | Humano | IA |
|---|---:|---:|
| Humano | 56334 | 559 |
| IA | 711 | 35241 |

## Limites

O teste é uma divisão aleatória da mesma base. Não comprova generalização para outras fontes, modelos geradores, temas ou idiomas. Foram removidas duplicatas normalizadas e conflitos de rótulo; paráfrases e textos da mesma origem ainda podem compartilhar características entre divisões. As pontuações não são probabilidades calibradas nem comprovam autoria.