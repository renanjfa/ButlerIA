import json
from pathlib import Path

root = Path(__file__).resolve().parent
metrics = json.loads((root / 'models/metrics.json').read_text(encoding='utf-8'))
lines = ['# Resultados: Naive Bayes e XGBoost', '',
         'Base: [AI vs Human Text, Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text).', '',
         '| Modelo | Acurácia | Precisão IA | Recall IA | F1 IA | F1 macro | ROC-AUC |',
         '|---|---:|---:|---:|---:|---:|---:|']
for name, label in [('naive_bayes','Naive Bayes'), ('xgboost','XGBoost')]:
    item = metrics[name]
    report = item['classification_report']
    values = [item['accuracy'], report['IA']['precision'], report['IA']['recall'],
              report['IA']['f1-score'], report['macro avg']['f1-score'], item['roc_auc']]
    lines.append('| ' + label + ' | ' + ' | '.join(f'{x:.4%}' for x in values) + ' |')
lines.extend(['', '## Divisão de dados', '', str(metrics['split']), '',
              f"Textos após limpeza: {metrics['dataset']['clean_rows']:,}.",
              f"Alpha do Naive Bayes: {metrics['selected_alpha']}.",
              f"Melhor rodada do XGBoost (contagem iniciada em 1): {metrics['best_xgboost_iteration'] + 1}.", '',
              'Vocabulário de até 10.000 n-gramas escolhido em até 10.000 textos do treino, com IDF ajustado em todo o treino. '
              'Ambos os modelos usam os mesmos dados, características e limiar 0,5. '
              'A seleção de alpha e a parada antecipada usam somente a validação.', '',
              '## Matrizes de confusão', '',
              'Linhas = classe real; colunas = previsão. Ordem: humano, IA.', ''])
for name in ('naive_bayes','xgboost'):
    matrix=metrics[name]['confusion_matrix']
    lines.extend([f'### {name}', '', '| Real / Previsto | Humano | IA |', '|---|---:|---:|',
                  f'| Humano | {matrix[0][0]} | {matrix[0][1]} |',
                  f'| IA | {matrix[1][0]} | {matrix[1][1]} |', ''])
lines.extend(['## Limites', '',
              'O teste é uma divisão aleatória da mesma base. Não comprova generalização para outras fontes, '
              'modelos geradores, temas ou idiomas. Foram removidas duplicatas normalizadas e conflitos de rótulo; '
              'paráfrases e textos da mesma origem ainda podem compartilhar características entre divisões. '
              'As pontuações não são probabilidades calibradas nem comprovam autoria.'])
(root / 'RESULTADOS.md').write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines[:10]))
