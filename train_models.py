"""Treina Naive Bayes e XGBoost na mesma divisão de textos."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier


def train(data_path, output_dir):
    data_path, output_dir = Path(data_path), Path(output_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f'CSV não encontrado: {data_path.resolve()}')
    print('Lendo base...', flush=True)
    df = pd.read_csv(data_path, usecols=['text', 'generated']).dropna().copy()
    raw_rows = len(df)
    df['text'] = df['text'].astype(str).str.strip()
    df = df.loc[df.text.ne('')].copy()
    df['generated'] = pd.to_numeric(df.generated, errors='raise')
    if not df.generated.isin([0, 1]).all():
        raise ValueError('generated deve conter 0 (humano) ou 1 (IA).')
    df['generated'] = df.generated.astype(int)
    df['_key'] = pd.util.hash_pandas_object(
        df.text.str.lower().str.replace(r'\s+', ' ', regex=True), index=False)
    conflicts = df.groupby('_key').generated.transform('nunique').gt(1)
    df = df.loc[~conflicts].drop_duplicates('_key').reset_index(drop=True)
    if set(df.generated.unique()) != {0, 1} or df.generated.value_counts().min() < 10:
        raise ValueError('São necessários pelo menos 10 textos distintos de cada classe.')
    dev, test = train_test_split(df.index, test_size=.2, stratify=df.generated, random_state=42)
    train_idx, val = train_test_split(dev, test_size=.25, stratify=df.loc[dev, 'generated'], random_state=42)
    print(f'Base limpa: {len(df)}; treino: {len(train_idx)}; validacao: {len(val)}; teste: {len(test)}', flush=True)
    # Descoberta limitada ao treino para evitar um dicionário temporário gigante.
    vocabulary_rows = min(10000, len(train_idx))
    vocabulary_sample = df.loc[train_idx, 'text'].sample(n=vocabulary_rows, random_state=42)
    selector = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2)
    selector.fit(vocabulary_sample)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000,
                                 vocabulary=selector.vocabulary_,
                                 min_df=2, sublinear_tf=True, dtype=np.float32)
    print('Calculando TF-IDF com vocabulario limitado ao treino...', flush=True)
    x_train = vectorizer.fit_transform(df.loc[train_idx, 'text'])
    x_val = vectorizer.transform(df.loc[val, 'text'])
    x_test = vectorizer.transform(df.loc[test, 'text'])
    print(f'TF-IDF pronto: {x_train.shape}', flush=True)
    y_train, y_val, y_test = (df.loc[idx, 'generated'] for idx in (train_idx, val, test))
    # Escolha de alpha somente na validação, mantendo o teste reservado.
    candidates = []
    for alpha in (.1, .5, 1., 2.):
        candidate = MultinomialNB(alpha=alpha).fit(x_train, y_train)
        candidates.append((roc_auc_score(y_val, candidate.predict_proba(x_val)[:, 1]), candidate))
    nb = max(candidates, key=lambda item: item[0])[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'vectorizer': vectorizer, 'model': nb, 'threshold': .5,
                 'labels': {0: 'Humano', 1: 'IA'}}, output_dir / 'naive_bayes.joblib')
    print(f'Naive Bayes treinado; alpha={nb.alpha}', flush=True)
    xgb = XGBClassifier(n_estimators=200, max_depth=4, max_bin=64, learning_rate=.08,
                        subsample=.8, colsample_bytree=.8, objective='binary:logistic',
                        eval_metric='logloss', tree_method='hist', early_stopping_rounds=30,
                        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
                        n_jobs=8, random_state=42)
    xgb.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=10)
    joblib.dump({'vectorizer': vectorizer, 'model': xgb, 'threshold': .5,
                 'labels': {0: 'Humano', 1: 'IA'}}, output_dir / 'xgboost_text.joblib')
    results = {}
    for name, model in [('naive_bayes', nb), ('xgboost', xgb)]:
        score = model.predict_proba(x_test)[:, 1]
        pred = (score >= .5).astype(int)
        results[name] = {'accuracy': accuracy_score(y_test, pred),
                         'roc_auc': roc_auc_score(y_test, score),
                         'classification_report': classification_report(y_test, pred, labels=[0, 1],
                             target_names=['Humano', 'IA'], output_dict=True, zero_division=0),
                         'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1]).tolist()}
        print(name, json.dumps(results[name], ensure_ascii=False, indent=2), flush=True)
    results['split'] = {'train': len(train_idx), 'validation': len(val), 'test': len(test), 'seed': 42}
    results['selected_alpha'] = nb.alpha
    results['best_xgboost_iteration'] = xgb.best_iteration
    results['xgboost_parameters'] = xgb.get_params()
    results['dataset'] = {'path': str(data_path), 'non_null_rows': raw_rows,
                          'clean_rows': len(df), 'conflicting_rows': int(conflicts.sum()),
                          'vocabulary_sample_rows': vocabulary_rows,
                          'features': len(vectorizer.vocabulary_),
                          'class_counts': {str(k): int(v) for k, v in df.generated.value_counts().items()}}
    (output_dir / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    # Guarda as posições na base limpa para reproduzir a comparação.
    np.savez(output_dir / 'split_indices.npz', train=train_idx, validation=val, test=test)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/ai_human.csv')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()
    train(args.data, args.output)
