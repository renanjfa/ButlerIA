import json, pathlib, ast, tempfile
import pandas as pd
import joblib
from train_models import train
cells=[{'cell_type':'markdown','metadata':{},'source':['# Comparar Naive Bayes e XGBoost\nExecute com a base real em data/ai_human.csv.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['from train_models import train\nresults = train("data/ai_human.csv", "models")']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['import pandas as pd\npd.DataFrame({name: {"acuracia": results[name]["accuracy"], "roc_auc": results[name]["roc_auc"]} for name in ("naive_bayes", "xgboost")}).T']}]
pathlib.Path('comparar_modelos.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=2),encoding='utf-8')
for filename in ('naive_bayes.ipynb','comparar_modelos.ipynb','xgboost.ipynb'):
    notebook=json.loads(pathlib.Path(filename).read_text(encoding='utf-8-sig'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    print('Sintaxe OK:',filename)
with tempfile.TemporaryDirectory() as folder:
    base=pathlib.Path(folder)
    rows=[{'text':f'{prefix} example number {i}', 'generated':label} for label,prefix in [(0,'personal memory informal diary'),(1,'structured automated generated explanation')] for i in range(30)]
    pd.DataFrame(rows).to_csv(base/'synthetic.csv',index=False)
    result=train(base/'synthetic.csv',base/'models')
    for name in ('naive_bayes','xgboost_text'):
        artifact=joblib.load(base/'models'/f'{name}.joblib')
        assert artifact['model'].predict_proba(artifact['vectorizer'].transform(['personal diary example'])).shape==(1,2)
    assert result['split']=={'train':36,'validation':12,'test':12,'seed':42}
    print('SMOKE TEST OK: ambos treinados, exportados e recarregados; dados sinteticos descartados.')
