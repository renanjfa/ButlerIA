"""Interface local para os modelos treinados. Execute: python app.py."""
import argparse
import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent
if (ROOT / '.deps').is_dir():
    sys.path.insert(0, str(ROOT / '.deps'))
import joblib

ARTIFACTS = {}
NAMES = {'naive_bayes': 'Naive Bayes', 'xgboost': 'XGBoost'}
MAX_TEXT = 50000


def load_models():
    for key, filename in [('naive_bayes', 'naive_bayes.joblib'), ('xgboost', 'xgboost_text.joblib')]:
        ARTIFACTS[key] = joblib.load(ROOT / 'models' / filename)


def classify(text, selected):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Cole um texto antes de analisar.')
    if len(text) > MAX_TEXT:
        raise ValueError('O limite é de 50.000 caracteres por análise.')
    if selected not in (*NAMES, 'both'):
        raise ValueError('Selecione um modelo válido.')
    results = []
    for key in NAMES if selected == 'both' else [selected]:
        artifact = ARTIFACTS[key]
        features = artifact['vectorizer'].transform([text.strip()])
        if features.nnz == 0:
            results.append({'model': NAMES[key], 'label': None, 'score': None,
                            'message': 'Nenhum termo reconhecido pelo vocabulário. Tente um texto mais longo em inglês.'})
            continue
        score = float(artifact['model'].predict_proba(features)[0, 1])
        results.append({'model': NAMES[key], 'label': 'IA' if score >= artifact.get('threshold', .5) else 'Humano',
                        'score': score})
    return {'results': results, 'short_text': len(text.split()) < 50}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type='application/json; charset=utf-8'):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/':
            self.send(200, (ROOT / 'frontend' / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/health':
            self.send(200, {'ready': len(ARTIFACTS) == 2})
        else:
            self.send(404, {'error': 'Página não encontrada.'})

    def do_POST(self):
        if self.path != '/api/predict':
            self.send(404, {'error': 'Página não encontrada.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 400000:
                raise ValueError('Requisição vazia ou muito grande.')
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError('Envie um objeto com texto e modelo.')
            self.send(200, classify(payload.get('text'), payload.get('model', 'both')))
        except (ValueError, TypeError, UnicodeDecodeError):
            self.send(400, {'error': 'Texto ou modelo inválido. Use de 1 a 50.000 caracteres.'})
        except Exception:
            self.send(500, {'error': 'Não foi possível analisar o texto. Verifique o terminal.'})
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print('Carregando os modelos...', flush=True)
    load_models()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Abra http://127.0.0.1:{args.port} no navegador. Ctrl+C para encerrar.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
