import os
import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance import HealthInsurance

# Caminho do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Carrega modelo
model_path = os.path.join(project_root, 'src', 'models', 'model_linear_regression.pkl')
model = pickle.load(open(model_path, 'rb'))

# Inicializa API
app = Flask(__name__)

@app.route('/heathinsurance/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json:
        if isinstance(test_json, dict):  # único registro
            df_raw = pd.DataFrame([test_json])
        else:  # múltiplos registros
            df_raw = pd.DataFrame(test_json)

        # Instancia pipeline
        pipeline = HealthInsurance()

        # Pré-processamento
        df_processed = pipeline.preprocess(df_raw)

        # Predição
        df_response = pipeline.predict(model, df_raw, df_processed)

        return df_response

    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
