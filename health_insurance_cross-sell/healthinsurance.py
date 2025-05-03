import pickle
import os
import pandas as pd

class HealthInsurance:

    def __init__(self):
        # Caminho base do projeto
        self.home_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Carregando encoders e scalers
        self.gender_encoder = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'target_encode_gender_scaler.pkl'), 'rb'))
        self.region_encoder = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'target_encode_region_code_scaler.pkl'), 'rb'))
        self.channel_encoder = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'fe_policy_sales_channel_scaler.pkl'), 'rb'))
        
        self.age_scaler = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'age_scaler.pkl'), 'rb'))
        self.premium_scaler = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'annual_premium_scaler.pkl'), 'rb'))
        self.vintage_scaler = pickle.load(open(os.path.join(self.home_path, 'src', 'features', 'vintage_scaler.pkl'), 'rb'))

    def preprocess(self, df):
        # Aplicar encoders
        df['gender'] = df['gender'].map(self.gender_encoder)
        df['region_code'] = df['region_code'].map(self.region_encoder).fillna(0)
        df['policy_sales_channel'] = df['policy_sales_channel'].map(self.channel_encoder)

        # Aplicar scalers
        df['age'] = self.age_scaler.transform(df[['age']])
        df['annual_premium'] = self.premium_scaler.transform(df[['annual_premium']])
        df['vintage'] = self.vintage_scaler.transform(df[['vintage']])
        # One-hot encoding
        df = pd.get_dummies(df, prefix='vehicle_age', columns=['vehicle_age'])

        # Garante que todas as colunas necessárias existam
        required_cols = [
            'annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage',
            'previously_insured', 'policy_sales_channel',
            'vehicle_age_mais_de_2_anos', 'vehicle_age_entre_1_e_2_anos', 'vehicle_age_menos_de_1_ano'
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        return df[required_cols]

    def predict(self, model, original_data, test_data):
        # Caminho para as features selecionadas
        features_path = os.path.join(self.home_path, 'src', 'models', 'selected_features.pkl')

        # Carrega as features utilizadas no treinamento
        with open(features_path, 'rb') as file:
            selected_features = pickle.load(file)

        # Seleciona apenas as colunas esperadas
        test_data = test_data[selected_features]

        # Faz a predição
        pred = model.predict_proba(test_data)[:, 1]

        # Adiciona a predição ao dataset original
        original_data['prediction'] = pred
        return original_data.to_json(orient='records', date_format='iso')
