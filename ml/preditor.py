import os
import pandas as pd
import joblib

# Cache local para evitar carregar o modelo múltiplas vezes
_MODEL_CACHE = None
_DEFAULTS_CACHE = None

def get_model_and_defaults():
    global _MODEL_CACHE, _DEFAULTS_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _DEFAULTS_CACHE

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'ml', 'modelo_evasao.joblib')
    data_path = os.path.join(base_dir, 'data', 'data_students.csv')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado. Execute o treinamento primeiro. ({model_path})")

    # Carrega o modelo treinado
    _MODEL_CACHE = joblib.load(model_path)

    # Carrega os valores padrão (medianas) das colunas do dataset para preencher dados ausentes
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, sep=';')
        df.columns = df.columns.str.strip()
        X = df.drop(columns=['Target'])
        _DEFAULTS_CACHE = X.median().to_dict()
    else:
        # Defaults aproximados caso o CSV não esteja presente
        _DEFAULTS_CACHE = {
            'Marital status': 1.0, 'Application mode': 1.0, 'Application order': 1.0, 'Course': 1.0,
            'Daytime/evening attendance': 1.0, 'Previous qualification': 1.0, 'Previous qualification (grade)': 125.0,
            'Nacionality': 1.0, "Mother's qualification": 1.0, "Father's qualification": 1.0,
            "Mother's occupation": 1.0, "Father's occupation": 1.0, 'Admission grade': 120.0,
            'Displaced': 1.0, 'Educational special needs': 0.0, 'Debtor': 0.0, 'Tuition fees up to date': 1.0,
            'Gender': 0.0, 'Scholarship holder': 0.0, 'Age at enrollment': 20.0, 'International': 0.0,
            'Curricular units 1st sem (credited)': 0.0, 'Curricular units 1st sem (enrolled)': 6.0,
            'Curricular units 1st sem (evaluations)': 6.0, 'Curricular units 1st sem (approved)': 5.0,
            'Curricular units 1st sem (grade)': 12.0, 'Curricular units 1st sem (without evaluations)': 0.0,
            'Curricular units 2nd sem (credited)': 0.0, 'Curricular units 2nd sem (enrolled)': 6.0,
            'Curricular units 2nd sem (evaluations)': 6.0, 'Curricular units 2nd sem (approved)': 5.0,
            'Curricular units 2nd sem (grade)': 12.0, 'Curricular units 2nd sem (without evaluations)': 0.0,
            'Unemployment rate': 11.0, 'Inflation rate': 1.4, 'GDP': 0.5
        }

    return _MODEL_CACHE, _DEFAULTS_CACHE

def prever_risco_evasao(dados_aluno: dict) -> dict:
    """
    Recebe um dicionário com informações parciais do aluno e retorna a predição e probabilidades.
    Exemplo de entrada:
    {
        'Tuition fees up to date': 1,  # 1 = Sim, 0 = Não
        'Debtor': 0,                   # 1 = Sim, 0 = Não
        'Scholarship holder': 1,       # 1 = Sim, 0 = Não
        'Curricular units 1st sem (approved)': 4,
        'Curricular units 1st sem (grade)': 11.5,
        'Curricular units 2nd sem (approved)': 3,
        'Curricular units 2nd sem (grade)': 10.2
    }
    """
    model_data, defaults = get_model_and_defaults()
    clf = model_data['model']
    features = model_data['features']

    # Inicializa o dicionário de features com os valores default
    input_data = defaults.copy()

    # Sobrescreve com os dados reais passados do aluno
    for key, value in dados_aluno.items():
        # Trata possíveis pequenas diferenças de nomenclatura (como 'Tuition fees up to date' vs 'tuition_fees')
        normalized_key = key.replace('_', ' ').strip()
        matched_key = None
        for feat in features:
            if feat.lower() == normalized_key.lower():
                matched_key = feat
                break
        
        if matched_key:
            input_data[matched_key] = float(value)

    # Converte para DataFrame com a ordem de features exata do treino
    df_input = pd.DataFrame([input_data])[features]

    # Predição e probabilidade
    prediction = clf.predict(df_input)[0]
    probabilities = clf.predict_proba(df_input)[0]
    class_probs = {clf.classes_[i]: float(probabilities[i]) for i in range(len(clf.classes_))}

    # Calcula um nível de risco qualitativo baseado na probabilidade de Dropout
    dropout_prob = class_probs.get('Dropout', 0.0)
    if dropout_prob > 0.5:
        risk_level = "ALTO"
    elif dropout_prob > 0.25:
        risk_level = "MÉDIO"
    else:
        risk_level = "BAIXO"

    return {
        'prediction': prediction,
        'dropout_probability': dropout_prob,
        'risk_level': risk_level,
        'probabilities': class_probs
    }

if __name__ == "__main__":
    # Teste rápido de inferência
    print("Testando preditor...")
    aluno_bom = {
        'Tuition fees up to date': 1,
        'Debtor': 0,
        'Scholarship holder': 1,
        'Curricular units 1st sem (approved)': 6,
        'Curricular units 1st sem (grade)': 14.5,
        'Curricular units 2nd sem (approved)': 6,
        'Curricular units 2nd sem (grade)': 15.0
    }
    aluno_em_risco = {
        'Tuition fees up to date': 0,
        'Debtor': 1,
        'Scholarship holder': 0,
        'Curricular units 1st sem (approved)': 0,
        'Curricular units 1st sem (grade)': 0.0,
        'Curricular units 2nd sem (approved)': 0,
        'Curricular units 2nd sem (grade)': 0.0
    }

    res_bom = prever_risco_evasao(aluno_bom)
    res_risco = prever_risco_evasao(aluno_em_risco)

    print(f"\nAluno Bom: Predição={res_bom['prediction']}, Risco={res_bom['risk_level']}, Probabilidade de Dropout={res_bom['dropout_probability']:.2%}")
    print(f"Aluno em Risco: Predição={res_risco['prediction']}, Risco={res_risco['risk_level']}, Probabilidade de Dropout={res_risco['dropout_probability']:.2%}")
