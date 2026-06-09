import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # Caminhos de arquivos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'data_students.csv')
    ml_dir = os.path.join(base_dir, 'ml')
    model_output_path = os.path.join(ml_dir, 'modelo_evasao.joblib')

    if not os.path.exists(ml_dir):
        os.makedirs(ml_dir)

    print("Carregando o dataset de estudantes...")
    # Lendo o CSV usando ponto-e-vírgula como separador
    df = pd.read_csv(data_path, sep=';')

    # Remove espaços em branco ou tabs nas colunas (como "Daytime/evening attendance\t")
    df.columns = df.columns.str.strip()

    # Separando Features (X) e Target (y)
    X = df.drop(columns=['Target'])
    y = df['Target']

    # Salva o nome das features para garantir ordem correta na inferência
    feature_names = X.columns.tolist()

    # Divisão de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Treinando o modelo HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=4,
        learning_rate=0.04,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Avaliação do modelo
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    
    metrics_str = (
        f"# Métricas de Avaliação do Modelo ML (HistGradientBoostingClassifier)\n\n"
        f"Abaixo estão as métricas calculadas na base de testes (20% do dataset):\n\n"
        f"| Métrica | Valor | Descrição |\n"
        f"|---|---|---|\n"
        f"| **Acurácia Geral (Accuracy)** | {accuracy:.4%} | Taxa geral de predições corretas. |\n"
        f"| **Precisão (Macro Average)** | {precision_macro:.4%} | Média simples de precisão por classe (relação de acertos positivos). |\n"
        f"| **Precisão (Weighted Average)** | {precision_weighted:.4%} | Precisão por classe ponderada pelo suporte de cada classe. |\n"
        f"| **Revocação / Sensibilidade (Macro Recall)** | {recall_macro:.4%} | Média simples de recall por classe (capacidade de achar positivos). |\n"
        f"| **Revocação / Sensibilidade (Weighted Recall)** | {recall_weighted:.4%} | Recall por classe ponderado pelo suporte. |\n"
        f"| **F1-Score (Macro Average)** | {f1_macro:.4%} | Média harmônica entre precisão e recall (macro). |\n"
        f"| **F1-Score (Weighted Average)** | {f1_weighted:.4%} | Média harmônica ponderada entre precisão e recall. |\n"
        f"| **ROC-AUC (Macro OVR)** | {roc_auc_ovr:.4%} | Área sob a curva ROC (One-vs-Rest) refletindo separabilidade. |\n\n"
        f"## Relatório de Classificação Detalhado:\n"
        f"```text\n"
        f"{classification_report(y_test, y_pred)}\n"
        f"```\n"
    )
    
    print("\n=== Resultados da Avaliação ===")
    print(f"Acurácia Geral: {accuracy:.4f}")
    print(f"Precisão (Macro): {precision_macro:.4f}")
    print(f"Revocação (Macro): {recall_macro:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"ROC-AUC (Macro OVR): {roc_auc_ovr:.4f}")
    
    # Salva o relatório no diretório ml
    report_path = os.path.join(ml_dir, 'metricas_modelo.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(metrics_str)
    print(f"\nRelatório de métricas salvo em: {report_path}")
    
    # Salvando o modelo e a lista de features juntas
    model_data = {
        'model': clf,
        'features': feature_names
    }
    joblib.dump(model_data, model_output_path)
    print(f"Modelo salvo com sucesso em: {model_output_path}")

if __name__ == "__main__":
    main()
