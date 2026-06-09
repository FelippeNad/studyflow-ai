# Métricas de Avaliação do Modelo ML (HistGradientBoostingClassifier)

Abaixo estão as métricas calculadas na base de testes (20% do dataset):

| Métrica | Valor | Descrição |
|---|---|---|
| **Acurácia Geral (Accuracy)** | 76.2712% | Taxa geral de predições corretas. |
| **Precisão (Macro Average)** | 71.4996% | Média simples de precisão por classe (relação de acertos positivos). |
| **Precisão (Weighted Average)** | 75.5474% | Precisão por classe ponderada pelo suporte de cada classe. |
| **Revocação / Sensibilidade (Macro Recall)** | 68.9943% | Média simples de recall por classe (capacidade de achar positivos). |
| **Revocação / Sensibilidade (Weighted Recall)** | 76.2712% | Recall por classe ponderado pelo suporte. |
| **F1-Score (Macro Average)** | 69.9092% | Média harmônica entre precisão e recall (macro). |
| **F1-Score (Weighted Average)** | 75.5795% | Média harmônica ponderada entre precisão e recall. |
| **ROC-AUC (Macro OVR)** | 88.8973% | Área sob a curva ROC (One-vs-Rest) refletindo separabilidade. |

## Relatório de Classificação Detalhado:
```text
              precision    recall  f1-score   support

     Dropout       0.81      0.72      0.76       284
    Enrolled       0.53      0.45      0.49       159
    Graduate       0.80      0.90      0.85       442

    accuracy                           0.76       885
   macro avg       0.71      0.69      0.70       885
weighted avg       0.76      0.76      0.76       885

```
