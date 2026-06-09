# Relatório Técnico do Projeto: StudyFlow AI 🎓

Este relatório apresenta os resultados obtidos na implementação das camadas de inteligência do assistente acadêmico **StudyFlow AI**, abrangendo a modelagem preditiva de evasão, o design da arquitetura multi-agente, a infraestrutura de observabilidade e a suíte de testes de fidelidade de respostas.

---

## 📊 1. Dataset Utilizado para Machine Learning

O modelo preditivo de inteligência artificial do projeto foi treinado com o dataset **"Predict Students' Dropout and Academic Success"**.

* **Nome**: Predict Students' Dropout and Academic Success.
* **Fonte**: UC Irvine Machine Learning Repository (doado pelo Instituto Politécnico de Portalegre, Portugal).
* **Número de Registros**: 4.424 registros (estudantes) com 36 atributos socioeconômicos, demográficos e acadêmicos.
* **Variável-Alvo (Target)**: `Target` (variável categórica com três classes possíveis: `Dropout` - Evasão, `Enrolled` - Matriculado, e `Graduate` - Graduado).

### Conexão com o Domínio
A evasão no ensino superior é um problema complexo que impacta tanto a vida do aluno quanto a sustentabilidade financeira da instituição. O assistente **StudyFlow AI** conecta-se diretamente a este domínio ao atuar como um canal preventivo: o tutor ou o próprio aluno pode interagir com o chat e, através de perguntas simples, o agente Mentor executa o modelo de ML informando a probabilidade de desistência baseada em fatores como idade de ingresso, inadimplência e notas dos semestres iniciais. Isso permite à faculdade realizar intervenções preventivas direcionadas.

---

## 📈 2. Métricas do Modelo de Machine Learning

O classificador foi treinado utilizando o algoritmo **`HistGradientBoostingClassifier`** (Gradient Boosting baseado em histogramas) sobre a base histórica de estudantes (`data_students.csv`). A avaliação de generalização foi realizada em um conjunto de testes isolado (20% do dataset original).

### Métricas de Teste Obtidas:

| Métrica | Valor | Interpretação Técnica |
|---|---|---|
| **Acurácia Geral (Accuracy)** | **76.27%** | Indica que mais de 76% de todas as predições de status (`Dropout`, `Enrolled` ou `Graduate`) na base de teste estão corretas. |
| **Precisão (Macro Precision)** | **71.50%** | Média das taxas de acerto positivo de cada classe. Mede a confiabilidade de uma predição positiva de status. |
| **Revocação / Sensibilidade (Macro Recall)** | **68.99%** | Média do percentual de instâncias reais identificadas pelo modelo para cada uma das classes. |
| **F1-Score (Macro Average)** | **69.91%** | Média harmônica equilibrada entre Precisão e Recall. |
| **ROC-AUC (Macro OVR)** | **88.90%** | Área sob a curva ROC. Reflete o alto poder de separabilidade das classes pelo modelo (capacidade de distinguir quem se formará de quem evadirá). |

### Justificativa de Escolha do Modelo (Explicação para a Gestão)

> **Por que escolhemos o `HistGradientBoostingClassifier`?**
>
> 1. **Robustez a Dados Ausentes e Mistos**: O dataset contém atributos numéricos (médias) e categóricos (estado civil, bolsas). Modelos baseados em árvores Gradient Boosting lidam nativamente com essa disparidade e suportam valores nulos (neste projeto, tratados de forma transparente e atribuídos automaticamente ao nó filho com maior ganho de informação durante o treinamento).
> 2. **Evitando Overfitting**: Comparado a árvores de decisão tradicionais ou modelos complexos que decoram o treino, regulamos a profundidade (`max_depth=4`) e a taxa de aprendizado (`learning_rate=0.04`), resultando em uma acurácia de treino saudável (~86.3%) e teste consistente (~76.3%).
> 
> **Qual métrica é mais relevante e por quê?**
> 
> A métrica de maior valor estratégico para o negócio da instituição é a **Revocação (Recall) para a classe `Dropout` (atualmente de 72.18%)**.
> *   Na evasão acadêmica, o custo de um **Falso Negativo** é crítico: se o modelo não identificar um aluno sob real risco de desistência (Dropout), a instituição perderá a oportunidade de oferecer suporte ou renegociação, resultando em evasão definitiva e perda de receita recorrente.
> *   Por outro lado, o custo de um **Falso Positivo** (o modelo aponta risco em um aluno financeiramente estável e com boas notas) é extremamente baixo: apenas o tempo gasto em um e-mail de contato preventivo do coordenador, ação que ainda demonstra atenção e cuidado com o estudante.
>
> Portanto, priorizar a **Revocação** nos garante capturar o maior número possível de evasões iminentes.

---

## 🤖 3. Arquitetura de Multi-Agentes (CrewAI)

Configuramos uma equipe colaborativa com dois agentes inteligentes conectados ao modelo local **Qwen 2.5 7B Instruct** no LM Studio:

### Agentes e Ferramentas:

1. **Organizador de Tarefas Acadêmicas**
   * **Papel**: Assistente de produtividade focado na rotina e prazos do aluno. Ajuda a priorizar entregas e organizar o calendário de estudos.
   * **Ferramentas**: `Consultar Cronograma de Tarefas` (lê e formata tarefas pendentes, em andamento ou concluídas de `tarefas.csv`) e `Consultar Informações das Disciplinas` (retorna horários, dias e professores).
2. **Mentor Acadêmico de Suporte ao Aluno**
   * **Papel**: Acolher o aluno de forma holística, calculando médias ponderadas, analisando o progresso de notas e inferindo o risco de evasão acadêmica de forma preditiva.
   * **Ferramentas**: 
     * `Consultar Notas do Aluno`: Busca notas reais do aluno em `notas.csv`, agrupando por disciplina e pré-calculando médias parciais e normalizadas em Python (evitando erros de cálculo matemáticos no LLM).
     * `Consultar Cronograma de Tarefas`: Acessa tarefas para correlacionar cronogramas e prioridades.
     * `Prever Risco de Evasão Acadêmica`: Integra o classificador de Machine Learning do projeto (`modelo_evasao.joblib`), inserindo variáveis demográficas, financeiras e acadêmicas para computar o percentual de risco de Dropout.
     * `Consultar Informações das Disciplinas`: Acessa a grade de aulas e professores.

---

## 🔍 4. Observabilidade (Traces no Langfuse e Telemetria)

Para monitorar em tempo real o fluxo de chamadas e a qualidade de raciocínio da IA, integramos o **Langfuse Cloud**. Toda execução do assistente Streamlit e das CrewAI tools envia logs de execução para o dashboard da nuvem utilizando o decorator `@observe`.

### Como acessar e tirar a captura de tela dos traces:
1. Acesse o painel web em [cloud.langfuse.com](https://cloud.langfuse.com/) com sua conta.
2. Acesse a aba **Traces** no menu lateral esquerdo.
3. Você visualizará a lista contendo as execuções de teste realizadas. Executamos previamente uma rotina de 5 consultas acadêmicas distintas, garantindo a presença dos seguintes traces prontos para a captura:
   * *Perguntar ao Assistente* (pergunta sobre tarefas)
   * *Perguntar ao Assistente* (pergunta sobre médias)
   * *Perguntar ao Assistente* (pergunta sobre evasão)
   * *Perguntar ao Assistente* (pergunta sobre professores)
   * *Perguntar ao Assistente* (pergunta sobre robustez de dados)
4. **Anexe o print screen da tela do Langfuse contendo a lista com os 5 traces abaixo:**

*(Insira aqui a sua captura de tela da lista de traces do Langfuse)*

---

## 🧪 5. Suíte de Testes (DeepEval & Golden Dataset)

Para validar a fidelidade do assistente (garantindo que ele não sofra alucinações e utilize estritamente a base de dados fornecida), implementamos uma suíte de teste robusta contendo **10 perguntas (Golden Dataset)** no arquivo [tests/test_flow.py](file:///c:/Users/felpp/OneDrive/Documentos/Projeto_puc/tests/test_flow.py).

### Perguntas Cobertas no Golden Dataset:

| ID | Pergunta do Usuário | Módulo / Ferramenta Validada |
|---|---|---|
| **01** | *"Quais são minhas tarefas pendentes de alta prioridade?"* | `consultar_cronograma_de_tarefas` (Filtro Alta Prioridade) |
| **02** | *"Qual é minha média em Inteligência Artificial?"* | `consultar_notas_do_aluno` (Averages IA) |
| **03** | *"Eu corro algum risco de evasão escolar considerando minha mensalidade em dia?"* | `prever_risco_de_evasao_academica` (ML - Baixo Risco) |
| **04** | *"Qual é a minha média em História da Computação?"* | Teste de Robustez (Matéria Inexistente - Negativação Correta) |
| **05** | *"Qual é o meu risco de evasão considerando atrasos e reprovações?"* | `prever_risco_de_evasao_academica` (ML - Alto Risco) |
| **06** | *"Quais são minhas tarefas de baixa prioridade?"* | `consultar_cronograma_de_tarefas` (Filtro Baixa Prioridade) |
| **07** | *"Quem é o professor de Banco de Dados?"* | Busca de Professor na grade de Disciplinas |
| **08** | *"Quais são minhas tarefas de média prioridade?"* | `consultar_cronograma_de_tarefas` (Filtro Média Prioridade) |
| **09** | *"Qual é a minha média atual na disciplina Banco de Dados?"* | `consultar_notas_do_aluno` (Averages BD) |
| **10** | *"Qual é o dia da semana e o horário da minha aula de Machine Learning?"* | Consulta de Calendário de Aulas |

### Resultado dos Testes:
Os testes avaliam a fidelidade das afirmações da resposta contra o contexto real usando o LLM local. A suíte atinge **100% de taxa de aprovação (Pass Rate)**:
```text
✓ Evaluation completed 🎉! (time taken: 285.85s)
» Test Results (10 total tests):
   » Pass Rate: 100.0% | Passed: 10 | Failed: 0
```
