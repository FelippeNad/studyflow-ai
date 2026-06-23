# StudyFlow AI — Descrição de Projeto e Tecnologias para Currículo 🎓

Este documento contém versões prontas para você copiar e colar no seu **Currículo (CV)**, **LinkedIn**, ou **Portfólio**, além de um guia com os principais pontos a serem destacados em **entrevistas de emprego**.

---

## 📄 1. Versões Prontas para Copiar e Colar

### Opção A: Versão Ultra-Direta (Ideal para Currículos de 1 página)
> **StudyFlow AI — Assistente Acadêmico Inteligente com IA & Machine Learning** (Portfólio Pessoal)
> *   Desenvolveu um assistente acadêmico completo utilizando uma **arquitetura de múltiplos agentes autônomos (CrewAI)** integrado a um modelo de linguagem local (**Qwen 2.5 7B via LM Studio**) e interface web em **Streamlit**.
> *   Criou um modelo preditivo de Machine Learning (**HistGradientBoosting** via **Scikit-Learn**) para estimativa em tempo real de risco de evasão escolar (Dropout) baseado no dataset da UCI, alcançando **ROC-AUC de 88,9%** e **76,2% de Acurácia**.
> *   Implementou uma suíte de testes de fidelidade de respostas (**DeepEval** e **Pytest**) contra um Golden Dataset de 10 casos complexos, mitigando alucinações de LLM e atingindo **100% de aprovação**.
> *   Estruturou a observabilidade e telemetria da aplicação utilizando **Langfuse** e **LiteLLM**, monitorando latência, consumo de tokens e a cadeia de chamadas dos agentes.
>
> **Tecnologias:** Python, CrewAI, Streamlit, Scikit-Learn, Pandas, LM Studio, Langfuse, DeepEval, Git.

---

### Opção B: Versão Detalhada por Tópicos (Ideal para LinkedIn ou Projetos no GitHub)
> **StudyFlow AI — Assistente Acadêmico Inteligente** | *Desenvolvedor e Engenheiro de IA*
>
> O **StudyFlow AI** é um ecossistema projetado para auxiliar estudantes a gerenciar sua rotina de estudos, grades e tarefas acadêmicas, contando com uma camada preventiva de análise de evasão baseada em inteligência artificial.
> 
> *   **Arquitetura Multi-Agente:** Projetou e orquestrou um sistema multi-agente usando **CrewAI** com dois agentes com papéis definidos (*Organizador de Tarefas* e *Mentor Acadêmico*) que colaboram entre si utilizando ferramentas (*tools*) de consulta a dados estruturados e modelos preditivos.
> *   **Engenharia de Machine Learning:** Treinou e otimizou um classificador **HistGradientBoosting** para predição de status acadêmico (Dropout/Enrolled/Graduate), com foco estratégico em **Recall (Sensibilidade) de 72,1%** na classe de evasão para permitir intervenções institucionais preventivas.
> *   **Integração e Engenharia de Prompt:** Integrou o modelo local **Qwen 2.5 7B Instruct** utilizando o **LM Studio** e o framework **LiteLLM**, garantindo segurança no processamento de dados confidenciais dos estudantes e custo zero de execução de API.
> *   **LLMOps e Observabilidade:** Configurou o **Langfuse** para monitoramento ponta a ponta (*traces* de execução) do fluxo de raciocínio da IA, permitindo auditar o uso de ferramentas e avaliar métricas de eficiência.
> *   **Qualidade e Testes de Fidelidade (RAG/Agents):** Criou um Golden Dataset com testes de asserção integrados ao **DeepEval** para avaliar a fidelidade (Faithfulness) das respostas geradas pelo LLM contra as bases de dados CSV (simulando tabelas de notas, tarefas e disciplinas).

---

## 🛠️ 2. Ficha Técnica Completa das Tecnologias Usadas

Se você precisar detalhar a pilha de tecnologias em uma seção específica do currículo, utilize a organização abaixo:

| Categoria | Tecnologia | Função no Projeto |
|---|---|---|
| **Linguagem** | **Python 3.10/3.11** | Linguagem base para o desenvolvimento do back-end, inteligência artificial e scripts. |
| **Framework de Agentes** | **CrewAI** | Criação, orquestração e atribuição de tarefas entre os agentes autônomos. |
| **Machine Learning** | **Scikit-Learn** | Pré-processamento de dados e treinamento do classificador preditivo (`HistGradientBoostingClassifier`). |
| **Persistência de Modelos**| **Joblib** | Serialização e salvamento do modelo preditivo para carregamento rápido em produção. |
| **Processamento de Dados** | **Pandas** | Manipulação, filtragem e mesclagem de tabelas de dados estruturados em CSV. |
| **Interface de Usuário** | **Streamlit** | Interface gráfica web responsiva para o chat e interação direta do estudante. |
| **Ambiente de LLM Local** | **LM Studio** | Execução local do modelo **Qwen 2.5 7B Instruct**, simulando endpoints compatíveis com a OpenAI. |
| **LLM Gateway** | **LiteLLM** | Ponte de tradução de chamadas entre a API local do LM Studio e a biblioteca de observabilidade. |
| **Observabilidade (LLMOps)**| **Langfuse** | Telemetria de agentes, gravação de *traces*, auditoria de chamadas de ferramentas e custos. |
| **Avaliação de LLM** | **DeepEval** | Framework de avaliação quantitativa e qualitativa das respostas dos agentes. |
| **Testes Automatizados** | **Pytest** | Automação e execução contínua da suíte de testes do Golden Dataset. |

---

## 🎯 3. Conquistas Técnicas Relevantes (Destaques para Entrevistas)

Em processos seletivos, os recrutadores valorizam dados numéricos e decisões de engenharia embasadas. Aqui estão os pontos mais fortes do seu projeto para você citar:

### 1. Decisão Estratégica em Machine Learning (Trade-off de Métricas)
*   **O que falar:** "Durante o desenvolvimento do modelo de risco de evasão baseado na base do UCI Repository, a métrica de negócio mais crítica identificada foi a **Revocação (Recall) de 72,18%** para a evasão (*Dropout*), e não apenas a acurácia global de 76,27%. Isso ocorreu porque o custo de um falso negativo (não detectar um aluno sob risco de evasão que acaba saindo) é imensamente maior do que um falso positivo (entrar em contato de suporte com um aluno que estava estável)."
*   **Por que impressiona:** Demonstra visão de negócio e entendimento de que métricas de ML servem aos objetivos da empresa.

### 2. Mitigação de Alucinações com DeepEval
*   **O que falar:** "Sistemas baseados em LLMs são propensos a alucinações, especialmente em cenários acadêmicos que envolvem notas e tarefas. Para mitigar isso, implementamos um pipeline de teste automatizado usando **DeepEval** que avalia métricas de *Faithfulness* (fidelidade às fontes). Criamos um Golden Dataset com as 10 principais consultas e atingimos **100% de taxa de aprovação**."
*   **Por que impressiona:** Mostra maturidade em engenharia de software e que você se preocupa com a confiabilidade de sistemas em produção.

### 3. Solução com LLMs Locais (Privacidade e Custo Zero)
*   **O que falar:** "Optamos por rodar o modelo **Qwen 2.5 7B Instruct** localmente via **LM Studio** em vez de consumir APIs pagas como as da OpenAI. Essa escolha garantiu **100% de privacidade dos dados confidenciais dos alunos** (que nunca deixam a infraestrutura local) e **zerou os custos recorrentes de inferência**, viabilizando testes locais ilimitados."
*   **Por que impressiona:** Aborda aspectos cruciais de segurança de dados (como LGPD) e otimização de custos de infraestrutura em nuvem.

### 4. Telemetria e Traces
*   **O que falar:** "Com múltiplos agentes e chamadas recursivas de ferramentas, a depuração de LLMs pode se tornar caótica. Integrei o **Langfuse** para mapear de forma visual todo o ciclo de vida de uma pergunta do usuário: desde a chamada do agente, a seleção da ferramenta em Pandas, a execução do modelo Scikit-Learn e a consolidação do texto final."
*   **Por que impressiona:** Demonstra familiaridade com práticas modernas de **LLMOps**.

---

## 💡 4. Perguntas Frequentes em Entrevistas de Emprego sobre este Projeto

### P1: "Por que você escolheu usar CrewAI em vez de programar os agentes manualmente?"
> **Resposta sugerida:** "O CrewAI nos fornece uma abstração robusta de orquestração de papéis, gerenciamento de memória e colaboração sequencial. Ao invés de reimplementar a lógica de loop de prompts e tratamento de exceções de ferramentas manualmente, o CrewAI permitiu focar no design das *tools* em Pandas e na definição clara de responsabilidades para o Organizador de Tarefas e para o Mentor Acadêmico."

### P2: "Por que você usou HistGradientBoostingClassifier em vez de redes neurais profundas para a classificação?"
> **Resposta sugerida:** "O dataset utilizado ('Predict Students' Dropout and Academic Success') contém uma mistura de atributos categóricos e contínuos, com alguns valores nulos implícitos. Algoritmos baseados em árvores de decisão impulsionadas (como Gradient Boosting) lidam nativamente com esse tipo de dado sem necessidade de pipelines complexos de normalização. Além disso, com a profundidade regulada em 4, conseguimos uma excelente generalização com baixo risco de overfitting e inferência ultrarápida (abaixo de 10ms), o que facilita a integração em tempo real com o assistente de IA."

### P3: "Como foi o processo de integração entre o modelo preditivo de ML e o agente do CrewAI?"
> **Resposta sugerida:** "Criamos uma ferramenta personalizada em Python (`prever_risco_evasao_tool`) decorada com a diretiva `@tool` do CrewAI. O agente Mentor Acadêmico é instruído a reconhecer perguntas sobre evasão ou desistência e invocar essa ferramenta fornecendo os parâmetros necessários (como se a mensalidade está em dia, notas parciais, etc). A função carrega o binário pré-treinado do classificador (`modelo_evasao.joblib`), faz a inferência usando o Pandas para preparar o vetor de características e devolve uma resposta estruturada em português que o LLM então consolida na conversa final."
