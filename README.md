# StudyFlow AI — Assistente Inteligente de Organização Acadêmica 🎓

O **StudyFlow AI** é um assistente acadêmico inteligente projetado para auxiliar estudantes a gerenciar tarefas, prazos, notas e disciplinas, além de estimar de forma preventiva o risco de evasão escolar. A solução é baseada em uma arquitetura multi-agente robusta desenvolvida com **CrewAI**, integrada com modelos locais via **LM Studio** e uma interface amigável desenvolvida em **Streamlit**.

---

## 📋 Declaração do Desafio CBL (Etapa 1)

Este projeto foi desenvolvido como parte da Etapa 1 do CBL, cujo objetivo é definir um desafio viável para a construção de um sistema com interface, LLM, dados estruturados, RAG, problema de Machine Learning e agentes/tarefas automatizadas.

O domínio escolhido foi a organização acadêmica de estudantes, com foco em auxiliar o acompanhamento de disciplinas, tarefas, prazos, notas e dúvidas frequentes sobre a vida universitária.

### 1. Grande Ideia
A grande ideia deste projeto é utilizar Inteligência Artificial para apoiar estudantes na organização da rotina acadêmica, facilitando o acompanhamento de tarefas, prazos, disciplinas e informações institucionais.

Em um contexto em que estudantes lidam com várias disciplinas, avaliações, trabalhos e responsabilidades ao mesmo tempo, um assistente inteligente pode ajudar a centralizar informações, reduzir esquecimentos e melhorar a tomada de decisão sobre o que priorizar nos estudos.

### 2. Pergunta Essencial
Como a Inteligência Artificial pode ajudar estudantes a organizarem melhor seus estudos, acompanharem prazos acadêmicos e acessarem informações importantes de forma rápida e personalizada?

### 3. Desafio
Criar um assistente acadêmico inteligente capaz de auxiliar estudantes na organização da rotina universitária.

O sistema deverá permitir que o estudante consulte tarefas pendentes, acompanhe disciplinas, verifique notas, receba sugestões de prioridade e tire dúvidas com base em documentos acadêmicos, como regulamentos, FAQs e materiais de orientação de estudo.

Além disso, o projeto deverá integrar dados estruturados, documentos para consulta via RAG, um problema de Machine Learning e agentes capazes de executar tarefas específicas, como listar prazos próximos, gerar um plano semanal de estudos e consultar a situação acadêmica do estudante.

### 4. Justificativa Pessoal
Com o passar do tempo sendo um estudante percebi que a rotina acadêmica exige organização constante, principalmente quando há várias disciplinas, trabalhos, provas e prazos acontecendo ao mesmo tempo. Muitas vezes, a dificuldade não está apenas em estudar o conteúdo, mas em saber o que deve ser priorizado, quais atividades estão pendentes e onde encontrar rapidamente informações importantes.

Por esse motivo, acredito que um assistente acadêmico inteligente pode ser útil para tornar o processo de estudo mais organizado, prático e eficiente, apoiando o estudante na gestão do tempo e no acompanhamento da própria vida acadêmica em um único lugar com pesquisas de forma rápida e eficiente.

---

## 🏗️ Arquitetura do Sistema

O ecossistema do projeto foi reformulado para utilizar tecnologias modernas de agentes autônomos sem a necessidade de orquestradores visuais:

- **Streamlit**: Interface gráfica do chat de usuário, amigável e reativa, responsável por capturar as interações e exibir as respostas dos agentes.
- **CrewAI**: Orquestrador da equipe multi-agente. Coordena a colaboração entre os agentes especializados e gerencia o uso de ferramentas (*tools*).
- **LM Studio**: Servidor local do modelo de linguagem (LLM) que roda localmente o modelo **Qwen 2.5 7B Instruct** (ou similar) garantindo privacidade e custo zero de execução.
- **Machine Learning (Predictor)**: Um classificador preditivo treinado em Python (`HistGradientBoostingClassifier`) integrado como ferramenta do CrewAI para analisar o risco de evasão acadêmica do aluno com base em dados de notas e finanças.
- **Bases de Dados CSVs**: Funcionam como o banco de dados estruturado do assistente (notas, tarefas e disciplinas).
- **Observabilidade (Langfuse)**: Monitoramento em tempo real das execuções dos agentes, permitindo a análise de latência, tokens consumidos e fidelidade das respostas do modelo.

---

## 📂 Estrutura do Projeto

Abaixo está a estrutura atual de diretórios e arquivos do projeto:

```text
Projeto_puc/
│
├── .deepeval/                  # Logs e cache da suíte de testes DeepEval
├── agents/                     # Definição dos agentes e tarefas CrewAI
│   └── study_crew.py           # Configuração da Crew, agentes, ferramentas e LLM
│
├── data/                       # Banco de dados em arquivos CSV
│   ├── data_students.csv       # Dataset histórico para treino do modelo de ML
│   ├── disciplinas.csv         # Grade curricular com disciplinas e professores
│   ├── notas.csv               # Notas e avaliações parciais do aluno
│   └── tarefas.csv             # Lista de tarefas acadêmicas com prazos e status
│
├── ml/                         # Módulo de Inteligência Artificial / Machine Learning
│   ├── modelo_evasao.joblib    # Binário do modelo preditivo de evasão treinado
│   ├── metricas_modelo.md      # Relatório detalhado das métricas obtidas
│   ├── preditor.py             # Script de inferência para o classificador
│   └── treinar_modelo.py       # Script de treinamento do classificador
│
├── scripts/                    # Scripts Python auxiliares de validação e testes
│   ├── consultar_dados.py      # Script para testes locais rápidos de consultas
│   └── validar_csvs.py         # Validador de integridade e chaves estrangeiras dos CSVs
│
├── tests/                      # Suíte de testes de fidelidade
│   └── test_flow.py            # Golden Dataset (10 testes) integrado com DeepEval
│
├── .env                        # Variáveis de ambiente configuradas
├── app.py                      # Aplicação web Streamlit (Frontend + Integração CrewAI)
├── relatorio_projeto.md        # Relatório técnico geral do projeto
└── README.md                   # Esta documentação do projeto
```

---

## 🛠️ Como Executar o Projeto

Siga o passo a passo abaixo para configurar e rodar o projeto localmente:

### 1. Requisitos Prévios
* **Python 3.10 ou 3.11** instalado.
* **LM Studio** aberto, com o modelo de chat (ex: `Qwen 2.5 7B Instruct`) carregado e o **Local Server** ativado na porta `1234`.
* Variáveis de ambiente configuradas no arquivo `.env`.

### 2. Validação dos Dados
Antes de rodar a aplicação, verifique a consistência dos dados nas tabelas CSV:
```bash
python scripts/validar_csvs.py
```

### 3. Rodar Testes de Consultas
Valide se as consultas estruturadas em Python estão retornando os dados corretos dos arquivos locais:
```bash
python scripts/consultar_dados.py
```

### 4. Executar o Streamlit
Inicie a interface gráfica da aplicação na porta 8502:
```bash
streamlit run app.py --server.port 8502
```
Acesse a aplicação no seu navegador no endereço: **http://localhost:8502**

---

## 🤖 Configuração dos Agentes (CrewAI)

A equipe é composta por dois agentes inteligentes definidos em [study_crew.py](file:///c:/Users/felpp/OneDrive/Documentos/Projeto_puc/agents/study_crew.py):

1. **Organizador de Tarefas Acadêmicas**:
   - **Objetivo**: Auxiliar o estudante a gerenciar prazos, prioridades e estruturar a rotina de estudos.
   - **Ferramentas**: `Consultar Cronograma de Tarefas` e `Consultar Informações das Disciplinas`.

2. **Mentor Acadêmico de Suporte ao Aluno**:
   - **Objetivo**: Analisar notas, médias, fornecer conselhos de apoio e acionar o preditor de evasão acadêmica.
   - **Ferramentas**: `Consultar Notas do Aluno`, `Consultar Cronograma de Tarefas`, `Consultar Informações das Disciplinas` e `Prever Risco de Evasão Acadêmica` (conexão com o modelo ML).

---

## 🧪 Suíte de Testes (DeepEval)

Para rodar a validação do Golden Dataset contendo as 10 perguntas cruciais de teste de fidelidade (garantindo 100% de precisão e zero alucinações), use o pytest integrado com o DeepEval:
```bash
deepeval test run tests/test_flow.py
```
*Certifique-se de que o LM Studio está ativo, pois o avaliador utiliza a API local compatível com OpenAI para rodar as métricas de Faithfulness.*
