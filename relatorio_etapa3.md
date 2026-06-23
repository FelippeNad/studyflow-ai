# Documento de Análise Crítica - StudyFlow AI

> **Nota para o Aluno**: Este é um rascunho base para a sua entrega final da Etapa 3. Modifique as seções com as suas próprias palavras e adicione a URL final do deploy e o link do vídeo do YouTube.

**Nome do Aluno:** Felippe
**Desafio Escolhido:** Organização da vida acadêmica (Gestão de Regulamento, Notas e Previsão de Evasão)
**URL da Aplicação (Deploy):** https://studyflow-ai-exyhdiosyx3d3xlix8zefj.streamlit.app/
**URL do Vídeo (Apresentação):** https://www.youtube.com/watch?v=vt2-bDaiV9c

---

## Parte A: O Desafio

### 1. Resumo do Desafio Original
Na Etapa 1, identifiquei que a vida acadêmica moderna sofre com a fragmentação de informações: regulamentos universitários longos em PDF, acompanhamento de tarefas descentralizado e a falta de visão preditiva sobre o risco de evasão. O desafio proposto foi construir uma solução unificada (StudyFlow AI) que atuasse como um painel cognitivo para o aluno.

### 2. A Solução Resolve o Desafio?
Sim, o projeto entrega uma solução prática e efetiva para o problema proposto. Através de um chat conversacional unificado, o aluno consegue:
1. Extrair respostas precisas sobre as regras da universidade (RAG).
2. Consultar prazos e médias de notas a partir de uma base estruturada (CSV).
3. Visualizar o seu risco de evasão através de uma ferramenta de Machine Learning treinada especificamente para este contexto acadêmico.

Essa abordagem "One-Stop-Shop" diminui o estresse cognitivo, atingindo plenamente o objetivo do Challenge-Based Learning.

---

## Parte B: Análise Técnica

### 1. O Que Deu Certo
*   **Orquestração de Agentes (CrewAI)**: A arquitetura multi-agente se mostrou extremamente flexível. Separar as "personas" (Agente do Regulamento, Agente de Tarefas e Analista de Machine Learning) evitou alucinações e permitiu que a IA delegasse a tarefa para a ferramenta certa.
*   **Vector DB em Cloud**: A migração do ChromaDB local para o **Qdrant Cloud** foi um grande acerto técnico. Removemos problemas de dependências nativas (falha crônica da DLL do HNSWLib no Windows) e melhoramos a velocidade da busca vetorial.
*   **Telemetria (Langfuse)**: A integração do Langfuse nos deu visibilidade total do custo de tokens e do caminho de execução dos agentes, o que foi vital para debug.
*   **Segurança Leve**: Para contornar limitações de VRAM com o PyTorch no Windows local, implementamos um pipeline de segurança *Lightweight* baseado em expressões regulares e heurísticas, que blinda o sistema contra injeção de prompt e PII (CPF, celular) sem consumir ciclos extras de GPU.

### 2. O Que Não Deu Tão Certo (Limitações)
*   **Latência em Hardware Local**: Como toda a inferência (Qwen 7B) está sendo rodada em um notebook com GPU/CPU modestas via LM Studio, o tempo de resposta do CrewAI (que usa raciocínio *ReAct* com múltiplos *loops* de pensamento) é demorado.
*   **Golden Dataset (DeepEval)**: Tivemos problemas de codificação (`UnicodeEncodeError` no terminal Windows) durante a execução dos testes com DeepEval, embora tenham sido superados. O LM Studio, quando usado como "Juiz LLM", sofreu para rodar prompts complexos de avaliação.

---

## Parte C: Evolução (Visão de Futuro)

Se este projeto fosse escalado para produção e patrocinado por um investidor, eu proporia as seguintes três evoluções tecnológicas:

### 1. Memória de Longo Prazo (Long-term Memory) e Banco Gráfico (Knowledge Graph)
*   **Tecnologia:** Neo4j e Mem0.
*   **Por quê:** Hoje, cada sessão do chat é esquecida. Se usarmos um banco de grafos (Neo4j) atrelado a um sistema de Memória, o assistente lembraria do perfil do aluno ao longo dos semestres.
*   **Impacto:** Personalização extrema. O assistente não precisaria perguntar em qual semestre o aluno está, e já sugeriria matérias baseadas no histórico de dificuldades (Graph RAG).

### 2. Agentes Ativos e Processamento Assíncrono (Event-Driven AI)
*   **Tecnologia:** Kafka (ou RabbitMQ) + LangGraph.
*   **Por quê:** Atualmente, o aluno precisa *perguntar* ao chat. Um sistema reativo (Event-Driven) monitoraria o sistema da faculdade o tempo todo.
*   **Impacto:** O assistente enviaria proativamente um alerta pelo WhatsApp: *"Felippe, vi que você tirou uma nota baixa na Prova 1. Seu Risco de Evasão subiu para 15%. Vamos agendar um horário de monitoria?"*. Isso mudaria a IA de reativa para proativa.

### 3. Deploy de LLM Próprio em Endpoint Serverless (vLLM / Groq)
*   **Tecnologia:** vLLM (para deploy de modelo open-source) ou Groq (LPU).
*   **Por quê:** O LM Studio local não escala para múltiplos alunos simultâneos.
*   **Impacto:** Hospedaríamos nosso próprio modelo Llama 3 8B ajustado para o contexto (Fine-Tuning) em uma nuvem Serverless. O tempo de resposta cairia de 20 segundos para menos de 1 segundo, possibilitando o acesso de toda a universidade simultaneamente, com custos viáveis.
