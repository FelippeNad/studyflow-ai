## Arquitetura da Implementação

Este projeto foi estruturado para utilizar tecnologias modernas integradas de forma coesa:

- **Flowise**: Atua como a interface principal e orquestrador visual do assistente.
- **LM Studio**: Utilizado como servidor local do modelo de linguagem (LLM), rodando os modelos pesados na máquina do host.
- **Documentos `.txt`**: Servem como base de conhecimento estática para a funcionalidade de Recuperação Aumentada por Geração (RAG).
- **Arquivos `.csv`**: Funcionam como banco de dados estruturado, guardando tarefas, notas e informações de disciplinas.
- **Scripts Python**: Realizam a validação de integridade dos arquivos e executam consultas específicas nos dados estruturados de forma rápida.

---

## Estrutura do Projeto

Abaixo está a estrutura atualizada do projeto:

```text
Projeto_puc/
│
├── data/                       # Diretório para dados estruturados
│   ├── disciplinas.csv
│   ├── notas.csv
│   └── tarefas.csv
│
├── documentos/                 # Diretório para os documentos do RAG
│   ├── faq_aluno.txt
│   ├── manual_estudos.txt
│   └── regulamento_academico.txt
│
├── scripts/                    # Scripts Python auxiliares
│   ├── consultar_dados.py
│   └── validar_csvs.py
│
├── docker-compose.yml          # Configuração do Docker para rodar o Flowise localmente
├── .env.example                # Variáveis de ambiente de exemplo
├── .gitignore                  # Arquivos ignorados pelo Git
└── README.md                   # Documentação do projeto
```

---

## Como Executar

Siga os comandos abaixo no seu terminal para iniciar e testar o projeto:

1. **Validar os arquivos CSV:**
   Garante que os dados estruturados possuem a formatação correta e IDs consistentes.
   ```bash
   python scripts/validar_csvs.py
   ```

2. **Testar as Consultas Estruturadas:**
   Gera um resumo acadêmico baseado nos CSVs atuais.
   ```bash
   python scripts/consultar_dados.py
   ```

3. **Subir a Infraestrutura (Flowise):**
   Inicia o container do Flowise no Docker em background.
   ```bash
   docker compose up -d
   ```

6. **Subir a Interface Gráfica (Streamlit):**
   ```bash
   streamlit run app.py --server.port 8502
   ```

   *(Opcional) Ver os logs do Flowise em background:*
   ```bash
   docker compose logs -f flowise
   ```

---

## Configuração no Flowise

Com o container em execução, realize os passos manuais abaixo para configurar o agente:

1. Acesse **http://localhost:3000** no seu navegador.
2. Crie um novo **Chatflow**.
3. Configure o nó do modelo (LLM) utilizando o componente **OpenAI-compatible endpoint** e insira a Base URL:
   `http://host.docker.internal:1234/v1`
   *(Observação: Certifique-se de que o servidor local no LM Studio está ativo na porta 1234).*
   *(Alternativa para redes externas/VPN: `http://26.141.70.87:1234/v1`)*
4. Configure o componente **Document Loader** (ex: Text File) apontando o caminho relativo de dentro do Docker:
   `/data/documentos/`
5. Conecte esse Loader a uma base vetorial (Vector Store) para indexar os documentos textuais.
6. Crie e conecte o fluxo final de RAG à interface de chat.
7. Realize o teste com perguntas baseadas nos documentos.

---

## Exemplos de Perguntas para Demonstração

Use as perguntas abaixo para testar o assistente.

**Perguntas via RAG (Baseadas em texto):**
- Qual é a frequência mínima para aprovação?
- O que fazer se eu perder uma prova?
- Quais técnicas de estudo posso usar para organizar minha semana?

**Perguntas via Dados Estruturados (Para testes dos scripts CSV):**
- Quais tarefas estão pendentes?
- Qual é minha média atual em Inteligência Artificial?
- Quais disciplinas tenho na quinta-feira?
- Quais tarefas possuem prioridade alta?

**Perguntas com Lógica de Agente (Avançado):**
- Quais tarefas devo priorizar hoje?
- Monte um plano de estudos para esta semana.
- Estou em risco de ficar abaixo da média em alguma disciplina?
