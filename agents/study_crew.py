from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import litellm
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

# Configuração explícita de callbacks do Langfuse no LiteLLM/CrewAI
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Caminhos absolutos para os dados
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTAS_PATH = os.path.join(BASE_DIR, 'data', 'notas.csv')
TAREFAS_PATH = os.path.join(BASE_DIR, 'data', 'tarefas.csv')
DISCIPLINAS_PATH = os.path.join(BASE_DIR, 'data', 'disciplinas.csv')
DOCUMENTOS_DIR = os.path.join(BASE_DIR, 'documentos')

# Configuração do LLM apontando para o LM Studio local
def get_active_model_name():
    import requests
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=2)
        models = r.json().get("data", [])
        # Filtra para obter o primeiro modelo de chat (não-embedding)
        for m in models:
            model_id = m.get("id", "")
            if "embedding" not in model_id.lower():
                return model_id
    except Exception:
        pass
    return "qwen2.5-7b-instruct" # Fallback

active_model = get_active_model_name()

lm_studio_llm = LLM(
    model=f"openai/{active_model}",
    base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key="lm-studio"
)

# --- DEFINIÇÃO DAS FERRAMENTAS ---

@tool("consultar_notas")
def consultar_notas_tool(disciplina: str = None) -> str:
    """
    Consulta as notas, avaliações e médias do aluno nas disciplinas.
    Pode filtrar por uma disciplina específica (ex: 'Inteligência Artificial') se fornecido.
    """
    try:
        if not os.path.exists(NOTAS_PATH) or not os.path.exists(DISCIPLINAS_PATH):
            return "Erro: Base de dados de notas ou disciplinas não encontrada."
        
        df_notas = pd.read_csv(NOTAS_PATH, sep=';')
        df_disc = pd.read_csv(DISCIPLINAS_PATH, sep=';')
        
        df_merged = pd.merge(df_notas, df_disc, on='id_disciplina')
        
        if disciplina:
            df_filtered = df_merged[df_merged['nome_disciplina'].str.strip().str.lower() == disciplina.strip().lower()]
            if df_filtered.empty:
                exists_in_disc = df_disc[df_disc['nome_disciplina'].str.strip().str.lower() == disciplina.strip().lower()]
                if not exists_in_disc.empty:
                    return f"A disciplina '{disciplina}' existe na grade curricular, mas o aluno não possui nenhuma nota ou avaliação registrada para ela."
                else:
                    return f"Erro: A disciplina '{disciplina}' não foi encontrada nos registros do aluno nem na grade curricular."
            df_merged = df_filtered
        
        if df_merged.empty:
            return "Nenhuma nota registrada."
        
        # Agrupa por disciplina para calcular médias e formatar
        grouped = df_merged.groupby('nome_disciplina')
        resultado = "=== NOTAS, AVALIAÇÕES E MÉDIAS DO ALUNO ===\n"
        
        for name, group in grouped:
            resultado += f"\nDisciplina: {name}\n"
            weighted_sum = 0.0
            sum_weights = 0.0
            for _, row in group.iterrows():
                nota = float(row['nota'])
                peso = float(row['peso'])
                resultado += f"- {row['avaliacao']} -> Nota {nota:.2f} (Peso: {peso:.2f})\n"
                weighted_sum += nota * peso
                sum_weights += peso
            
            # Média ponderada simples e normalizada
            if sum_weights > 0:
                media_simples = weighted_sum
                media_normalizada = weighted_sum / sum_weights
                resultado += f"  * Média Ponderada Atual (Soma Ponderada): {media_simples:.2f} (Soma dos pesos: {sum_weights:.2f})\n"
                if sum_weights < 1.0:
                    resultado += f"  * Nota Ponderada Normalizada (Proporcional): {media_normalizada:.2f}\n"
            else:
                resultado += "  * Sem avaliações com peso registrado.\n"
                
        return resultado
    except Exception as e:
        return f"Erro ao consultar notas: {e}"

@tool("consultar_tarefas")
def consultar_tarefas_tool(prioridade: str = None) -> str:
    """
    Consulta a lista de tarefas, prazos, prioridades e status do aluno.
    Pode filtrar por prioridade (ex: 'Alta', 'Média', 'Baixa') se especificado.
    """
    try:
        if not os.path.exists(TAREFAS_PATH) or not os.path.exists(DISCIPLINAS_PATH):
            return "Erro: Base de dados de tarefas ou disciplinas não encontrada."
        
        df_tarefas = pd.read_csv(TAREFAS_PATH, sep=';')
        df_disc = pd.read_csv(DISCIPLINAS_PATH, sep=';')
        
        df_merged = pd.merge(df_tarefas, df_disc, on='id_disciplina')
        
        if prioridade:
            df_merged = df_merged[df_merged['prioridade'].str.lower() == prioridade.lower()]
        
        if df_merged.empty:
            return f"Nenhuma tarefa cadastrada com prioridade '{prioridade}'." if prioridade else "Nenhuma tarefa cadastrada."
        
        resultado = f"=== CRONOGRAMA DE TAREFAS (Prioridade: {prioridade if prioridade else 'Todas'}) ===\n"
        for _, row in df_merged.iterrows():
            resultado += f"- [{row['status']}] {row['descricao']} ({row['nome_disciplina']}) - Entrega: {row['data_entrega']} [Prioridade: {row['prioridade']}]\n"
        return resultado
    except Exception as e:
        return f"Erro ao consultar tarefas: {e}"

@tool("prever_evasao")
def prever_risco_evasao_tool(
    tuition_fees_up_to_date: int = 1,
    debtor: int = 0,
    scholarship_holder: int = 0,
    age_at_enrollment: int = 20,
    u1_approved: int = 5,
    u1_grade: float = 12.0,
    u2_approved: int = 5,
    u2_grade: float = 12.0
) -> str:
    """
    Executa o modelo preditivo de Machine Learning para estimar a probabilidade do estudante
    evadir da faculdade (Dropout) ou se formar (Graduate).
    Parametros:
      - tuition_fees_up_to_date: 1 se mensalidade está em dia, 0 se atrasada
      - debtor: 1 se o aluno é devedor/inadimplente, 0 caso contrário
      - scholarship_holder: 1 se o aluno é bolsista, 0 caso contrário
      - age_at_enrollment: idade do estudante ao se matricular
      - u1_approved: número de matérias aprovadas no 1º semestre (ex: 0 a 10)
      - u1_grade: média de notas no 1º semestre (escala 0 a 20)
      - u2_approved: número de matérias aprovadas no 2º semestre (ex: 0 a 10)
      - u2_grade: média de notas no 2º semestre (escala 0 a 20)
    """
    try:
        from ml.preditor import prever_risco_evasao
        dados = {
            'Tuition fees up to date': tuition_fees_up_to_date,
            'Debtor': debtor,
            'Scholarship holder': scholarship_holder,
            'Age at enrollment': age_at_enrollment,
            'Curricular units 1st sem (approved)': u1_approved,
            'Curricular units 1st sem (grade)': u1_grade,
            'Curricular units 2nd sem (approved)': u2_approved,
            'Curricular units 2nd sem (grade)': u2_grade
        }
        
        resultado = prever_risco_evasao(dados)
        
        # Traduz os termos em inglês do modelo de ML para o português
        traducao = {
            'Graduate': 'Graduado',
            'Dropout': 'Evasão',
            'Enrolled': 'Matriculado'
        }
        predicao_pt = traducao.get(resultado['prediction'], resultado['prediction'])
        probabilidades_pt = {traducao.get(k, k): f"{v:.2%}" for k, v in resultado['probabilities'].items()}
        
        return (
            f"=== ANÁLISE PREVENTIVA DE EVASÃO (ML) ===\n"
            f"- Situação Geral Estimada: {predicao_pt}\n"
            f"- Nível de Risco de Evasão (Dropout): {resultado['risk_level']}\n"
            f"- Probabilidade de Desistência: {resultado['dropout_probability']:.2%}\n"
            f"- Detalhamento de Probabilidades: {probabilidades_pt}\n"
        )
    except Exception as e:
        return f"Erro ao rodar predição de ML: {e}"

@tool("consultar_disciplinas")
def consultar_disciplinas_tool() -> str:
    """Consulta informações sobre as disciplinas ofertadas, incluindo o nome do professor responsável, carga horária, dia da semana e horário das aulas."""
    try:
        if not os.path.exists(DISCIPLINAS_PATH):
            return "Erro: Base de dados de disciplinas não encontrada."
        df_disc = pd.read_csv(DISCIPLINAS_PATH, sep=';')
        if df_disc.empty:
            return "Nenhuma disciplina encontrada."
        resultado = "=== DISCIPLINAS, PROFESSORES E HORÁRIOS ===\n"
        for _, row in df_disc.iterrows():
            resultado += (
                f"- Disciplina: {row['nome_disciplina']} | Professor: {row['professor']} | "
                f"Dia da Semana: {row['dia_semana']} | Horário: {row['horario']} | Carga Horária: {row['carga_horaria']}h\n"
            )
        return resultado
    except Exception as e:
        return f"Erro ao consultar disciplinas: {e}"

@tool("consultar_regulamento_rag")
def consultar_regulamento_tool(pergunta: str = None) -> str:
    """
    Busca semanticamente nos documentos institucionais da faculdade usando RAG
    (Retrieval-Augmented Generation) com ChromaDB e embeddings locais.
    Os documentos indexados são: Regulamento Acadêmico, FAQ do Aluno e Manual de Estudos.
    Use esta ferramenta para responder perguntas sobre:
    - Regras de notas, média mínima para aprovação (6.0), cálculo de média ponderada
    - Prova de recuperação (Exame Final): quem tem direito, como é calculada
    - Segunda Chamada: prazo (48h) e documentos necessários
    - Faltas e frequência mínima obrigatória (75%)
    - Desligamento do curso (reprovar 3x na mesma disciplina)
    - Revisão de nota: prazo de 5 dias úteis
    - Trancamento ou cancelamento de disciplina
    - Dicas de estudo (Técnica Pomodoro, Active Recall, Repetição Espaçada)
    - Qualquer outra regra ou política acadêmica institucional
    Parâmetro 'pergunta': a dúvida do aluno para busca semântica nos documentos.
    """
    try:
        from rag.document_rag import buscar_documentos_rag
        query = pergunta if pergunta else "regulamento acadêmico notas avaliação"
        return buscar_documentos_rag(query, n_results=4)
    except Exception as e:
        return f"Erro ao consultar documentos (RAG): {e}"


# --- DEFINIÇÃO DOS AGENTES ---

# 1. Organizador de Tarefas
task_agent = Agent(
    role="Organizador de Tarefas Acadêmicas",
    goal="Ajudar o estudante a gerenciar seus prazos, priorizar entregas e estruturar sua rotina de estudos.",
    backstory="Você é um assistente de produtividade extremamente focado e disciplinado. Seu objetivo é ajudar o aluno a nunca perder uma entrega e manter o calendário de estudos em ordem.",
    tools=[consultar_tarefas_tool, consultar_disciplinas_tool, consultar_regulamento_tool],
    llm=lm_studio_llm,
    verbose=True
)

# 2. Mentor Acadêmico
mentor_agent = Agent(
    role="Mentor Acadêmico de Suporte ao Aluno",
    goal="Analisar as notas do estudante, calcular seu desempenho médio, prever riscos acadêmicos ou de evasão usando a ferramenta de ML, verificar tarefas acadêmicas e horários de aula, e responder dúvidas sobre regras e regulamentos institucionais.",
    backstory="Você é um conselheiro estudantil amigável, acolhedor e focado no sucesso do estudante. Você usa dados de notas, ferramentas de predição de ML, informações de disciplinas/professores, tarefas e documentos institucionais para responder às dúvidas do estudante de forma integrada.",
    tools=[consultar_notas_tool, consultar_tarefas_tool, prever_risco_evasao_tool, consultar_disciplinas_tool, consultar_regulamento_tool],
    llm=lm_studio_llm,
    verbose=True
)

def run_study_crew(pergunta_usuario: str) -> str:
    """
    Orquestra a execução da equipe de agentes do CrewAI para responder à pergunta do usuário.
    """
    # Define uma tarefa geral que permite ao Mentor colaborar com o Organizador de Tarefas
    resposta_task = Task(
        description=(
            f"Responda à seguinte pergunta do estudante: '{pergunta_usuario}'\n\n"
            "Diretrizes:\n"
            "- Se a pergunta envolver prazos, entregas ou cronograma, consulte as tarefas.\n"
            "- Se a pergunta envolver notas, médias, desempenho ou risco de evasão/desistência, consulte as notas e use a ferramenta de previsão de evasão.\n"
            "- Se a pergunta envolver professores, horários, dias da semana, aulas ou carga horária, consulte as disciplinas.\n"
            "- Se a pergunta envolver regras acadêmicas, média mínima, reprovação, recuperação, segunda chamada, faltas, trancamento, revisão de nota, dicas de estudo ou qualquer política institucional, consulte os documentos acadêmicos.\n"
            "- FORA DO ESCOPO: Se a pergunta não puder ser respondida com nenhuma das ferramentas disponíveis (assuntos completamente não acadêmicos), responda exatamente: 'Não possuo informações sobre isso na minha base de dados. Posso te ajudar com: notas e médias, tarefas e prazos, horários e professores, regulamentos e regras acadêmicas, ou risco de evasão acadêmica.'\n"
            "- NUNCA retorne uma resposta vazia ou apenas com marcadores sem conteúdo. Se não houver dados disponíveis, informe claramente.\n"
            "- IMPORTANTE: Limite-se estritamente às informações extraídas das ferramentas. Não invente, não deduza e não extrapole informações.\n"
            "- IMPORTANTE: Responda de forma extremamente direta, concisa e objetiva. Liste apenas os dados de forma limpa. Não adicione saudações, introduções ou comentários de encerramento/observações finais (por exemplo, NÃO escreva 'estou à disposição', 'parabéns pelo bom desempenho', 'não precisa fazer imediatamente' ou similares). Escreva apenas a lista factual de dados.\n"
            "- FORMATO DE LISTAS: Se a resposta contiver múltiplos itens ou dados, você deve formatá-los obrigatoriamente como uma lista com marcadores do markdown (ex: usando `-` no início de cada linha), com cada item em sua própria linha para garantir uma renderização visual correta."
        ),
        expected_output="Uma resposta estritamente direta, concisa, objetiva e factual formatada com listas de tópicos do markdown (usando `-`) para quebra de linhas, baseada exclusivamente nos dados consultados pelas ferramentas.",
        agent=mentor_agent
    )

    # Configura a Crew com os dois agentes
    crew = Crew(
        agents=[mentor_agent, task_agent],
        tasks=[resposta_task],
        process=Process.sequential,
        verbose=True
    )

    # Executa a Crew
    resultado = crew.kickoff()
    
    # Retorna o resultado final formatado como string
    return str(resultado)

if __name__ == "__main__":
    print("Testando execução da Crew localmente...")
    # Execução de teste rápida
    resposta = run_study_crew("Qual é a minha média em Inteligência Artificial e qual meu risco de evasão?")
    print("\n--- RESPOSTA FINAL DO AGENTE ---")
    print(resposta)
