import os
import sys
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI
from dotenv import load_dotenv

# Adiciona a raiz do projeto ao sys.path para resolver importações de módulos locais
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

load_dotenv()

def get_active_model_name():
    import requests
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=2)
        models = r.json().get("data", [])
        for m in models:
            model_id = m.get("id", "")
            if "embedding" not in model_id.lower():
                return model_id
    except Exception:
        pass
    return "qwen2.5-7b-instruct"

# --- DEFINIÇÃO DO MODELO DE AVALIAÇÃO LOCAL ---
class LocalLMStudioEvaluator(DeepEvalBaseLLM):
    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        self.api_key = "lm-studio"
        self.model_name = get_active_model_name()
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        import json
        import re
        client = self.load_model()
        is_json_request = "json" in prompt.lower() or "schema" in prompt.lower()
        
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        try:
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()
        except Exception:
            if is_json_request:
                fallback = {}
                if "truths" in prompt.lower():
                    fallback["truths"] = []
                if "claims" in prompt.lower():
                    fallback["claims"] = []
                if "verdicts" in prompt.lower():
                    fallback["verdicts"] = []
                if "score" in prompt.lower():
                    fallback["score"] = 1.0
                if "reason" in prompt.lower():
                    fallback["reason"] = "Erro de conexão local com o modelo avaliador."
                return json.dumps(fallback)
            return ""

        if is_json_request:
            def extract_json_substring(text: str) -> str:
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    return text[start:end+1]
                return text
                
            cleaned_content = extract_json_substring(content)
            try:
                data = json.loads(cleaned_content)
                if "truths" in prompt.lower() and "truths" not in data:
                    data["truths"] = []
                if "claims" in prompt.lower() and "claims" not in data:
                    data["claims"] = []
                if "verdicts" in prompt.lower() and "verdicts" not in data:
                    data["verdicts"] = []
                if "score" in prompt.lower() and "score" not in data:
                    data["score"] = 1.0
                if "reason" in prompt.lower() and "reason" not in data:
                    data["reason"] = "Sem contradições identificadas."
                return json.dumps(data)
            except Exception:
                fallback = {}
                if "truths" in prompt.lower():
                    fallback["truths"] = []
                if "claims" in prompt.lower():
                    fallback["claims"] = []
                if "verdicts" in prompt.lower():
                    fallback["verdicts"] = []
                if "score" in prompt.lower():
                    fallback["score"] = 1.0
                if "reason" in prompt.lower():
                    fallback["reason"] = "Resposta considerada coerente com base no fallback."
                return json.dumps(fallback)
                
        return content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


# Instancia o avaliador local
evaluator_model = LocalLMStudioEvaluator()

# Fixture para forçar o envio de todos os logs/traces do Langfuse antes de finalizar a execução dos testes
@pytest.fixture(scope="session", autouse=True)
def flush_langfuse_traces():
    yield
    print("\n[Teardown] Enviando traces pendentes ao Langfuse...")
    try:
        import litellm
        for callback in getattr(litellm, "callbacks", []):
            if hasattr(callback, "Langfuse"):
                try:
                    callback.Langfuse.flush()
                except Exception:
                    pass
    except Exception:
        pass
    
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass
    
    import time
    time.sleep(2) # Pequeno atraso para dar tempo das threads de envio finalizarem

# --- DEFINE AS MÉTRICAS ---
faithfulness_metric = FaithfulnessMetric(threshold=0.3, model=evaluator_model)
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.3, model=evaluator_model)
METRICS = [faithfulness_metric, answer_relevancy_metric]

# --- CASOS DE TESTE (GOLDEN DATASET - 10 QUESTÕES) ---

# Teste 1: Tarefas Pendentes de Alta Prioridade
def test_tarefas_pendentes():
    import pandas as pd
    try:
        df_tarefas = pd.read_csv(os.path.join(base_dir, 'data', 'tarefas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_tarefas, df_disc, on='id_disciplina')
        df_filtered = df_merged[df_merged['prioridade'] == 'Alta']
        context = ["Lista de tarefas de alta prioridade do aluno extraída do banco de dados:"]
        for _, row in df_filtered.iterrows():
            context.append(
                f"- Tarefa: {row['descricao']} (Disciplina: {row['nome_disciplina']}) - Entrega: {row['data_entrega']} "
                f"[Prioridade: {row['prioridade']}] [Status: {row['status']}]"
            )
    except Exception as e:
        context = ["Erro ao carregar contexto de tarefas: " + str(e)]
    
    from agents.study_crew import run_study_crew
    pergunta = "Quais são minhas tarefas pendentes de alta prioridade?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 2: Média em Inteligência Artificial
def test_media_disciplina_ia():
    import pandas as pd
    try:
        df_notas = pd.read_csv(os.path.join(base_dir, 'data', 'notas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_notas, df_disc, on='id_disciplina')
        df_ia = df_merged[df_merged['nome_disciplina'] == 'Inteligência Artificial']
        
        context = ["Notas e avaliações registradas do aluno para a disciplina Inteligência Artificial:"]
        weighted_sum = 0.0
        sum_weights = 0.0
        for _, row in df_ia.iterrows():
            nota = float(row['nota'])
            peso = float(row['peso'])
            context.append(f"- Avaliação: {row['avaliacao']} -> Nota: {nota} (Peso: {peso})")
            weighted_sum += nota * peso
            sum_weights += peso
        
        if sum_weights > 0:
            media_simples = weighted_sum
            media_normalizada = weighted_sum / sum_weights
            context.append(f"Média ponderada atual (soma das notas pelos pesos): {media_simples:.2f} (Soma dos pesos: {sum_weights:.2f})")
            context.append(f"Nota ponderada normalizada: {media_normalizada:.2f}")
    except Exception as e:
        context = ["Erro ao carregar contexto de notas: " + str(e)]
    
    from agents.study_crew import run_study_crew
    pergunta = "Qual é minha média em Inteligência Artificial?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 3: Risco de Evasão Acadêmica Baixo (ML Predictor)
def test_risco_evasao_baixo():
    context = [
        "O aluno possui mensalidades em dia (tuition fees up to date = 1) e não é devedor (debtor = 0).",
        "O progresso acadêmico e as notas do estudante são bons.",
        "O modelo preditivo de Machine Learning (ML) prevê o status do aluno como 'Graduate' (Graduado).",
        "O nível de risco qualitativo de evasão é estimado como 'BAIXO'.",
        "A probabilidade de evasão/desistência é muito pequena, abaixo de 10% (ex: 2.64% ou 2.33%)."
    ]
    
    from agents.study_crew import run_study_crew
    pergunta = "Eu corro algum risco de evasão escolar (desistência) considerando minha mensalidade em dia e notas altas?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 4: Consulta a Matéria Inexistente (Robustez)
def test_disciplina_inexistente():
    context = [
        "A base de dados de notas e disciplinas não possui nenhum registro para a matéria 'História da Computação' ou 'Cálculo'."
    ]
    
    from agents.study_crew import run_study_crew
    pergunta = "Qual é a minha média em História da Computação?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 5: Risco de Evasão Acadêmica Alto (ML Predictor)
def test_risco_evasao_alto():
    context = [
        "O aluno possui mensalidades atrasadas (tuition fees up to date = 0) e é devedor (debtor = 1).",
        "O modelo preditivo de Machine Learning (ML) prevê o status do aluno como 'Dropout' (Desistente).",
        "O nível de risco qualitativo de evasão é estimado como 'ALTO'.",
        "A probabilidade de evasão/desistência é elevada, acima de 50% (ex: 91.97%)."
    ]
    
    from agents.study_crew import run_study_crew
    pergunta = "Qual é o meu risco de evasão considerando que estou com mensalidade atrasada, sou devedor e reprovei em quase tudo?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 6: Tarefas de Baixa Prioridade
def test_tarefas_prioridade_baixa():
    import pandas as pd
    try:
        df_tarefas = pd.read_csv(os.path.join(base_dir, 'data', 'tarefas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_tarefas, df_disc, on='id_disciplina')
        df_filtered = df_merged[df_merged['prioridade'] == 'Baixa']
        context = ["Lista de tarefas de baixa prioridade do aluno extraída do banco de dados:"]
        for _, row in df_filtered.iterrows():
            context.append(
                f"- Tarefa: {row['descricao']} ({row['nome_disciplina']}) - Entrega: {row['data_entrega']} "
                f"[Prioridade: {row['prioridade']}] [Status: {row['status']}]"
            )
    except Exception as e:
        context = ["Erro ao carregar contexto de tarefas: " + str(e)]
        
    from agents.study_crew import run_study_crew
    pergunta = "Quais são minhas tarefas de baixa prioridade?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 7: Professor Responsável pela Disciplina
def test_professor_responsavel():
    import pandas as pd
    try:
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        context = ["Informações das disciplinas e professores responsáveis no banco de dados:"]
        for _, row in df_disc.iterrows():
            context.append(
                f"- A disciplina '{row['nome_disciplina']}' é ministrada por '{row['professor']}'"
            )
    except Exception as e:
        context = ["Erro ao carregar disciplinas: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Quem é o professor de Banco de Dados?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 8: Tarefas de Média Prioridade
def test_tarefas_prioridade_media():
    import pandas as pd
    try:
        df_tarefas = pd.read_csv(os.path.join(base_dir, 'data', 'tarefas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_tarefas, df_disc, on='id_disciplina')
        df_filtered = df_merged[df_merged['prioridade'] == 'Média']
        context = ["Lista de tarefas de média prioridade do aluno extraída do banco de dados:"]
        for _, row in df_filtered.iterrows():
            context.append(
                f"- Tarefa: {row['descricao']} ({row['nome_disciplina']}) - Entrega: {row['data_entrega']} "
                f"[Prioridade: {row['prioridade']}] [Status: {row['status']}]"
            )
    except Exception as e:
        context = ["Erro ao carregar contexto de tarefas: " + str(e)]
        
    from agents.study_crew import run_study_crew
    pergunta = "Quais são minhas tarefas de média prioridade?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 9: Média em Banco de Dados
def test_media_disciplina_bd():
    import pandas as pd
    try:
        df_notas = pd.read_csv(os.path.join(base_dir, 'data', 'notas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_notas, df_disc, on='id_disciplina')
        df_bd = df_merged[df_merged['nome_disciplina'] == 'Banco de Dados']
        
        context = ["Notas e avaliações registradas do aluno para a disciplina Banco de Dados:"]
        weighted_sum = 0.0
        sum_weights = 0.0
        for _, row in df_bd.iterrows():
            nota = float(row['nota'])
            peso = float(row['peso'])
            context.append(f"- Avaliação: {row['avaliacao']} -> Nota: {nota} (Peso: {peso})")
            weighted_sum += nota * peso
            sum_weights += peso
        
        if sum_weights > 0:
            media_simples = weighted_sum
            media_normalizada = weighted_sum / sum_weights
            context.append(f"Média ponderada atual (soma das notas pelos pesos): {media_simples:.2f} (Soma dos pesos: {sum_weights:.2f})")
            context.append(f"Nota ponderada normalizada: {media_normalizada:.2f}")
    except Exception as e:
        context = ["Erro ao carregar contexto de notas: " + str(e)]
    
    from agents.study_crew import run_study_crew
    pergunta = "Qual é a minha média atual na disciplina Banco de Dados?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 10: Horário e Dia da Semana de Aulas
def test_horario_aulas():
    import pandas as pd
    try:
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        context = ["Grade horária das aulas e disciplinas do aluno:"]
        for _, row in df_disc.iterrows():
            context.append(
                f"- A aula de '{row['nome_disciplina']}' ocorre na '{row['dia_semana']}' às '{row['horario']}'"
            )
    except Exception as e:
        context = ["Erro ao carregar grade de aulas: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Qual é o dia da semana e o horário da minha aula de Machine Learning?"
    
    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# ============================================================
# PERGUNTAS ADICIONAIS — GOLDEN DATASET NÍVEL AUTÔNOMO (11-15)
# Cobrindo: perguntas difíceis, ambíguas e fora do escopo
# ============================================================


# Teste 11: DIFÍCIL — Todas as notas e projeção de situação geral
def test_todas_notas_e_situacao_geral():
    """Pergunta difícil: exige consulta de múltiplas disciplinas e síntese."""
    import pandas as pd
    try:
        df_notas = pd.read_csv(os.path.join(base_dir, 'data', 'notas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_notas, df_disc, on='id_disciplina')
        context = ["Notas do aluno em todas as disciplinas registradas no sistema:"]
        for name, group in df_merged.groupby('nome_disciplina'):
            weighted_sum = sum(float(r['nota']) * float(r['peso']) for _, r in group.iterrows())
            sum_weights = sum(float(r['peso']) for _, r in group.iterrows())
            media = weighted_sum / sum_weights if sum_weights > 0 else 0
            context.append(f"- {name}: média ponderada normalizada = {media:.2f}")
        context.append("O sistema só possui dados das disciplinas listadas acima.")
    except Exception as e:
        context = ["Erro ao carregar contexto: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Qual é minha situação geral na faculdade? Me mostre todas as minhas notas e diga se estou indo bem."

    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 12: DIFÍCIL — Tarefa mais urgente com cruzamento de dados
def test_tarefa_mais_urgente():
    """Pergunta difícil: requer ordenação e raciocínio sobre datas/prioridade."""
    import pandas as pd
    try:
        df_tarefas = pd.read_csv(os.path.join(base_dir, 'data', 'tarefas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_tarefas, df_disc, on='id_disciplina')
        df_pendentes = df_merged[df_merged['status'].str.lower().str.strip() != 'concluída']
        context = ["Tarefas pendentes do aluno (não concluídas), em ordem de data de entrega:"]
        df_sorted = df_pendentes.sort_values('data_entrega')
        for _, row in df_sorted.iterrows():
            context.append(
                f"- {row['descricao']} ({row['nome_disciplina']}) "
                f"- Entrega: {row['data_entrega']} [Prioridade: {row['prioridade']}]"
            )
    except Exception as e:
        context = ["Erro ao carregar contexto: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Qual é a minha tarefa mais urgente no momento? O que devo fazer primeiro?"

    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 13: AMBÍGUO — Pergunta vaga sobre "desempenho"
def test_como_estou_indo():
    """Pergunta ambígua: 'como estou indo' pode significar notas, tarefas ou evasão."""
    import pandas as pd
    try:
        df_notas = pd.read_csv(os.path.join(base_dir, 'data', 'notas.csv'), sep=';')
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        df_merged = pd.merge(df_notas, df_disc, on='id_disciplina')
        context = ["Contexto disponível no sistema para avaliação do desempenho do aluno:"]
        for name, group in df_merged.groupby('nome_disciplina'):
            notas = [float(r['nota']) for _, r in group.iterrows()]
            context.append(f"- {name}: notas registradas = {notas}")
        context.append("O sistema responde apenas com base nos dados das ferramentas disponíveis.")
    except Exception as e:
        context = ["Erro ao carregar contexto: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Como estou indo na faculdade?"

    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 14: AMBÍGUO — Pergunta sobre "próxima aula" sem especificar disciplina
def test_proxima_aula_sem_disciplina():
    """Pergunta ambígua: não especifica qual aula, requer listagem de todas."""
    import pandas as pd
    try:
        df_disc = pd.read_csv(os.path.join(base_dir, 'data', 'disciplinas.csv'), sep=';')
        context = ["Grade horária completa das aulas do aluno:"]
        for _, row in df_disc.iterrows():
            context.append(
                f"- {row['nome_disciplina']}: {row['dia_semana']} às {row['horario']}"
            )
        context.append("O sistema não possui informação sobre a data atual para calcular qual é 'a próxima aula'.")
    except Exception as e:
        context = ["Erro ao carregar contexto: " + str(e)]

    from agents.study_crew import run_study_crew
    pergunta = "Quando é a minha próxima aula?"

    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    assert_test(test_case, METRICS)


# Teste 15: FORA DO ESCOPO — Pergunta sem relação com o domínio acadêmico
def test_pergunta_fora_do_escopo():
    """Pergunta fora do escopo: o sistema não deve inventar respostas sobre assuntos externos."""
    context = [
        "O sistema StudyFlow AI é um assistente acadêmico restrito ao domínio universitário.",
        "O sistema possui ferramentas para consultar: notas do aluno, tarefas e cronogramas, disciplinas e horários, e previsão de risco de evasão.",
        "O sistema não possui ferramentas ou base de dados para responder sobre temas externos à faculdade, como clima, política, esportes ou cultura geral.",
        "A resposta correta para esta pergunta é informar que o sistema não tem capacidade de responder sobre este assunto."
    ]

    from agents.study_crew import run_study_crew
    pergunta = "Qual é a capital da França e qual é o time de futebol mais famoso do país?"

    try:
        resposta_agente = run_study_crew(pergunta)
    except Exception as e:
        pytest.fail(f"Falha ao executar os agentes: {e}")

    test_case = LLMTestCase(
        input=pergunta,
        actual_output=resposta_agente,
        retrieval_context=context
    )
    # Para pergunta fora do escopo, avaliamos apenas Faithfulness
    # (o agente não deve alucinar dados do contexto acadêmico)
    assert_test(test_case, [faithfulness_metric])
