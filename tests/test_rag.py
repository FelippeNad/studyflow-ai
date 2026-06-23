import os
from dotenv import load_dotenv
load_dotenv()

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM

from langchain_openai import ChatOpenAI

# 1. Configurando o DeepEval para usar o LLM local (LM Studio)
class LMStudioLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model_name="qwen2.5-7b-instruct",
            openai_api_base=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            openai_api_key="lm-studio"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "LM Studio Qwen 2.5 7B"


# 2. Casos de Teste (LLM as a Judge)
def test_rag_revisao_notas():
    # O que o aluno perguntou
    input_text = "quantos dias tenho para pedir revisão de nota?"
    
    # A resposta que o nosso agente gerou (você testou agora há pouco)
    actual_output = "5 dias úteis após a divulgação oficial do resultado no sistema"
    
    # O contexto real recuperado do Qdrant Cloud (trecho do Regulamento)
    retrieval_context = [
        "O pedido de revisão de nota deve ser protocolado na secretaria em até 5 dias úteis "
        "após a divulgação oficial do resultado no sistema acadêmico."
    ]

    # Criando o caso de teste
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    # 3. Métricas de Avaliação
    # Relevância da Resposta: a resposta atende à pergunta?
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=LMStudioLLM())
    
    # Fidelidade: a resposta é fiel ao documento ou o modelo inventou dados (hallucination)?
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=LMStudioLLM())

    # 4. Executando os Testes usando LLM as a Judge
    assert_test(test_case, [relevancy_metric, faithfulness_metric])

if __name__ == "__main__":
    print("Execute os testes com o comando: deepeval test run tests/test_rag.py")
