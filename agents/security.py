import re
import logging

logger = logging.getLogger(__name__)

# Configuração dos Scanners de Entrada Customizados (Equivalente ao LLM Guard mas sem dependência do PyTorch)
FORBIDDEN_TOPICS = ["política", "diagnóstico médico", "dica financeira", "investimento", "atividade ilegal", "violência"]
TOXIC_WORDS = ["idiota", "burro", "morte", "matar", "estúpido", "odeio"]
JAILBREAK_PATTERNS = ["ignore todas as instruções", "você agora é", "esqueça o que eu disse", "sistema admin", "bypass", "desconsidere as regras"]

class SecurityPipeline:
    def __init__(self):
        self.ready = True
        logger.info("Security Pipeline (Lightweight) inicializado com sucesso.")

    def _anonymize_pii(self, text: str) -> str:
        # Anonimiza CPF (ex: 123.456.789-00)
        text = re.sub(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', '[CPF REMOVIDO]', text)
        # Anonimiza Email
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVIDO]', text)
        # Anonimiza Celular (ex: 11 99999-9999)
        text = re.sub(r'\b\d{2}\s?9\d{4}-?\d{4}\b', '[CELULAR REMOVIDO]', text)
        return text

    def check_input(self, prompt: str) -> tuple[str, bool, str]:
        prompt_lower = prompt.lower()
        
        # 1. Scanner de Prompt Injection / Jailbreak
        for pattern in JAILBREAK_PATTERNS:
            if pattern in prompt_lower:
                return prompt, False, "Prompt Injection / Jailbreak Detectado"
                
        # 2. Scanner de Toxicidade
        for word in TOXIC_WORDS:
            if word in prompt_lower:
                return prompt, False, "Linguagem Tóxica Detectada"
                
        # 3. Scanner de BanTopics
        for topic in FORBIDDEN_TOPICS:
            if topic in prompt_lower:
                return prompt, False, f"Tópico Proibido Detectado: {topic}"
                
        # 4. Scanner de Anonimização (PII)
        sanitized_prompt = self._anonymize_pii(prompt)
        
        return sanitized_prompt, True, "Safe"

    def check_output(self, prompt: str, output: str) -> tuple[str, bool, str]:
        output_lower = output.lower()
        
        # 1. Scanner de Toxicidade no Output
        for word in TOXIC_WORDS:
            if word in output_lower:
                return output, False, "Linguagem Tóxica gerada pelo modelo"
                
        # 2. Prevenção de Vazamento de PII gerada indevidamente
        sanitized_output = self._anonymize_pii(output)
        
        return sanitized_output, True, "Safe"

# Instância global (singleton) para ser importada no app.py
_security_pipeline = None

def get_security_pipeline():
    global _security_pipeline
    if _security_pipeline is None:
        _security_pipeline = SecurityPipeline()
    return _security_pipeline
