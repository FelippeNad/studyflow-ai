from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
try:
    from langfuse.decorators import observe
except ImportError:
    from langfuse import observe
from agents.study_crew import run_study_crew

# Configurações da página
st.set_page_config(
    page_title="StudyFlow AI",
    page_icon="🎓",
    layout="centered"
)

# Estilos Customizados CSS
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho da aplicação
st.markdown('<p class="main-title">StudyFlow AI 🎓</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Seu assistente inteligente para regras acadêmicas, notas e tarefas.</p>', unsafe_allow_html=True)

# Inicializa o histórico de chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Eu sou o **StudyFlow AI**. Como posso te ajudar com a faculdade hoje?"}
    ]

# Renderiza as mensagens do histórico na tela
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def flush_traces():
    try:
        from langfuse import Langfuse
        Langfuse().flush()
    except Exception:
        pass
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

# Função observada pelo Langfuse para processar a pergunta com CrewAI
@observe(name="Perguntar ao Assistente")
def query_assistant(question):
    try:
        response = run_study_crew(question)
        return response
    except Exception as e:
        return (
            f"🚨 Erro ao executar a equipe de agentes acadêmicos.\n\n"
            f"Certifique-se de que o **LM Studio** está aberto e com o Local Server ativo em `http://localhost:1234`.\n\n"
            f"Detalhes: {e}"
        )

from agents.security import get_security_pipeline

@st.cache_resource
def load_security_pipeline():
    return get_security_pipeline()

# Recebe o input do usuário na barra de chat
if user_input := st.chat_input("Digite sua pergunta... (Ex: Quais minhas tarefas de prioridade alta?)"):
    
    # Adiciona a mensagem do usuário no histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1. SEGURANÇA: Validar Input do Usuário
    sec_pipeline = load_security_pipeline()
    sanitized_input, is_valid_input, input_reason = sec_pipeline.check_input(user_input)
    
    with st.chat_message("assistant"):
        if not is_valid_input:
            # Bloqueia e exibe motivo
            bot_response = f"🚨 **Acesso Bloqueado:** {input_reason}\n\n*Por favor, faça perguntas apenas sobre o escopo acadêmico do sistema.*"
            st.markdown(bot_response)
        else:
            # Exibe o indicador de 'digitando...' e faz a requisição usando o input sanitizado (PII removido se houver)
            with st.spinner("Analisando bases de dados com agentes inteligentes..."):
                bot_response_raw = query_assistant(sanitized_input)
                
                # 2. SEGURANÇA: Validar Output do Agente
                sanitized_output, is_valid_output, output_reason = sec_pipeline.check_output(sanitized_input, bot_response_raw)
                
                if not is_valid_output:
                    bot_response = f"🚨 **Resposta Bloqueada:** O sistema de segurança interceptou a resposta. Motivo: {output_reason}"
                else:
                    bot_response = sanitized_output
                    
                st.markdown(bot_response)
                # Força o envio imediato de todos os traces do Langfuse e LiteLLM
                flush_traces()
    
    # Salva a resposta do bot no histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

