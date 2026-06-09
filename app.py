from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
try:
    from langfuse import observe
except ImportError:
    from langfuse.decorators import observe
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

# Recebe o input do usuário na barra de chat
if user_input := st.chat_input("Digite sua pergunta... (Ex: Quais minhas tarefas de prioridade alta?)"):
    
    # Adiciona a mensagem do usuário no histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Exibe o indicador de 'digitando...' e faz a requisição
    with st.chat_message("assistant"):
        with st.spinner("Analisando bases de dados com agentes inteligentes..."):
            bot_response = query_assistant(user_input)
            st.markdown(bot_response)
            # Força o envio imediato de todos os traces do Langfuse e LiteLLM
            flush_traces()
    
    # Salva a resposta do bot no histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

