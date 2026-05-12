import streamlit as st
import requests

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

# URL da API do Flowise
FLOWISE_API_URL = "http://localhost:3000/api/v1/prediction/a8e32462-20fa-4d2a-912a-e5542bfb0c35"

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

# Função para chamar a API do Flowise
def query_flowise(question):
    payload = {
        "question": question
    }
    try:
        response = requests.post(FLOWISE_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "Desculpe, a resposta do servidor veio vazia.")
    except Exception as e:
        return f"🚨 Erro ao conectar com o Flowise: certifique-se de que o Docker está rodando. Detalhes: {e}"

# Recebe o input do usuário na barra de chat
if user_input := st.chat_input("Digite sua pergunta... (Ex: Quais minhas tarefas de prioridade alta?)"):
    
    # Adiciona a mensagem do usuário no histórico e exibe na tela
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Exibe o indicador de 'digitando...' e faz a requisição
    with st.chat_message("assistant"):
        with st.spinner("Analisando bases de dados..."):
            bot_response = query_flowise(user_input)
            st.markdown(bot_response)
    
    # Salva a resposta do bot no histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
