import os
import tempfile
import streamlit as st
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

# Variavél de ambiente
os.environ["GROQ_API_KEY"] = config("GROQ_API_KEY")

# Criando diretorio para RAG
persistDirectory = "Database"

# Criando funções do app


def processPDF(file):
    # Salvar no disco temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tempFile:
        # Escrevendo em disco o binário do arquivo e armazenando em pasta temporária no disco
        tempFile.write(file.read())
        # Pegando o caminho do arquivo
        tempFilePath = tempFile.name
    # Carregando o arquivo com o Loader
    loader = PyPDFLoader(tempFilePath)
    docs = loader.load()
    # Remoção do arquivo em disco
    os.remove(tempFilePath)
    # Quebrando em chunks e criando overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
    )
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

# Carregando banco de dados existente


def load_existing_vector_store():
    if os.path.exists(os.path.join(persistDirectory)):
        vectorStore = Chroma(
            persist_directory=persistDirectory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vectorStore
    return None


def add_to_vector_store(chunks, vectorStore=None):
    if vectorStore:
        vectorStore.add_documents(chunks)
    else:
        vectorStore = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persistDirectory,
        )
    return vectorStore

# Respondendo perguntas


def ask_question(model, query, vectorStore):
    # Instaciar modelo
    llm = ChatGroq(model=model)
    retriever = vectorStore.as_retriever()

    system_prompt = """
    Use o contexto para responder as perguntas.
    Se não encontrar uma resposta no contexto,
    pesquise na internet sobre o assunto caso não tenha na base de dados.
    Responda em formato de markdown e com visualizações elaboradas e interativas.
    Contexto: {context}
    """
    # Enviando a mensagem para o sistema e as mensagens do streamlit
    messages = [("system", system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get("role"), message.get("content")))
    messages.append(("human", "{input}"))
    # Gera um chat prompt template
    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({"input": query})
    return response.get("answer")

# Interface do usuário


# Carregando documentos ja existentes no banco
vectorStore = load_existing_vector_store()


# Configurações de página
st.set_page_config(
    page_title="ChatBot",
    page_icon="📲",
)
# Cabeçalho
st.header("🤖 Tire suas dúvidas")

# Dentro do sidebar, adicionando componentes
with st.sidebar:
    # Upload de arquivos
    st.header("📂 Upload de arquivos")
    uploaded_files = st.file_uploader(
        label="Faça upload de arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    # Processando upload de arquivos
    if uploaded_files:
        # Spinner enquanto faz o processamento
        with st.spinner("Processando documentos..."):
            # Lista de chunks
            allChunks = []
            # Percorrendo arquivos que o usuário enviou
            for uploaded_file in uploaded_files:
                # Quebra o primeiro e adiciona no allChunks e assim sucessivamente
                chunks = processPDF(file=uploaded_file)
                allChunks.extend(chunks)
            # Embedding
            vectorStore = add_to_vector_store(
                chunks=allChunks,
                vectorStore=vectorStore,
            )
            # Quando sair do allChunks irá mostrar os pedaços
            st.write("Processamento concluído! Detalhes:", allChunks)

    # Listando modelos de llm
    modelOptions = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
    ]
    # Selecionando modelos de llm
    selected_model = st.sidebar.selectbox(
        label="Selecione o modelo de IA",
        options=modelOptions,
    )

if "messages" not in st.session_state:
    # Armazena o histórico de mensagens. caso não tenha cria uma lista vazia
    st.session_state["messages"] = []

# Chat embaixo para o usuário escrever
questionUser = st.chat_input("Como posso ajudá-lo hoje?")

if vectorStore and questionUser:
    # Pegar toda o histórico de conversa e mostra ao usuário
    for message in st.session_state.messages:
        # Role retorna se a mensagem foi da IA ou user. o content mostra o conteúdo de fato
        st.chat_message(message.get("role")).write(message.get("content"))

    # Crio a ultima pergunta que o usuário enviou
    st.chat_message("user").write(questionUser)
    # Adicionando na memória
    st.session_state.messages.append({"role": "user", "content": questionUser})

    with st.spinner("Buscando melhor resposta..."):
        response = ask_question(
            model=selected_model,
            query=questionUser,
            vectorStore=vectorStore,
        )

        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "ai", "content": response})