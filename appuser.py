# appuser.py (Final, Stable, Offline Version)

# Hot-fix for ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import zipfile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="BBS Document Assistant", page_icon="📄", layout="wide")
st.title("📄 Bina Bangsa School Document Assistant")
st.markdown("### Your intelligent search tool for school policies and documents.")
st.write("---")

# --- Load Secrets ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🚨 **Error:** Secrets not found. This app is not configured correctly. Please contact the administrator.")
    st.stop()

PERSIST_DIRECTORY = "./chroma_db"
DB_ZIP_PATH = "./chroma_db.zip"

# --- KNOWLEDGE BASE LOGIC (NOW OFFLINE) ---
@st.cache_resource(show_spinner="Loading and preparing the knowledge base...")
def load_knowledge_base():
    if not os.path.exists(PERSIST_DIRECTORY):
        if not os.path.exists(DB_ZIP_PATH):
            st.error("FATAL: chroma_db.zip not found in the repository. The app cannot function.")
            st.stop()
        
        with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall('.')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store

# --- MAIN APP LOGIC ---
vector_store = load_knowledge_base()

prompt_template = """
Use the following pieces of context to answer the user's question. This is a closed-book task.
If you don't know the answer based on the context provided, just say that you don't know the answer. Do not try to make up an answer.
---
CONTEXT: {context}
---
QUESTION: {question}
Answer:
"""
STRICT_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": STRICT_PROMPT}
)

# --- UI/UX DESIGN ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.header("About This App")
    st.markdown("This is an intelligent assistant for **Bina Bangsa School**.")
    st.header("How to Use")
    st.markdown("Type your question in the chat box at the bottom of the screen and press Enter.")

# --- CHAT INTERFACE LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question about our documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Searching for an answer..."):
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})