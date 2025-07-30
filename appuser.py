# appuser.py (Final Version built on the user-confirmed stable base)

# This is the crucial hot-fix for ChromaDB on Streamlit Community Cloud.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
# NOTE: PromptTemplate is not needed in this version because we are not using the strict prompt.

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BBS Document Assistant",
    page_icon="📄",
    layout="wide"
)

# --- SECRET & ENV SETUP ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    google_creds_dict = json.loads(google_creds_json_str)
    credentials_path = "google_creds.json"
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)
except (KeyError, FileNotFoundError) as e:
    st.error("🚨 **Error:** Secrets not found. This app is not configured correctly. Please contact the administrator.")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
# This ID is from the version you confirmed was working.
GOOGLE_FOLDER_ID = "1DldLKFlvopu3dhauAHGtdB5UK87CntNn"

# --- KNOWLEDGE BASE LOGIC (This is from your stable code) ---
@st.cache_resource(show_spinner="Connecting to documents and preparing knowledge base...")
def build_or_load_knowledge_base():
    if not os.path.exists(PERSIST_DIRECTORY):
        st.write("First-time setup: Building the knowledge base. This may take a few minutes...")
        
        loader = GoogleDriveLoader(
            folder_id=GOOGLE_FOLDER_ID,
            file_types=["document", "pdf"],
            service_account_key=credentials_path,
            recursive=False
        )
        documents = loader.load()
        if not documents:
            st.error("No compatible documents (Google Docs or PDFs) found in the folder. Please check the folder ID and sharing permissions.")
            st.stop()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        st.write("Loading existing knowledge base...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return vector_store

# --- MAIN APP LOGIC (This is from your stable code) ---
vector_store = build_or_load_knowledge_base()

# The QA chain is now configured WITHOUT the strict prompt, matching your working version.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True # We will keep this True but hide it from the UI.
)

# --- NEW UI/UX DESIGN ---

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_column_width=True)
    st.header("About This App")
    st.markdown("""
    This is an intelligent assistant for **Bina Bangsa School**. 
    
    It is connected to a set of internal documents and can answer questions about their content.
    """)
    st.header("How to Use")
    st.markdown("""
    1.  Type your question in the chat box at the bottom of the screen.
    2.  Press Enter to get the answer.
    3.  Your conversation will be displayed in the chat window.
    """)

st.title("📄 Bina Bangsa School Document Assistant")
st.markdown("### Your intelligent search tool for school policies and documents.")
st.write("---")

# --- NEW CHAT INTERFACE LOGIC ---
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