# appuser.py (Final Version with Grounding and All Fixes)

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
from langchain.prompts import PromptTemplate # <-- We need this to create the custom prompt

# --- CONFIGURATION ---
st.set_page_config(page_title="Bina Bangsa School Sandbox", layout="wide")
st.title("Bina Bangsa School Sandbox")

# --- SECRET & ENV SETUP ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    google_creds_dict = json.loads(google_creds_json_str)
    credentials_path = "google_creds.json"
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)
except (KeyError, FileNotFoundError) as e:
    st.error(f"Missing secrets! Please ensure both OPENAI_API_KEY and GOOGLE_CREDENTIALS_JSON are set in your Streamlit secrets. Error: {e}")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
# This should be your final, working Google Drive Folder ID.
GOOGLE_FOLDER_ID = "1DldLKFlvopu3dhauAHGtdB5UK87CntNn"

# --- KNOWLEDGE BASE LOGIC ---
@st.cache_resource(show_spinner="Connecting to documents and building knowledge base...")
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
            st.error("No compatible documents (Google Docs or PDFs) found in the folder. Please check the Folder ID and sharing permissions.")
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

# --- MAIN APP LOGIC ---
vector_store = build_or_load_knowledge_base()

# --- START OF NEW/CORRECTED CODE ---
# This template provides strict instructions to the AI.
prompt_template = """
Use the following pieces of context to answer the user's question. This is a closed-book task.
If you don't know the answer based on the context provided, just say that you don't know the answer. Do not use any other information. Do not try to make up an answer.
---
CONTEXT: {context}
---
QUESTION: {question}
Answer:
"""
STRICT_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
# --- END OF NEW/CORRECTED CODE ---

# Create the Question-Answering chain, now with the strict prompt.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    # This is the crucial part that injects our strict instructions.
    chain_type_kwargs={"prompt": STRICT_PROMPT}
)

st.success("Knowledge base is ready. You can now ask questions.")

# Create the user interface for asking questions.
query = st.text_input("Ask a question about the content of your documentss:")

if query:
    with st.spinner("Searching for the answer..."):
        try:
            result = qa_chain.invoke({"query": query})
            st.subheader("Answer")
            st.write(result["result"])
            with st.expander("Show Source Documents"):
                st.write("The answer was generated from the following sources:")
                for doc in result["source_documents"]:
                    st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")