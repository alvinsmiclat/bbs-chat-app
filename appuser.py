# appuser.py (Final Version with Professional UI/UX)

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
from langchain.prompts import PromptTemplate

# --- PAGE CONFIGURATION ---
# This is a one-time setup for the page.
st.set_page_config(
    page_title="BBS Document Assistant",
    page_icon="📄", # A relevant emoji for the page icon
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
except (KeyError, FileNotFoundError):
    st.error("🚨 **Error:** Secrets not found. This app is not configured correctly. Please contact the administrator.")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
GOOGLE_FOLDER_ID = "1DldLKFlvopu3dhauAHGtdB5UK87CntNn"

# --- KNOWLEDGE BASE LOGIC ---
@st.cache_resource(show_spinner="Connecting to documents and preparing knowledge base...")
def build_or_load_knowledge_base():
    if not os.path.exists(PERSIST_DIRECTORY):
        # First-time setup message will be shown by the spinner
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
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    return vector_store

# --- MAIN APP LOGIC ---
vector_store = build_or_load_knowledge_base()

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

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": STRICT_PROMPT}
)

# --- UI/UX DESIGN ---

# Sidebar for information and instructions
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This is an intelligent assistant for **Bina Bangsa School**. 
    
    It is connected to a specific set of internal documents and can answer questions based only on their content.
    """)
    
    st.header("How to Use")
    st.markdown("""
    1.  Type your question in the text box on the main page.
    2.  Press Enter to get the answer.
    3.  The assistant will search the documents and provide a response.
    4.  If the answer isn't in the documents, it will say so.
    """)
    st.info("The knowledge base was updated when this app was last deployed. For new documents to be included, the app needs to be refreshed by an admin.")


# Main page title and introduction
st.title("📄 Bina Bangsa School Document Assistant")
st.markdown("### Your intelligent search tool for school policies and documents.")
st.write("---") # A horizontal line for visual separation

# Container for the main interaction
chat_container = st.container()

with chat_container:
    # Get user input
    query = st.text_input(
        "**Ask a question about the content of our documents:**",
        placeholder="e.g., What is the school's policy on password security?"
    )

    if query:
        with st.spinner("Searching the knowledge base for an answer..."):
            try:
                # Get the result from the QA chain
                result = qa_chain.invoke({"query": query})
                
                # Display the answer in a styled box
                st.markdown("#### Answer")
                st.info(result["result"])

                # Display the source documents in an expander
                with st.expander("Click to see the source documents used for this answer"):
                    for doc in result["source_documents"]:
                        # We use st.markdown to format the source nicely
                        st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}`")
                        # Display a snippet of the page content
                        st.markdown(f"> {doc.page_content[:350]}...")
                        st.write("---")

            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")