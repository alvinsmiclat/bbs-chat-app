# appuser.py (Final Diagnostic Version 2)

import streamlit as st
import os
import json
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- CONFIGURATION ---
st.set_page_config(page_title="Document Chat", layout="wide")
st.title("📄 Chat with Our Documents")

# --- SECRET & ENV SETUP ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    google_creds_dict = json.loads(google_creds_json_str)
    
    # --- START OF DIAGNOSTIC CODE ---
    # We will get the project_id from the credentials and display it.
    creds_project_id = google_creds_dict.get("project_id")
    st.info(f"Attempting to use credentials for Google Cloud Project ID: '{creds_project_id}'")
    # --- END OF DIAGNOSTIC CODE ---

    credentials_path = "google_creds.json"
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)

except (KeyError, FileNotFoundError) as e:
    st.error(f"Missing secrets! Please ensure both OPENAI_API_KEY and GOOGLE_CREDENTIALS_JSON are set in your Streamlit secrets. Error: {e}")
    st.stop()

PERSIST_DIRECTORY = "./chroma_db"
GOOGLE_FOLDER_ID = "1DIdLKFlvopu3dhauAHGtdB5UK87CntNn"

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

# --- MAIN APP LOGIC ---
if "YOUR_GOOGLE_DRIVE_FOLDER_ID" in GOOGLE_FOLDER_ID: 
    st.warning("ERROR: Please replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID' in the script with your actual Google Drive folder ID.")
    st.stop()

vector_store = build_or_load_knowledge_base()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)
st.success("Knowledge base is ready. You can now ask questions.")
query = st.text_input("Ask a question about the content of your documents:")
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