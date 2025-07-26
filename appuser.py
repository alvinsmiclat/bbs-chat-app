# appuser.py (Final Version)

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
    
    # This path is temporary for the Streamlit server instance
    credentials_path = "google_creds.json" 

    # Write the dictionary to a temporary file for the loader
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)

except (KeyError, FileNotFoundError) as e:
    st.error(f"Missing secrets! Please add OPENAI_API_KEY and GOOGLE_CREDENTIALS_JSON to your Streamlit secrets. Error: {e}")
    st.stop()

PERSIST_DIRECTORY = "./chroma_db"
# IMPORTANT: Put your folder ID here!
GOOGLE_FOLDER_ID = "1v3Nl5PoC3oD73uZTTrROTit9KD04eK1Ri"

# --- KNOWLEDGE BASE LOGIC ---
@st.cache_resource(show_spinner="Connecting to documents and building knowledge base...")
def build_or_load_knowledge_base():
    """Build the knowledge base if it doesn't exist, otherwise load it."""
    if not os.path.exists(PERSIST_DIRECTORY):
        st.write("First-time setup: Building the knowledge base. This may take a few minutes...")
        
        # 1. Load Documents using Service Account
        loader = GoogleDriveLoader(
            folder_id=GOOGLE_FOLDER_ID,
            # THIS IS THE KEY CHANGE: Use the service account key
            service_account_key=credentials_path,
            recursive=False
        )
        documents = loader.load()
        if not documents:
            st.error("No documents found. Check folder ID and ensure the service account email has 'Viewer' access to the folder.")
            st.stop()
            
        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 3. Create and Persist Vector Store
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
    st.warning("Please replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID' in the script with your actual Google Drive folder ID.")
    st.stop()

vector_store = build_or_load_knowledge_base()

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

st.success("Knowledge base is ready. You can now ask questions.")

# Chat Interface
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