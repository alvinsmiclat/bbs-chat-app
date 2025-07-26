# appuser.py (Final Production Version)

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
# This block securely loads credentials from Streamlit's secrets manager.
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    # The credentials string from secrets is parsed into a dictionary.
    google_creds_dict = json.loads(google_creds_json_str)
    
    # Define a temporary path on the server for the credentials file.
    credentials_path = "google_creds.json" 

    # Write the dictionary to the temporary file for the loader to use.
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)

except (KeyError, FileNotFoundError) as e:
    st.error(f"Missing secrets! Please ensure both OPENAI_API_KEY and GOOGLE_CREDENTIALS_JSON are set in your Streamlit secrets. Error: {e}")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
# IMPORTANT: Replace this with your actual Google Drive Folder ID.
GOOGLE_FOLDER_ID = "1v3Nl5PoC3oD73uZTTrROTit9KD04eK1Ri"

# --- KNOWLEDGE BASE LOGIC ---
# @st.cache_resource is a Streamlit decorator that caches the output of this function.
# This means the knowledge base is only built once, on the app's first run.
# Subsequent user sessions will load the cached result, making the app much faster.
@st.cache_resource(show_spinner="Connecting to documents and building knowledge base...")
def build_or_load_knowledge_base():
    """
    Builds the knowledge base from Google Drive documents if it doesn't already exist on the server.
    If it exists, it loads the database from the local directory.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        st.write("First-time setup: Building the knowledge base. This may take a few minutes...")
        
        # This loader is now correctly configured for both Google Docs and PDFs.
        loader = GoogleDriveLoader(
            folder_id=GOOGLE_FOLDER_ID,
            file_types=["document", "pdf"], # Specifies which file types to process.
            service_account_key=credentials_path, # Uses the secure service account method.
            recursive=False # Set to True if you want to include files in subfolders.
        )
        documents = loader.load()

        if not documents:
            st.error("No compatible documents (Google Docs or PDFs) found in the folder. Please check the Folder ID and ensure the Service Account has 'Viewer' access to the folder.")
            st.stop()
            
        # Split the loaded documents into smaller chunks for processing.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Create embeddings for the chunks using OpenAI.
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Create the Chroma vector store and persist it to the server's disk.
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
    else:
        st.write("Loading existing knowledge base...")
        # If the database already exists, load it from disk.
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        
    return vector_store

# --- MAIN APP LOGIC ---
# A simple check to ensure the placeholder ID has been replaced.
if "YOUR_GOOGLE_DRIVE_FOLDER_ID" in GOOGLE_FOLDER_ID: 
    st.warning("ERROR: Please replace 'YOUR_GOOGLE_DRIVE_FOLDER_ID' in the script with your actual Google Drive folder ID.")
    st.stop()

# Build or load the knowledge base.
vector_store = build_or_load_knowledge_base()

# Create the Question-Answering chain.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

st.success("Knowledge base is ready. You can now ask questions.")

# Create the user interface for asking questions.
query = st.text_input("Ask a question about the content of your documents:")

if query:
    with st.spinner("Searching for the answer..."):
        try:
            # Invoke the QA chain with the user's query.
            result = qa_chain.invoke({"query": query})
            
            # Display the answer.
            st.subheader("Answer")
            st.write(result["result"])

            # Optionally, display the source documents used to generate the answer.
            with st.expander("Show Source Documents"):
                st.write("The answer was generated from the following sources:")
                for doc in result["source_documents"]:
                    st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")