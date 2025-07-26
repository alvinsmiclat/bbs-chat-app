import streamlit as st
import os
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- App Configuration ---
st.set_page_config(page_title="Chat with Your Google Drive Documents", layout="wide")
st.title("📄 Chat with Your Google Drive Documents")
st.write("This app allows you to chat with the documents in a specified Google Drive folder.")

# --- API Key Management ---
# Try to get the API key from Streamlit secrets, otherwise from environment variables.
# This is more secure than hardcoding the key in the script.
try:
    # Recommended: Use Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    # Fallback: Use environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key is not set. Please set it in your Streamlit secrets or as an environment variable.")
    st.stop() # Stop the app if the key is not available

# --- Helper Functions ---
# Use a session state to cache the QA chain and avoid re-creating it on every interaction.
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

def load_and_split_docs(folder_id):
    """Loads documents from a Google Drive folder and splits them into chunks."""
    try:
        if not os.path.exists("credentials.json"):
            st.error("Error: The 'credentials.json' file was not found. Please follow the setup instructions.")
            return None

        loader = GoogleDriveLoader(
            folder_id=folder_id,
            credentials_path="credentials.json",
            token_path="token.json",
            recursive=False
        )
        docs = loader.load()

        if not docs:
            st.warning("No documents found in the specified Google Drive folder.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        return splits
    except Exception as e:
        st.error(f"An error occurred while loading documents: {e}")
        return None

def create_vector_store(documents):
    """Creates a Chroma vector store from the document splits."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None

def create_qa_chain(vector_store):
    """Creates the RetrievalQA chain."""
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_api_key
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"An error occurred while creating the QA chain: {e}")
        return None

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    folder_id = st.text_input("Enter your Google Drive Folder ID:")

    if st.button("Load Documents"):
        if not folder_id:
            st.warning("Please enter your Google Drive Folder ID.")
        else:
            with st.spinner("Accessing Google Drive, loading documents, and building knowledge base... Please wait."):
                documents = load_and_split_docs(folder_id)
                if documents:
                    vector_store = create_vector_store(documents)
                    if vector_store:
                        st.session_state.qa_chain = create_qa_chain(vector_store)
                        if st.session_state.qa_chain:
                            st.success("Documents loaded successfully! You can now ask questions.")
                        else:
                            st.error("Failed to create the Question-Answering chain.")
                    else:
                        st.error("Failed to create the vector store.")
                else:
                    st.error("Failed to load or process documents from Google Drive.")

# --- Main Chat Interface ---
st.header("Ask a Question")

if st.session_state.qa_chain is None:
    st.info("Please enter your Google Drive Folder ID in the sidebar and click 'Load Documents'.")
else:
    query = st.text_input("Ask a question about the content of your documents:")

    if query:
        with st.spinner("Searching for the answer..."):
            try:
                result = st.session_state.qa_chain({"query": query})
                st.subheader("Answer")
                st.write(result["result"])

                with st.expander("Show Source Documents"):
                    st.write("The answer was generated from the following sources:")
                    for doc in result["source_documents"]:
                        st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:300] + "...")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {e}")