# app.py

import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- CONFIGURATION ---
# The app will get the API key from Streamlit secrets when deployed
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback for local development if secrets aren't set
    openai_api_key = os.getenv("OPENAI_API_KEY", "YOUR_DEFAULT_KEY_IF_NEEDED")

if not openai_api_key:
    st.error("OpenAI API key is not configured. Please set it in Streamlit secrets.")
    st.stop()
    
PERSIST_DIRECTORY = "./chroma_db"

# --- CACHING ---
# Cache the expensive resources to load them only once
@st.cache_resource
def load_knowledge_base():
    """Loads the pre-built Chroma vector store from disk."""
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Knowledge base not found at {PERSIST_DIRECTORY}. Please run the build script first.")
        st.stop()
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store

@st.cache_resource
def create_qa_chain(_vector_store): # Underscore to indicate it's from a cached resource
    """Creates the RetrievalQA chain from the loaded vector store."""
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=openai_api_key
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(),
        return_source_documents=True
    )
    return qa_chain


# --- APP LAYOUT ---
st.set_page_config(page_title="Document Chat", layout="wide")
st.title("📄 Chat with Our Documents")
st.write("This app is connected to a fixed set of documents. Ask any question about their content.")

# --- MAIN LOGIC ---
vector_store = load_knowledge_base()
qa_chain = create_qa_chain(vector_store)

# Chat Interface
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching for the answer..."):
        try:
            result = qa_chain({"query": query})
            st.subheader("Answer")
            st.write(result["result"])

            with st.expander("Show Source Documents"):
                st.write("The answer was generated from the following sources:")
                for doc in result["source_documents"]:
                    st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content[:300] + "...")
        except Exception as e:
            st.error(f"An error occurred: {e}")