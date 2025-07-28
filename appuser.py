# appuser.py (Final Professional UI Version)

# This is the crucial hot-fix for ChromaDB on Streamlit Community Cloud.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import zipfile
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- PAGE CONFIGURATION ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Bina Bangsa School Document Assistant",
    page_icon="📄", # You can use a local image file here as well
    layout="wide"
)

# --- SECRET & ENV SETUP ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API Key not found in secrets. The app cannot function without it.")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
DB_ZIP_PATH = "./chroma_db.zip"

# --- KNOWLEDGE BASE LOGIC (OFFLINE & STABLE) ---
@st.cache_resource(show_spinner="Loading and preparing the knowledge base...")
def load_knowledge_base():
    """
    Unzips and loads the pre-built Chroma database.
    This function is cached so it only runs once per session.
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        if not os.path.exists(DB_ZIP_PATH):
            st.error("FATAL: chroma_db.zip not found in the repository. The app cannot function.")
            st.stop()
        
        st.write("First-time session setup: Unzipping the knowledge base...")
        with zipfile.ZipFile(DB_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall('.')
            os.rename("./chroma_db_build", PERSIST_DIRECTORY)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store

# --- SIDEBAR ---
with st.sidebar:
    # Use a try-except block to handle the case where the logo file is not found
    try:
        st.image("logo.png", width=150) # Assumes your logo file is named logo.png
    except:
        st.title("Bina Bangsa School") # Fallback to text title if logo is not found
    
    st.markdown("---")
    st.title("About this App")
    st.info(
        "This is an intelligent assistant designed to answer questions "
        "about official Bina Bangsa School documents. Simply type your "
        "question in the chat box and get an instant, evidence-based answer."
    )
    
    st.markdown("---")
    with st.expander("Technical Details & Limitations", expanded=False):
        st.write(
            """
            - **Powered by:** OpenAI's GPT-3.5 and LangChain.
            - **Knowledge Base:** This assistant's knowledge is strictly limited to the content of the documents uploaded in its database.
            - **Data Privacy:** Your questions are sent to the OpenAI API for processing. Please refrain from using sensitive personal information.
            - **Accuracy:** The AI strives for accuracy, but it is not infallible. Always verify critical information against the source documents.
            """
        )

# --- MAIN CHAT INTERFACE ---
st.title("Bina Bangsa School Document Assistant 🤖")
st.markdown("Welcome! I am your AI-powered assistant. I can answer questions based on our internal school documents. Please ask me a question below.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LOAD THE KNOWLEDGE BASE AND QA CHAIN ---
vector_store = load_knowledge_base()

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

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": STRICT_PROMPT}
)

# --- CHAT INPUT AND RESPONSE LOGIC ---
if prompt := st.chat_input("Ask a question about the school's policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                
                # Format the response with source documents
                response_text = result["result"]
                sources = [doc.metadata.get('source', 'Unknown').split('/')[-1] for doc in result["source_documents"]]
                # Remove duplicate source names
                unique_sources = list(set(sources)) 
                
                if unique_sources:
                    response_text += f"\n\n**Sources:**\n"
                    for source in unique_sources:
                        response_text += f"- {source}\n"
                
                message_placeholder.markdown(response_text)
            except Exception as e:
                response_text = f"I'm sorry, an error occurred: {e}"
                message_placeholder.markdown(response_text)
                
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})