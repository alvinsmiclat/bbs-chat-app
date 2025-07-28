# appuser.py (Final Version with Professional UI)

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
# Sets the browser tab title, icon, and layout for a professional look.
st.set_page_config(
    page_title="Bina Bangsa School Document Assistant",
    page_icon="📄",
    layout="wide"
)

# --- SECRET & ENV SETUP ---
# Securely loads credentials from Streamlit's secrets manager.
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    google_creds_dict = json.loads(google_creds_json_str)
    
    credentials_path = "google_creds.json"
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)

except (KeyError, FileNotFoundError) as e:
    st.error(f"Missing secrets! Please ensure both OPENAI_API_KEY and GOOGLE_CREDENTIALS_JSON are set in your Streamlit secrets. The app cannot function without them.")
    st.stop()

# --- CONSTANTS ---
PERSIST_DIRECTORY = "./chroma_db"
# This should be your final, working Google Drive Folder ID.
GOOGLE_FOLDER_ID = "1DldLKFlvopu3dhauAHGtdB5UK87CntNn"

# --- KNOWLEDGE BASE LOGIC ---
# This function connects to Google Drive and builds the knowledge base.
# It's cached by Streamlit, so it only runs on the first load or when the app is rebuilt.
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
            st.error("No compatible documents (Google Docs or PDFs) found. Please check the Folder ID and sharing permissions.")
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
        # This message is removed for a cleaner UI on subsequent loads.
        pass
    
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vector_store

# --- SIDEBAR ---
# This section creates the professional sidebar with logo and information.
with st.sidebar:
    try:
        st.image("logo.png", width=150) # Assumes you have a logo.png file in your repo
    except:
        st.title("Bina Bangsa School") # Fallback if logo.png is not found
    
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
            - **Knowledge Base:** This assistant's knowledge is strictly limited to the content of the documents in its connected Google Drive folder.
            - **Data Privacy:** Your questions are sent to the OpenAI API for processing. Please refrain from using sensitive personal information.
            - **Accuracy:** The AI strives for accuracy, but it is not infallible. Always verify critical information.
            """
        )

# --- MAIN CHAT INTERFACE ---
st.title("Bina Bangsa School Document Assistant 🤖")
st.markdown("Welcome! I am your AI-powered assistant. I can answer questions based on our internal school documents. Please ask me a question below.")

# Initialize chat history in session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages from history on every app rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LOAD THE KNOWLEDGE BASE AND QA CHAIN ---
vector_store = build_or_load_knowledge_base()

# This is the strict prompt to prevent the AI from using its general knowledge.
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

# Create the Question-Answering chain with the strict prompt.
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=False, # We set this to False as requested.
    chain_type_kwargs={"prompt": STRICT_PROMPT}
)

# --- CHAT INPUT AND RESPONSE LOGIC ---
# This captures the user's input from the chat box.
if prompt := st.chat_input("Ask a question about school policies..."):
    # Add the user's message to the chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message in the chat.
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display the assistant's response.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # Get the answer from the QA chain.
                result = qa_chain.invoke({"query": prompt})
                response_text = result["result"]
                message_placeholder.markdown(response_text)
            except Exception as e:
                response_text = f"I'm sorry, an error occurred: {e}"
                message_placeholder.markdown(response_text)
                
    # Add the assistant's response to the chat history.
    st.session_state.messages.append({"role": "assistant", "content": response_text})