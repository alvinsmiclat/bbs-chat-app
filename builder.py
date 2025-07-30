# builder.py (Runs on Streamlit Cloud to create the Linux-compatible DB)

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Knowledge Base Builder", layout="wide")
st.title("Admin Tool: Knowledge Base Builder")

# --- Load Secrets ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Failed to load secrets: {e}")
    st.stop()

PERSIST_DIRECTORY = "./chroma_db_build"
INPUT_JSON = "./extracted_data.json"

if st.button("Build Knowledge Base from JSON"):
    if not os.path.exists(INPUT_JSON):
        st.error(f"FATAL: '{INPUT_JSON}' not found in repository. Please run the local extractor script first.")
        st.stop()
        
    with st.spinner("Loading text data and building knowledge base..."):
        try:
            # 1. Load the clean text from our JSON file
            with open(INPUT_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert the data back into LangChain Document objects
            documents = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]
            st.write(f"Loaded {len(documents)} document sections from JSON file.")

            # 2. Split (optional but good practice)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            st.write(f"Split documents into {len(splits)} chunks.")

            # 3. Create Vector Store (on the Linux server)
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            st.success("Knowledge Base built successfully in Linux format!")

            # 4. Zip and provide a download link
            zip_filename = "chroma_db"
            shutil.make_archive(zip_filename, 'zip', PERSIST_DIRECTORY)
            with open(f"{zip_filename}.zip", "rb") as f:
                st.download_button(
                    label="Download Linux-compatible chroma_db.zip",
                    data=f,
                    file_name="chroma_db.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")