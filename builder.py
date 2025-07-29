# builder.py (Your one-time tool to build the database on the server)

# Hot-fix for ChromaDB on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import shutil
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Knowledge Base Builder", layout="wide")
st.title("Admin Tool: Knowledge Base Builder")
st.warning("This app is for admins only. Use it to create the database, then download it.")

# --- Load Secrets ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    google_creds_json_str = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    google_creds_dict = json.loads(google_creds_json_str)
    credentials_path = "google_creds.json"
    with open(credentials_path, "w") as f:
        json.dump(google_creds_dict, f)
except Exception as e:
    st.error(f"Failed to load secrets: {e}")
    st.stop()

PERSIST_DIRECTORY = "./chroma_db_build"
GOOGLE_FOLDER_ID = "1DldLKFlvopu3dhauAHGtdB5UK87CntNn" # Your sandbox folder ID

if st.button("Build Knowledge Base"):
    if os.path.exists(PERSIST_DIRECTORY):
        st.info("Existing database found. Deleting it to rebuild.")
        shutil.rmtree(PERSIST_DIRECTORY)
        
    with st.spinner("Connecting to Google Drive and building knowledge base... This can take several minutes."):
        try:
            # 1. Load Documents
            loader = GoogleDriveLoader(
                folder_id=GOOGLE_FOLDER_ID,
                file_types=["document", "pdf"],
                service_account_key=credentials_path,
                recursive=False
            )
            documents = loader.load()
            if not documents:
                st.error("No compatible documents found in Google Drive.")
                st.stop()
            st.write(f"Loaded {len(documents)} document(s) successfully.")

            # 2. Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            st.write(f"Split documents into {len(splits)} chunks.")

            # 3. Create Vector Store
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            st.success("Knowledge Base built successfully!")

            # 4. Zip the directory
            zip_filename = "chroma_db"
            shutil.make_archive(zip_filename, 'zip', PERSIST_DIRECTORY)
            st.success(f"Database has been zipped into '{zip_filename}.zip'.")

            # 5. Provide a download link
            with open(f"{zip_filename}.zip", "rb") as f:
                st.download_button(
                    label="Download chroma_db.zip",
                    data=f,
                    file_name="chroma_db.zip",
                    mime="application/zip"
                )
        except Exception as e:
            st.error(f"An error occurred during the build process: {e}")