# build_knowledge_base.py

import os
from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file (for your OpenAI API key)
load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in your .env file.")

# The ID of the Google Drive folder
# Replace this with your actual folder ID
GOOGLE_FOLDER_ID = "1v3Nl5PoC3oD73uZTrROTit9KDO4eK1Ri"

# The path where the vector store will be saved
PERSIST_DIRECTORY = "./chroma_db"

def main():
    """
    Main function to build and save the knowledge base.
    """
    print("--- Starting Knowledge Base Creation ---")

    # 1. Load Documents from Google Drive
    print(f"Loading documents from Google Drive Folder: {GOOGLE_FOLDER_ID}...")
    try:
        loader = GoogleDriveLoader(
            folder_id=GOOGLE_FOLDER_ID,
            credentials_path="credentials.json",
            token_path="token.json",
            recursive=False
        )
        documents = loader.load()
        if not documents:
            print("No documents found in the folder. Exiting.")
            return
        print(f"Successfully loaded {len(documents)} documents.")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 2. Split Documents into Chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} chunks.")

    # 3. Create Embeddings and Save to Chroma Vector Store
    print("Creating embeddings and saving to vector store. This may take a few minutes...")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Create the vector store and persist it to disk
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print("--- Knowledge Base Creation Complete! ---")
        print(f"Vector store saved to: {PERSIST_DIRECTORY}")

    except Exception as e:
        print(f"Error creating vector store: {e}")

if __name__ == "__main__":
    main()