# extractor.py (Run this on your Mac)

import os
import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

SOURCE_DIRECTORY = "./source_documents"
OUTPUT_FILE = "./extracted_data.json"

def main():
    print("--- Starting Text Extraction from Local Documents ---")
    
    # Use a powerful loader that can handle OCR if needed, running on your local machine
    loader = DirectoryLoader(SOURCE_DIRECTORY, glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print(f"No PDF documents found in '{SOURCE_DIRECTORY}'. Exiting.")
        return
    print(f"Successfully loaded and extracted text from {len(documents)} document(s).")
    
    # Convert the LangChain documents to a simple list of dictionaries
    # This format is easy to use in our next step
    output_data = []
    for doc in documents:
        output_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
        
    # Save the clean text to a JSON file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"Clean text has been saved to '{OUTPUT_FILE}'.")
    print("You can now add this JSON file to your GitHub repository.")

if __name__ == "__main__":
    main()