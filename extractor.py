# extractor.py (Final Corrected Version with OCR)

import os
import json
# We are using a more powerful loader that can perform OCR on scanned documents.
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader

SOURCE_DIRECTORY = "./source_documents"
OUTPUT_FILE = "./extracted_data.json"

def main():
    print("--- Starting Text Extraction from Local Documents (with OCR) ---")
    
    # This loader is configured to use the Unstructured library for all PDFs.
    # It will automatically perform OCR on scanned documents.
    loader = DirectoryLoader(
        SOURCE_DIRECTORY, 
        glob="./*.pdf", 
        loader_cls=UnstructuredPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    print("Loading and processing documents... This may take a moment.")
    documents = loader.load()

    if not documents:
        print(f"No PDF documents found in '{SOURCE_DIRECTORY}'. Exiting.")
        return
    print(f"\nSuccessfully loaded and extracted text from {len(documents)} document(s).")
    
    output_data = []
    for doc in documents:
        output_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"Clean text has been saved to '{OUTPUT_FILE}'.")
    print("You can now add this JSON file to your GitHub repository.")

if __name__ == "__main__":
    main()