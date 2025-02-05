import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import streamlit as st

def load_documents(directory):
    """Load documents from the specified directory"""
    loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def process_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts

def create_vector_store(texts):
    """Create and persist the vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="vectorstore"
    )
    vectorstore.persist()
    return vectorstore

def main():
    st.title("Document Ingestion")
    
    # Input for documents directory
    docs_dir = st.text_input(
        "Enter the path to your documents directory:",
        value="docs"
    )
    
    if st.button("Start Ingestion"):
        with st.spinner("Processing documents..."):
            try:
                # Create docs directory if it doesn't exist
                os.makedirs(docs_dir, exist_ok=True)
                
                # Check if directory is empty
                if not os.listdir(docs_dir):
                    st.warning(f"The directory '{docs_dir}' is empty. Please add your documents first.")
                    return
                
                # Load and process documents
                documents = load_documents(docs_dir)
                st.info(f"Loaded {len(documents)} documents")
                
                texts = process_documents(documents)
                st.info(f"Split into {len(texts)} text chunks")
                
                # Create vector store
                vectorstore = create_vector_store(texts)
                st.success("Successfully created and persisted the vector store!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
