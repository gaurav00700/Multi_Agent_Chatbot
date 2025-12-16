import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS  # in-memory
from langchain_chroma import Chroma # local
from src.utils.agent_utils import create_agent, get_llm, get_embedding
import src.configs.config as cfg

def ingest_pdf_vectordb(file_path: str, vector_store: VectorStore= cfg.DEFAULT_VECTOR_DB) -> List:
    """
    Ingest the pdf files in a vector DB

    Args:
        file_path(str): Path of pdf file
        vector_store(VectorStore): Vector DB client

    Returns:
        Dict: Dictionary ingested documents
    """

    # Load documents
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Create chunk of document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
    chunks = splitter.split_documents(docs)

    # Create embedding and store them in vectorDB
    docs_ids = vector_store.add_documents(chunks)
    print(f"[OK] Number of document ingested in vector DB: {len(docs_ids)}")

    return docs_ids


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a PDF into the vector database")
    parser.add_argument("--file_path", type=str, default="data/in/SLM_are_Future_of_Agentic_AI.pdf", help="Path to the PDF file to ingest")
    args = parser.parse_args()

    # Ingest data
    ids = ingest_pdf_vectordb(
        file_path=args.file_path #"data/in/SLM_are_Future_of_Agentic_AI.pdf"
    )