import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any

from langsmith import traceable
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS  # in-memory
from langchain_chroma import Chroma # local
from langchain_core.tools import tool
from src.utils.agent_utils import create_agent, get_llm, get_embedding
import src.configs.config as cfg

# @tool("rag_tool", description="Retrieve relevant information from database for a given query")
# @traceable(run_type="tool", name="rag_tool")
def rag_tool(query: str, vector_store: VectorStore) -> dict:
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.

    Args:
        query (str): Input query
        vector_store (VectorStore): Vector store as knowledge base 
    Returns:
        dict: Dictionary containing query, context, metadata
    """
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
        )

    # Retriever results
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }

# Build tool
# python_tool = StructuredTool.from_function(
#     func=rag_tool,
#     name="rag_tool",
#     description="Execute Python code. Inputs: code (str)."
#     )

if __name__ == "__main__":
    
    pass