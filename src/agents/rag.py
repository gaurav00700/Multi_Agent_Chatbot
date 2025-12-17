import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any

import langsmith as ls
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS  # in-memory
from langchain_chroma import Chroma # local
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import BaseCheckpointSaver, InMemorySaver
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END
from src.utils.prompts import SYSTEM_PROMPT_RAG
from src.utils.agent_utils import create_agent, get_llm, get_embedding, chatbot
from src.tools.rag_tools import rag_tool
import src.configs.config as cfg

# Initialize LLM and embedding model
with ls.tracing_context(enabled=True):
    LLM = get_llm(
        llm_provider=cfg.LLM_PROVIDER,
        model_name=cfg.MODEL_NAME,
        api_key=cfg.OPENAI_API_KEY,
        temperature=0.0
    )

@traceable(run_type="tool", name="retrieval_tool")
def retrieval_tool(query: str) -> dict:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that might be answered from the stored documents.

    Args:
        query (str): Input query
    """
    return rag_tool(query=query, vector_store=cfg.DEFAULT_VECTOR_DB)


# Initialize in-memory checkpointing
checkpointer = InMemorySaver()  

# Create Data Analyst agent
rag_agent = create_agent(
    llm=LLM,
    tools=[retrieval_tool],
    system_prompt= SYSTEM_PROMPT_RAG,
    checkpointer=checkpointer,
)

# Creating node for websearch
@traceable(name="RAG node")
def rag_node(state: MessagesState) -> Command[Literal["__end__"]]:
    result = rag_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="RAG node")
            ]
        },
        goto=END,       # "__end__", END
    )

if __name__ == "__main__":
    
    # Print the agent
    print(rag_agent.get_graph().draw_ascii()) 
    
    # Start chat
    chatbot(agent=rag_agent)