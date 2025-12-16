import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from typing import Annotated, List
from typing_extensions import TypedDict

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any
from langsmith import traceable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import BaseCheckpointSaver, InMemorySaver


# @traceable(name="Creating Chatbot")
def get_llm(
    llm_provider: Literal["openai", "ollama"], 
    model_name: str , 
    api_key: str , 
    **kwargs
    ):
    """Create an LLM instance"""

    if llm_provider == "openai":
        return ChatOpenAI(
            api_key=api_key,
            model=model_name,
            **kwargs
        )
    elif llm_provider == "ollama":
        return ChatOllama(
            model=model_name,
            **kwargs
        )
    else:
        raise NotImplementedError
    
def get_embedding(
    llm_provider: Literal["openai", "ollama"], 
    embedding_model: str , 
    api_key: str , 
    **kwargs
    ):
    """Create an embedding model instance"""

    if llm_provider == "openai":
        return OpenAIEmbeddings(
            api_key=api_key,
            model=embedding_model,
            **kwargs
        )
    elif llm_provider == "ollama":
        return OllamaEmbeddings(
            model=embedding_model,
            **kwargs
        )
    else:
        raise NotImplementedError

def chatbot(agent: StateGraph, initial_state: MessagesState = None, config: dict = {"configurable": {"thread_id": "1"}}) -> None:
    """ Functions for multi turn conversation using Langgraph agent

    Args:
        agent (StateGraph): Name of Langgraph agent
        config (dict): Configuration of agent

    Returns:
        None
    """    
    # Initialize the worker history
    if initial_state is None:
        initial_state = {
            "messages": [],
            "worker_hist": []
        }
    
    # Conversation loop
    while True:
        
        # User input
        query = input("User👨: ").strip()
        
        # Termination condition
        if query.lower() in ("exit", "quit", "bye"):
            break

        # Create payload
        # update state
        initial_state["messages"] = [HumanMessage(content=query)]
        # payload = {"messages": [HumanMessage(content=query)]}
        
        # ===Invoke chatbot with streaming===
        print("Agent🤖:", end=" ", flush=True)
        final_text = ""

        # for msg, metadata in agent.stream(payload, config=config, stream_mode="messages"):
        for namespace, data in agent.stream(initial_state, config=config, subgraphs=True, stream_mode="messages"):
            
            # Unpack data
            msg, metadata = data

            # skip supervisor messages
            if metadata.get("langgraph_node") == "supervisor":  
                continue
        
            # Handle tool calls
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                print(f"\n[Using tool🔨: {tool_name}]\n", flush=True)
                continue

            # Handle AI tokens
            if (
                (isinstance(msg, AIMessage) or isinstance(msg, ToolMessage)) and 
                msg.content
                ):                
                print(msg.content, end="", flush=True)
                final_text += msg.content

        if not final_text:
            print("No answer", end="")
        print("\n-----------------\n")

        # ===Invoke chatbot without streaming===
        # result = agent.invoke(payload, config=config)

        # # Print the agent message
        # out = (
        #     result["messages"][-1].content or
        #     "No answer"
        #     )
        # print('Agent🤖:', out)

class AgentState(TypedDict):
    """Graph Sate"""
    # messages: Annotated[List, add_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# @traceable(name="Langgraph graph builder")
def create_agent(
    system_prompt: str,
    llm: BaseChatModel ,
    llm_schema: BaseModel | None = None,
    tools: List = [],
    checkpointer: BaseCheckpointSaver = None,
    **kwargs
    )-> StateGraph:
    """Langgraph graph builder

    Args:
        system_prompt (str): Prompt for LLM.
        llm (BaseChatMode): LLM instance.
        llm_schema (List, BaseModel | TypedDict | None): Schema of LLM. Defaults to None.
        tools (List, optional): List for tool. Defaults to [].
        checkpointer (BaseCheckpointSaver, optional): Memory saver for agent. Defaults to None.
        kwargs: Additional arguments.

    Returns:
        StateGraph: Agent graph builder
    """
    
    # Enable structured output if schema is provided
    if llm_schema: 
        llm = llm.with_structured_output(llm_schema)

    # Bind tools with LLM
    llm = llm.bind_tools(tools=tools)

    # Define the nodes
    # LLM node
    def llm_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        payload = [SystemMessage(content=system_prompt)] + messages
        response = llm.invoke(payload)
        return {"messages": response}
    
    # Tool node
    tools_node = ToolNode(tools=tools)

    LLM_NODE = "LLM_NODE"
    TOOLS_NAME = "tools"
    
    # Initialize graph
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node(LLM_NODE, llm_node)
    builder.add_node(TOOLS_NAME, tools_node)
    builder.add_conditional_edges(
        LLM_NODE,
        tools_condition
    )

    # Add edges
    builder.add_edge(TOOLS_NAME, LLM_NODE)

    # Define entry point
    builder.set_entry_point(LLM_NODE)
    
    # Compile 
    agent = builder.compile(checkpointer=checkpointer)

    return agent