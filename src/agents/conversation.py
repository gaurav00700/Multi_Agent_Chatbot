import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from typing import Dict, TypedDict, Literal, Annotated, Optional, Sequence, Type, Union, List, Any

import langsmith as ls
from langsmith import traceable
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver, InMemorySaver
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END
from src.tools.local_python_executor import local_python_executor, BASE_BUILTIN_MODULES
from src.utils.prompts import SYSTEM_PROMPT_CONVERSATION
from src.utils.agent_utils import create_agent, get_llm, chatbot
import src.configs.config as cfg

# Initialize LLM
with ls.tracing_context(enabled=True):
    LLM = get_llm(
        llm_provider=cfg.LLM_PROVIDER,
        model_name=cfg.MODEL_NAME,
        api_key=cfg.OPENAI_API_KEY,
        temperature=0.8
    )

# Initialize in-memory checkpointing
checkpointer = InMemorySaver()  

# Create Data Analyst agent
conversation_agent = create_agent(
    llm=LLM,
    tools=[],
    system_prompt= SYSTEM_PROMPT_CONVERSATION,
    checkpointer=checkpointer,
    )

# Creating node for websearch
@traceable(name="Conversation node")
def conversation_node(state: MessagesState) -> Command[Literal["__end__"]]:
    result = conversation_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="conversation_node")
            ]
        },
        goto=END,       # "__end__" END
    )

if __name__ == "__main__":

    # Print the agent
    print(conversation_agent.get_graph().draw_ascii()) 
    
    # Start chat
    chatbot(agent=conversation_agent)