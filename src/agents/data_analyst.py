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
from src.utils.prompts import SYSTEM_PROMPT_DATA_ANALYST, POSTGRES_PROMPT, SQLITE_PROMPT
from src.utils.agent_utils import create_agent, get_llm, chatbot
import src.configs.config as cfg

# Initialize LLM
with ls.tracing_context(enabled=True):
    LLM = get_llm(
        llm_provider=cfg.LLM_PROVIDER,
        model_name=cfg.MODEL_NAME,
        api_key=cfg.OPENAI_API_KEY,
        temperature=0.0
    )

# Allowed module imports for python executor
CUSTOM_MODULES = ['sqlalchemy', 'sqlite3', "matplotlib", 'dotenv', 'os', 'sys', 'pandas']
AUTHORIZED_IMPORTS = list(set(BASE_BUILTIN_MODULES) | set(CUSTOM_MODULES))

# @tool("python_tool", description="Execute Python code. Inputs: code (str).")
@traceable(run_type="tool", name="Local Python executor")
def python_tool(code: str):
    """Execute Python code safely with restricted imports.

    Args:
        code (str): The code to execute.

    Returns:
        The result of the execution.
    """
    try:
        return local_python_executor(code, AUTHORIZED_IMPORTS)
    except Exception as e:
        return {
            "error": str(e),
            "recovery_plan": (
                "Inspect error → Verify assumptions → Take countermeasures → Retry"
                )
            }
# Build tool
# python_tool = StructuredTool.from_function(
#     func=python_tool,
#     name="python_tool",
#     description="Execute Python code. Inputs: code (str)."
#     )

# Initialize in-memory checkpointing
checkpointer = InMemorySaver()  

# Create Data Analyst agent
data_analyst_agent = create_agent(
    llm=LLM,
    tools=[python_tool],
    system_prompt= SYSTEM_PROMPT_DATA_ANALYST + SQLITE_PROMPT,
    checkpointer=checkpointer,
)

# Creating node for websearch
@traceable(name="Data Analyst node")
def data_analyst_node(state: MessagesState) -> Command[Literal["__end__"]]:
    result = data_analyst_agent.invoke(state)
    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="data_analyst_node")
            ]
        },
        goto=END,       # "__end__", END
    )

if __name__ == "__main__":

    # Print the agent
    print(data_analyst_agent.get_graph().draw_ascii()) 
    
    # Start chat
    chatbot(agent=data_analyst_agent)