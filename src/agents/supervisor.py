import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path

import langsmith as ls
from langsmith import traceable
from pydantic import BaseModel, Field
import operator
from typing import Callable, Literal, Optional, Sequence, Type, TypeVar, Union, cast
from typing_extensions import Annotated, TypedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import BaseCheckpointSaver, InMemorySaver
from src.agents.data_analyst import data_analyst_node
from src.agents.conversation import conversation_node
from src.agents.rag import rag_node
from src.utils.agent_utils import get_llm, chatbot
from src.utils.prompts import SYSTEM_PROMPT_SUPERVISOR
import src.configs.config as cfg

# Initialize LLM
with ls.tracing_context(enabled=True):
    LLM = get_llm(
        llm_provider=cfg.LLM_PROVIDER,
        model_name=cfg.MODEL_NAME,
        api_key=cfg.OPENAI_API_KEY,
        temperature=0.5
    )

# List of agents we want to manage
MEMBERS = ["data_analyst", "conversation", "rag", ]     # "FINISH"

class AgentState(TypedDict):
    """Graph Sate"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    worker_hist: Annotated[list[str], operator.add]

class Router(BaseModel):
    """Worker to route to next"""
    next: Literal["data_analyst", "conversation", "rag"] = Field(description="The next worker to route to.")   # "FINISH"

@traceable(name="Supervisor agent")
def supervisor_node(state: AgentState) -> Command[Literal["data_analyst", "conversation", "rag"]]:
    """Supervisor agent to route to next worker based on user request and current state."""

    # Create prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT_SUPERVISOR.format(MEMBERS=MEMBERS, WORKER_HIST=state["worker_hist"]))] + state["messages"]

    # Invoke the LLM
    response = LLM.with_structured_output(Router).invoke(messages)
    
    # Check for next worker to route to
    goto = response.next

    print(f"[INFO] Supervisor: Next agent ==> {goto} ")
    # print(f"\nWorker history: {state["worker_hist"]}")

    # if goto == "FINISH": #or len(set(state["worker_hist"][:-2])) == 1 :
    #     goto = END

    return Command(update={"worker_hist": [goto]}, goto=goto)

# Building the supervisor agent
# Graph initializing
builder = StateGraph(MessagesState) 

# Add nodes
builder.add_node("supervisor", supervisor_node)
builder.add_node("data_analyst", data_analyst_node)
builder.add_node("conversation", conversation_node)
builder.add_node("rag", rag_node)

# Add edges
builder.add_edge(START, "supervisor")

#Compile
checkpointer = InMemorySaver()
supervisor_agent = builder.compile(checkpointer=checkpointer)

print(supervisor_agent.get_graph().draw_ascii()) # Print the agent

if __name__ == "__main__":

    # Start chat
    chatbot(
        agent=supervisor_agent,
        initial_state = {
            "messages": [],
            "worker_hist": []
            }
        )