import os, sys
parent_dir = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, parent_dir)  # add repo entrypoint to python path
import uuid
import streamlit as st
import requests
import shutil
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_chroma import Chroma # local
from src.utils.data_ingest_sqlite import ingest_file_sqlite
from src.utils.data_ingest_vectordb import ingest_pdf_vectordb
from src.agents.supervisor import supervisor_agent as chatbot 
from src.configs import config as cfg

# =====Environment variables=====
BACKEND_API = os.getenv("BACKEND_URL", "http://localhost:8000")
TEMP_PATH = "data/temp/"

# =====Utilities=====
def generate_thread_id():
    """generating unique id"""
    return str(uuid.uuid4())

def reset(clear_file: bool = True):
    """Reset chatbot + optionally clear uploaded file widget."""

    if clear_file:
        st.session_state["file_path"] = None

    # reset thread + messages
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

    # Reset the vector db
    cfg.DEFAULT_VECTOR_DB.reset_collection()

    # Delete old files from temp folder
    for name in os.listdir(TEMP_PATH):
        path = os.path.join(TEMP_PATH, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except Exception as e:
            print(f"Warning: Failed to delete temp entry {name}: {e}")

# =====Session Initialization=====
# Setup session states
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
    reset()

if "file_path" not in st.session_state:
    st.session_state["file_path"] = None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Create temp directory
# os.makedirs(TEMP_PATH, exist_ok=True)

# =====Sidebar=====
st.sidebar.title("Settings")
st.sidebar.markdown(f"**Thread ID:** `{st.session_state['thread_id']}`")
# st.sidebar.markdown(f"**Current file path:**`{st.session_state['file_path']}`")

st.sidebar.subheader("📂 Dataset Source")
    
# Upload files and handle it
uploaded_file = st.sidebar.file_uploader(
    "Upload a file",
    type=["csv", "txt", "json", "xls", "xlsx", "pdf"],     
    accept_multiple_files=False,
    key=st.session_state["file_uploader_key"]
)

if uploaded_file:
    filename = uploaded_file.name   # file name
    file_type = filename.split(".")[-1] # file extension
    file_path = os.path.join(TEMP_PATH, filename) # file path
    # os.makedirs(os.path.dirname(temp_path), exist_ok=True) # Create directory 

    # Process only if new file uploaded
    if st.session_state.get("file_path") != file_path:

        with st.sidebar.status(f"Indexing {file_type}…", expanded=True) as status_box:

            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state["file_path"] = file_path
            st.sidebar.success(f"File saved as: `{file_path}`")

            # Ingest pdf to vectordb
            if file_type == "pdf":
                # Reset the vector db
                cfg.DEFAULT_VECTOR_DB.reset_collection()

                # Ingest new pdf
                ids = ingest_pdf_vectordb(
                    file_path=file_path,
                    vector_store=cfg.DEFAULT_VECTOR_DB
                )

            # Ingest into Sqlite DB
            elif file_type in ("csv", "txt", "json", "xls", "xlsx"):
                ingest_file_sqlite(
                    file_path=file_path,
                    db_path=os.path.join(TEMP_PATH,"ingested.db"),
                    table_name=filename.split(".")[0]
                    )
            else:
                status_box.update(label="❌ Ingestion failed", state="error", expanded=False)
            
            # Show the status
            status_box.update(label="✅ File ingested", state="complete", expanded=False)
            # Reset chat but NOT the uploader
            # reset(clear_file=False)
            # st.rerun()
else:
    st.sidebar.info("No file indexed yet.")

# New chat button
if st.sidebar.button("🆕 New Chat", use_container_width=True):
    # Increment key to reset the file_uploader widget
    st.session_state["file_uploader_key"] += 1

    reset(clear_file=True)
    st.rerun()

# =====Main UI======
st.title("Multi-Agent Chatbot")

# Display message history
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input Box
user_input = st.chat_input("Ask a question…")

if user_input:
    # Save user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Streaming config
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Streaming block
    with st.chat_message("assistant"):
        # stream_box = st.empty()
        # stream_box.markdown("⏳ Thinking…")
        status_holder = {"box": None}

        streamed_text = ""
        first_token = False

        # Create payload
        payload = {"messages": [HumanMessage(content=user_input)]}
        def ai_only_stream():
            temp_token = ""
            for (namespace, data) in chatbot.stream(payload, config=CONFIG, subgraphs=True, stream_mode="messages"):
                # Unpack data
                msg, metadata = data

                # skip supervisor messages
                if metadata.get("langgraph_node") == "supervisor":  
                    continue
                
                # Handle tool calls
                if isinstance(msg, ToolMessage):
                    # Get tool name
                    tool_name = getattr(msg, "name", "tool")

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                            )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if (
                    (isinstance(msg, AIMessage) or isinstance(msg, ToolMessage))  and 
                    msg.content and 
                    getattr(msg, "chunk_position", None) != "last"
                    ):

                    #TODO: Handle code tokens
                    token = msg.content
                    # print(msg.content, end="", flush=True)
                    
                    yield msg.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Add assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

st.divider()
st.caption("💡 Powered by LangGraph + Multi-Agent Reasoning")

# To Run: streamlit run src/chatbot.py