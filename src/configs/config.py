import os, sys
project_root = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, project_root)  # add repo entrypoint to python path
from dotenv import load_dotenv
from langchain_chroma import Chroma # local
from src.utils.agent_utils import get_embedding, get_llm

# Load environment variables from .env file
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy_key")

# Postgres DB
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

# SQlite DB
DB_PATH = os.getenv("DB_PATH", "data/temp/ingested.db")

# Langsmith
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

# Create default LLM model
DEFAULT_LLM = get_llm(
    llm_provider=LLM_PROVIDER,
    model_name=MODEL_NAME,
    api_key=OPENAI_API_KEY,
)

# Create default embedding model
DEFAULT_EMBEDDING = get_embedding(
    llm_provider=LLM_PROVIDER,
    embedding_model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY,
)

# Create default vector store
DEFAULT_VECTOR_DB = Chroma(
    collection_name="collection",
    embedding_function=DEFAULT_EMBEDDING,
    persist_directory="data/vectordb",   # If not provided -> in-memory mode
    )