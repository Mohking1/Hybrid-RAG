import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Also load from elastic-start-local/.env if it exists
import pathlib
elastic_env_path = pathlib.Path(__file__).parent / "elastic-start-local" / ".env"
if elastic_env_path.exists():
    load_dotenv(elastic_env_path)

# Configuration file for RAG Pipeline

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "hosts": [os.getenv("ES_LOCAL_URL", os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200"))],
    "api_key": os.getenv("ES_LOCAL_API_KEY"),
    "username": os.getenv("ELASTICSEARCH_USERNAME", os.getenv("ES_LOCAL_USERNAME", "elastic")),
    "password": os.getenv("ES_LOCAL_PASSWORD"),
    "timeout": int(os.getenv("ES_TIMEOUT", "30")),
    "max_retries": int(os.getenv("ES_MAX_RETRIES", "3")),
    "retry_on_timeout": os.getenv("ES_RETRY_ON_TIMEOUT", "True").lower() == "true",
    "verify_certs": os.getenv("ES_VERIFY_CERTS", "False").lower() == "true",
    "port": int(os.getenv("ES_LOCAL_PORT", "9200"))
}

# Gemini Configuration
GEMINI_CONFIG = {
    "embedding_model": "gemini-embedding-001",
    "chat_model": "gemini-2.5-flash",
    "document_understanding_model": "gemini-2.5-flash-lite",
    "embedding_dimension": 768,
    "embedding_task_type": "QUESTION_ANSWERING"
}

# PDF Chunking Configuration
CHUNKING_CONFIG = {
    "max_chunk_size": 400,
    "min_chunk_size": 50,
    "overlap_size": 50
}

# RAG Pipeline Configuration
RAG_CONFIG = {
    "semantic_weight": float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.7")),
    "bm25_weight": float(os.getenv("RAG_BM25_WEIGHT", "0.3")),
    "num_chunks_to_recall": int(os.getenv("RAG_NUM_CHUNKS_TO_RECALL", "150")),
    "final_k": int(os.getenv("RAG_FINAL_K", "20")),
    "reranking_enabled": os.getenv("RAG_RERANKING_ENABLED", "True").lower() == "true",
    "reranker_model": os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
    "batch_size": int(os.getenv("RAG_BATCH_SIZE", "100")),
    "warmup_queries_count": int(os.getenv("RAG_WARMUP_QUERIES_COUNT", "10")),
    "rerank_multiplier": int(os.getenv("RAG_RERANK_MULTIPLIER", "10"))
}

# Streamlit App Configuration
STREAMLIT_CONFIG = {
    "max_file_size_mb": 50,
    "supported_file_types": [".pdf"],
    "max_files": 10,
    "chat_history_limit": 50
}