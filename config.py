import os

# Configuration file for RAG Pipeline

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "hosts": [os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")],
    "api_key": os.getenv("ES_LOCAL_API_KEY"),
    "username": os.getenv("ELASTICSEARCH_USERNAME", "elastic"),
    "password": os.getenv("ES_LOCAL_PASSWORD", "changeme"),
    "timeout": 30,
    "max_retries": 3,
    "retry_on_timeout": True,
    "verify_certs": False
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
    "semantic_weight": 0.8,
    "bm25_weight": 0.2,
    "num_chunks_to_recall": 150,
    "final_k": 20,
    "reranking_enabled": True,
    "reranker_model": "BAAI/bge-reranker-v2-m3"
}

# Streamlit App Configuration
STREAMLIT_CONFIG = {
    "max_file_size_mb": 50,
    "supported_file_types": [".pdf"],
    "max_files": 10,
    "chat_history_limit": 50
}