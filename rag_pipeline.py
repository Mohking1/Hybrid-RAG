from dotenv import load_dotenv

load_dotenv()
"""
End-to-end RAG pipeline script based on the guide notebook.
This script covers:
1. Setup and imports
2. VectorDB and ContextualVectorDB classes
3. Basic RAG retrieval and evaluation
4. Contextual Embeddings and evaluation
5. Contextual BM25 hybrid search and evaluation
6. Reranking step
"""

import json
import os
import pathlib
import pickle
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

# External libraries
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import configuration - config.py handles all environment variables and defaults
from config import ELASTICSEARCH_CONFIG, GEMINI_CONFIG, RAG_CONFIG


# --- VectorDB Class ---
class VectorDB:
    def __init__(self, name: str, gemini_model=None):
        if gemini_model is None:
            gemini_model = GEMINI_CONFIG["embedding_model"]
        self.client = genai.Client()
        self.model = gemini_model
        self.output_dim = GEMINI_CONFIG["embedding_dimension"]
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, dataset: list[dict[str, Any]]):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return
        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)
        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for doc in dataset:
                for chunk in doc["chunks"]:
                    texts_to_embed.append(chunk["content"])
                    metadata.append(
                        {
                            "doc_id": doc["doc_id"],
                            "original_uuid": doc["original_uuid"],
                            "chunk_id": chunk["chunk_id"],
                            "original_index": chunk["original_index"],
                            "content": chunk["content"],
                        }
                    )
                    pbar.update(1)
        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()
        print(
            f"Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}"
        )

    def _embed_and_store(self, texts: list[str], data: list[dict[str, Any]]):
        batch_size = RAG_CONFIG["batch_size"]  # Use configurable batch size
        embeddings = []
        with tqdm(total=len(texts), desc="Embedding chunks (Gemini)") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=GEMINI_CONFIG["embedding_dimension"],
                        task_type=GEMINI_CONFIG["embedding_task_type"],
                    ),
                )
                for embedding_obj in result.embeddings:
                    values = np.array(embedding_obj.values)
                    normed = values / np.linalg.norm(values)
                    embeddings.append(normed)
                pbar.update(len(batch))
        self.embeddings = embeddings
        self.metadata = data

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            result = self.client.models.embed_content(
                model=self.model,
                contents=[query],
                config=types.EmbedContentConfig(
                    output_dimensionality=GEMINI_CONFIG["embedding_dimension"],
                    task_type=GEMINI_CONFIG["embedding_task_type"],
                ),
            )
            values = np.array(result.embeddings[0].values)
            query_embedding = values / np.linalg.norm(values)
            self.query_cache[query] = query_embedding
        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")
        similarities = np.dot(np.array(self.embeddings), query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])


# --- Pydantic Model for Structured Metadata ---
class ChunkMetadata(BaseModel):
    content_summary: str = Field(
        description="Brief 1-2 sentence summary of the chunk content"
    )
    page_info: str = Field(
        description="Page number(s) for this chunk (e.g., 'Page: 5' or 'Pages: 5-6')"
    )


# --- ContextualVectorDB Class ---
class ContextualVectorDB:
    def __init__(self, name: str, gemini_model=None):
        if gemini_model is None:
            gemini_model = GEMINI_CONFIG["embedding_model"]
        self.client = genai.Client()
        self.model = gemini_model
        self.output_dim = GEMINI_CONFIG["embedding_dimension"]
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/contextual_vector_db.pkl"
        self.token_counts = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }
        self.token_lock = threading.Lock()

    # Gemini-2.5-flash-lite PDF document understanding
    def summarize_pdf(
        self, pdf_path: str, prompt: str = "Summarize this document"
    ) -> str:
        file_path = pathlib.Path(pdf_path)
        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type="application/pdf",
                ),
                prompt,
            ],
        )
        return response.text

    # Generate contextual metadata for individual chunks
    def generate_chunk_metadata(
        self, pdf_path: str, chunk_content: str, page_info: str
    ) -> str:
        file_path = pathlib.Path(pdf_path)
        prompt = f"""
        Analyze this document and the specific chunk below to provide structured metadata.

        Chunk content:
        {chunk_content}

        {page_info}

        Instructions:
        - Provide a concise summary focusing on the main topic/concept.
        """

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(
                    data=file_path.read_bytes(),
                    mime_type="application/pdf",
                ),
                prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ChunkMetadata,
            },
        )

        # Parse the structured response
        metadata = response.parsed

        # Convert to formatted string for embedding
        formatted_metadata = []
        formatted_metadata.append(f"Content Summary: {metadata.content_summary}")
        formatted_metadata.append(f"Page(s): {metadata.page_info}")

        return "\n".join(formatted_metadata)

    def load_data(self, dataset: list[dict[str, Any]], parallel_threads: int = 1):
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return
        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        def process_chunk(doc, chunk):
            # Add page number support
            pdf_path = doc.get("pdf_path")
            page_start = chunk.get("page_start")
            page_end = chunk.get("page_end")
            if page_start is not None and page_end is not None:
                if page_start == page_end:
                    page_info = f"Page: {page_start}"
                else:
                    page_info = (
                        f"Pages: {page_start}-{page_end} (chunk spans multiple pages)"
                    )
            elif page_start is not None:
                page_info = f"Page: {page_start}"
            else:
                page_info = "Page: Unknown"

            # Generate structured metadata
            if pdf_path:
                context = self.generate_chunk_metadata(
                    pdf_path, chunk["content"], page_info
                )
            else:
                context = f"{chunk['content']}\n{page_info}"
            return {
                "text_to_embed": f"{chunk['content']}\n\n{context}",
                "metadata": {
                    "doc_id": doc["doc_id"],
                    "original_uuid": doc["original_uuid"],
                    "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "original_content": chunk["content"],
                    "contextualized_content": context,
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_info": page_info,
                },
            }

        print(f"Processing {total_chunks} chunks with {parallel_threads} threads")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = []
            for doc in dataset:
                for chunk in doc["chunks"]:
                    futures.append(executor.submit(process_chunk, doc, chunk))
            for future in tqdm(
                as_completed(futures), total=total_chunks, desc="Processing chunks"
            ):
                result = future.result()
                texts_to_embed.append(result["text_to_embed"])
                metadata.append(result["metadata"])
        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()
        print(
            f"Contextual Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}"
        )

    def _embed_and_store(self, texts: list[str], data: list[dict[str, Any]]):
        batch_size = RAG_CONFIG["batch_size"]  # Use configurable batch size
        embeddings = []
        with tqdm(
            total=len(texts), desc="Embedding contextual chunks (Gemini)"
        ) as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=GEMINI_CONFIG["embedding_dimension"],
                        task_type=GEMINI_CONFIG["embedding_task_type"],
                    ),
                )
                for embedding_obj in result.embeddings:
                    values = np.array(embedding_obj.values)
                    normed = values / np.linalg.norm(values)
                    embeddings.append(normed)
                pbar.update(len(batch))
        self.embeddings = embeddings
        self.metadata = data

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            result = self.client.models.embed_content(
                model=self.model,
                contents=[query],
                config=types.EmbedContentConfig(
                    output_dimensionality=GEMINI_CONFIG["embedding_dimension"],
                    task_type=GEMINI_CONFIG["embedding_task_type"],
                ),
            )
            values = np.array(result.embeddings[0].values)
            query_embedding = values / np.linalg.norm(values)
            self.query_cache[query] = query_embedding
        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")
        similarities = np.dot(np.array(self.embeddings), query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)
        return top_results

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])


# --- ElasticsearchBM25 Class ---
class ElasticsearchBM25:
    def __init__(
        self,
        index_name: str = "contextual_bm25_index",
        es_config: dict[str, Any] | None = None,
    ):
        # Use configuration from config.py or default
        if es_config is None:
            es_config = ELASTICSEARCH_CONFIG.copy()

        try:
            # Prepare valid Elasticsearch client parameters
            es_params = {
                "hosts": es_config.get("hosts", ["http://localhost:9200"]),
                "request_timeout": es_config.get("timeout", 30),
                "retry_on_timeout": es_config.get("retry_on_timeout", True),
                "verify_certs": es_config.get("verify_certs", False),
            }

            # Add authentication
            if es_config.get("api_key"):
                es_params["api_key"] = es_config["api_key"]
            elif (
                "username" in es_config
                and "password" in es_config
                and es_config["password"]
            ):
                # Use basic auth with credentials from config
                es_params["basic_auth"] = (es_config["username"], es_config["password"])
            else:
                raise ValueError(
                    "Elasticsearch authentication not configured. Set ES_LOCAL_API_KEY or ES_LOCAL_PASSWORD in environment."
                )

            self.es_client = Elasticsearch(**es_params)

            # Test connection
            self.es_client.info()
            self.index_name = index_name
            self.available = True  # Set available before calling create_index
            self.create_index()
            print("✅ Successfully connected to Elasticsearch")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Elasticsearch not available: {e}")
            print("BM25 search will be disabled. Only semantic search will be used.")
            self.available = False
            self.es_client = None

    def create_index(self):
        if not self.available:
            return

        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False,
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "keyword", "index": False},
                    "chunk_id": {"type": "keyword", "index": False},
                    "original_index": {"type": "integer", "index": False},
                }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")

    def index_documents(self, documents: list[dict[str, Any]]):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "content": doc["original_content"],
                    "contextualized_content": doc["contextualized_content"],
                    "doc_id": doc["doc_id"],
                    "chunk_id": doc["chunk_id"],
                    "original_index": doc["original_index"],
                },
            }
            for doc in documents
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        self.es_client.indices.refresh(index=self.index_name)
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "contextualized_content"],
                }
            },
            "size": k,
        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "doc_id": hit["_source"]["doc_id"],
                "original_index": hit["_source"]["original_index"],
                "content": hit["_source"]["content"],
                "contextualized_content": hit["_source"]["contextualized_content"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]


# --- Utility Functions ---
def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


# --- Evaluation Functions ---
def evaluate_retrieval(
    queries: list[dict[str, Any]], retrieval_function: Callable, db, k: int = 20
) -> dict[str, float]:
    total_score = 0
    total_queries = len(queries)
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (
                    doc
                    for doc in query_item["golden_documents"]
                    if doc["uuid"] == doc_uuid
                ),
                None,
            )
            if not golden_doc:
                continue
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["index"] == chunk_index
                ),
                None,
            )
            if not golden_chunk:
                continue
            golden_contents.append(golden_chunk["content"].strip())
        if not golden_contents:
            continue
        retrieved_docs = retrieval_function(query, db, k=k)
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", doc["metadata"].get("content", ""))
                    .strip()
                )
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries,
    }


def retrieve_base(query: str, db, k: int = 20) -> list[dict[str, Any]]:
    return db.search(query, k=k)


def evaluate_db(db, original_jsonl_path: str, k):
    original_data = load_jsonl(original_jsonl_path)
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Total Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")
    return results


# --- Contextual BM25 Hybrid Search ---
def create_elasticsearch_bm25_index(
    db: ContextualVectorDB,
    index_name: str | None = None,
    es_config: dict[str, Any] | None = None,
):
    if index_name is None:
        index_name = "contextual_bm25_index"
    if es_config is None:
        es_config = ELASTICSEARCH_CONFIG
    es_bm25 = ElasticsearchBM25(index_name, es_config)
    es_bm25.index_documents(db.metadata)
    return es_bm25


def retrieve_advanced(
    query: str,
    db: ContextualVectorDB,
    es_bm25: ElasticsearchBM25,
    k: int,
    semantic_weight: float | None = None,
    bm25_weight: float | None = None,
):
    # Use config defaults if not provided
    if semantic_weight is None:
        semantic_weight = RAG_CONFIG["semantic_weight"]
    if bm25_weight is None:
        bm25_weight = RAG_CONFIG["bm25_weight"]

    num_chunks_to_recall = RAG_CONFIG["num_chunks_to_recall"]
    semantic_results = db.search(query, k=num_chunks_to_recall)
    ranked_chunk_ids = [
        (result["metadata"]["doc_id"], result["metadata"]["original_index"])
        for result in semantic_results
    ]
    bm25_results = es_bm25.search(query, k=num_chunks_to_recall)
    ranked_bm25_chunk_ids = [
        (result["doc_id"], result["original_index"]) for result in bm25_results
    ]
    chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
    chunk_id_to_score = {}
    for chunk_id in chunk_ids:
        score = 0
        if chunk_id in ranked_chunk_ids:
            index = ranked_chunk_ids.index(chunk_id)
            score += semantic_weight * (1 / (index + 1))
        if chunk_id in ranked_bm25_chunk_ids:
            index = ranked_bm25_chunk_ids.index(chunk_id)
            score += bm25_weight * (1 / (index + 1))
        chunk_id_to_score[chunk_id] = score
    sorted_chunk_ids = sorted(
        chunk_id_to_score.keys(),
        key=lambda x: (chunk_id_to_score[x], x[0], x[1]),
        reverse=True,
    )
    for index, chunk_id in enumerate(sorted_chunk_ids):
        chunk_id_to_score[chunk_id] = 1 / (index + 1)
    final_results = []
    semantic_count = 0
    bm25_count = 0
    for chunk_id in sorted_chunk_ids[:k]:
        chunk_metadata = next(
            chunk
            for chunk in db.metadata
            if chunk["doc_id"] == chunk_id[0] and chunk["original_index"] == chunk_id[1]
        )
        is_from_semantic = chunk_id in ranked_chunk_ids
        is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
        final_results.append(
            {
                "chunk": chunk_metadata,
                "score": chunk_id_to_score[chunk_id],
                "from_semantic": is_from_semantic,
                "from_bm25": is_from_bm25,
            }
        )
        if is_from_semantic and not is_from_bm25:
            semantic_count += 1
        elif is_from_bm25 and not is_from_semantic:
            bm25_count += 1
        else:
            semantic_count += 0.5
            bm25_count += 0.5
    return final_results, semantic_count, bm25_count


def evaluate_db_advanced(
    db: ContextualVectorDB,
    original_jsonl_path: str,
    k: int,
    es_config: dict[str, Any] | None = None,
):
    original_data = load_jsonl(original_jsonl_path)
    if es_config is None:
        es_config = ELASTICSEARCH_CONFIG
    es_bm25 = create_elasticsearch_bm25_index(db, es_config=es_config)
    try:
        warm_up_queries = original_data[: RAG_CONFIG["warmup_queries_count"]]
        for query_item in warm_up_queries:
            _ = retrieve_advanced(query_item["query"], db, es_bm25, k)
        total_score = 0
        total_semantic_count = 0
        total_bm25_count = 0
        total_results = 0
        for query_item in tqdm(original_data, desc="Evaluating retrieval"):
            query = query_item["query"]
            golden_chunk_uuids = query_item["golden_chunk_uuids"]
            golden_contents = []
            for doc_uuid, chunk_index in golden_chunk_uuids:
                golden_doc = next(
                    (
                        doc
                        for doc in query_item["golden_documents"]
                        if doc["uuid"] == doc_uuid
                    ),
                    None,
                )
                if golden_doc:
                    golden_chunk = next(
                        (
                            chunk
                            for chunk in golden_doc["chunks"]
                            if chunk["index"] == chunk_index
                        ),
                        None,
                    )
                    if golden_chunk:
                        golden_contents.append(golden_chunk["content"].strip())
            if not golden_contents:
                continue
            retrieved_docs, semantic_count, bm25_count = retrieve_advanced(
                query, db, es_bm25, k
            )
            chunks_found = 0
            for golden_content in golden_contents:
                for doc in retrieved_docs[:k]:
                    retrieved_content = doc["chunk"]["original_content"].strip()
                    if retrieved_content == golden_content:
                        chunks_found += 1
                        break
            query_score = chunks_found / len(golden_contents)
            total_score += query_score
            total_semantic_count += semantic_count
            total_bm25_count += bm25_count
            total_results += len(retrieved_docs)
        total_queries = len(original_data)
        average_score = total_score / total_queries
        pass_at_n = average_score * 100
        semantic_percentage = (
            (total_semantic_count / total_results) * 100 if total_results > 0 else 0
        )
        bm25_percentage = (
            (total_bm25_count / total_results) * 100 if total_results > 0 else 0
        )
        results = {
            "pass_at_n": pass_at_n,
            "average_score": average_score,
            "total_queries": total_queries,
        }
        print(f"Pass@{k}: {pass_at_n:.2f}%")
        print(f"Average Score: {average_score:.2f}")
        print(f"Total queries: {total_queries}")
        print(f"Percentage of results from semantic search: {semantic_percentage:.2f}%")
        print(f"Percentage of results from BM25: {bm25_percentage:.2f}%")
        return results, {"semantic": semantic_percentage, "bm25": bm25_percentage}
    finally:
        if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
            es_bm25.es_client.indices.delete(index=es_bm25.index_name)
            print(f"Deleted Elasticsearch index: {es_bm25.index_name}")


# --- Reranking Step ---
def chunk_to_content(chunk: dict[str, Any]) -> str:
    original_content = chunk["metadata"]["original_content"]
    contextualized_content = chunk["metadata"]["contextualized_content"]
    return f"{original_content}\n\nContext: {contextualized_content}"


# Efficient reranker using configurable model
def rerank_with_m3(
    query: str, candidate_chunks: list[dict[str, Any]], k: int
) -> list[dict[str, Any]]:
    pairs = [(query, chunk_to_content(chunk)) for chunk in candidate_chunks]
    tokenizer = AutoTokenizer.from_pretrained(RAG_CONFIG["reranker_model"])
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        RAG_CONFIG["reranker_model"], device_map=device_map
    )
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
    # Sort candidates by score (descending) and select top k
    top_indices = torch.topk(scores, k).indices.tolist()
    reranked = [
        {"chunk": candidate_chunks[i]["metadata"], "score": float(scores[i])}
        for i in top_indices
    ]
    return reranked


def retrieve_rerank(query: str, db, k: int) -> list[dict[str, Any]]:
    # Retrieve more results than needed, then rerank
    candidate_results = db.search(query, k=k * RAG_CONFIG["rerank_multiplier"])
    return rerank_with_m3(query, candidate_results, k)


def evaluate_retrieval_rerank(
    queries: list[dict[str, Any]], retrieval_function: Callable, db, k: int = 20
) -> dict[str, float]:
    total_score = 0
    total_queries = len(queries)
    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (
                    doc
                    for doc in query_item["golden_documents"]
                    if doc["uuid"] == doc_uuid
                ),
                None,
            )
            if golden_doc:
                golden_chunk = next(
                    (
                        chunk
                        for chunk in golden_doc["chunks"]
                        if chunk["index"] == chunk_index
                    ),
                    None,
                )
                if golden_chunk:
                    golden_contents.append(golden_chunk["content"].strip())
        if not golden_contents:
            continue
        retrieved_docs = retrieval_function(query, db, k)
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = doc["chunk"]["original_content"].strip()
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break
        query_score = chunks_found / len(golden_contents)
        total_score += query_score
    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {
        "pass_at_n": pass_at_n,
        "average_score": average_score,
        "total_queries": total_queries,
    }


def evaluate_db_rerank(db, original_jsonl_path, k):
    original_data = load_jsonl(original_jsonl_path)

    def retrieval_function(query, db, k):
        return retrieve_rerank(query, db, k)

    results = evaluate_retrieval_rerank(original_data, retrieval_function, db, k)
    print(f"Pass@{k}: {results['pass_at_n']:.2f}%")
    print(f"Average Score: {results['average_score']}")
    print(f"Total queries: {results['total_queries']}")
    return results


# --- Utility Functions for Multi-Document RAG ---
def create_multi_document_dataset(
    pdf_paths: list[str], chunker=None
) -> list[dict[str, Any]]:
    """Create dataset from multiple PDF files."""
    if chunker is None:
        from pdf_chunker import PDFClusterSemanticChunker

        chunker = PDFClusterSemanticChunker()

    dataset = []
    for pdf_path in pdf_paths:
        try:
            doc_entry = chunker.create_dataset_from_pdf(pdf_path)
            dataset.append(doc_entry)
            print(f"Processed {pdf_path}: {len(doc_entry['chunks'])} chunks")
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {pdf_path}: {e}")

    return dataset


def create_rag_system(
    dataset: list[dict[str, Any]],
    db_name: str = "multi_doc_rag",
    es_config: dict[str, Any] | None = None,
) -> tuple:
    """Create a complete RAG system with vector DB and BM25."""
    # Create contextual vector database
    contextual_db = ContextualVectorDB(db_name)
    contextual_db.load_data(dataset, parallel_threads=8)

    # Create BM25 index
    if es_config is None:
        es_config = ELASTICSEARCH_CONFIG
    es_bm25 = ElasticsearchBM25(f"{db_name}_bm25", es_config)
    es_bm25.index_documents(contextual_db.metadata)

    return contextual_db, es_bm25


def answer_question(
    query: str,
    contextual_db,
    es_bm25,
    gemini_client=None,
    k: int = 5,
    use_reranking: bool = True,
):
    """Answer a question using the RAG system."""
    if gemini_client is None:
        gemini_client = genai.Client()

    # Get search results
    if use_reranking:
        # Get more results for reranking
        candidate_results = contextual_db.search(query, k=k * 3)
        search_results = rerank_with_m3(query, candidate_results, k)
    else:
        # Use hybrid search
        results, _, _ = retrieve_advanced(query, contextual_db, es_bm25, k)
        search_results = results

    # Prepare context
    context_parts = []
    citations = []

    for i, result in enumerate(search_results):
        chunk_data = result.get("chunk", result.get("metadata", {}))
        content = chunk_data.get("original_content", "")
        doc_id = chunk_data.get("doc_id", "Unknown")
        page_start = chunk_data.get("page_start")
        page_end = chunk_data.get("page_end")

        # Format citation
        if page_start and page_end:
            if page_start == page_end:
                page_info = f"p. {page_start}"
            else:
                page_info = f"pp. {page_start}-{page_end}"
        else:
            page_info = "page unknown"

        citation = f"[{i + 1}] {doc_id}, {page_info}"
        citations.append(citation)
        context_parts.append(f"[Source {i + 1}]: {content}")

    # Generate response
    context_text = "\n\n".join(context_parts)
    prompt = f"""Based on the following context from documents, answer the question accurately and helpfully.

Context:
{context_text}

Question: {query}

Instructions:
1. Answer based primarily on the provided context
2. Include relevant citations using [1], [2], etc. format
3. If the context doesn't contain enough information, say so clearly
4. Be precise and helpful

Answer:"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp", contents=[prompt]
        )

        answer = response.text

        # Add citations
        if citations:
            answer += "\n\n**Sources:**\n" + "\n".join(citations)

        return answer, search_results

    except Exception as e:  # noqa: BLE001
        return f"Error generating response: {e!s}", search_results


# --- Main Pipeline ---
def main():
    # Example usage
    pdf_paths = ["data/document1.pdf", "data/document2.pdf"]

    # Create dataset from PDFs
    from pdf_chunker import PDFClusterSemanticChunker

    chunker = PDFClusterSemanticChunker()
    dataset = create_multi_document_dataset(pdf_paths, chunker)

    # Create RAG system
    contextual_db, es_bm25 = create_rag_system(dataset)

    # Example question
    query = "What are the main topics discussed in these documents?"
    answer, _results = answer_question(query, contextual_db, es_bm25)
    print(f"Question: {query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
