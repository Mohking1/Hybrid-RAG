from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from config import ELASTICSEARCH_CONFIG, GEMINI_CONFIG
from parent_child_chunker import ChildChunk, ParentChunk


def reciprocal_rank_fusion(
    dense_rankings: list[dict[str, Any]],
    sparse_rankings: list[dict[str, Any]],
    k_rrf: int = 60,
    w_dense: float = 1.0,
    w_sparse: float = 1.0,
) -> list[dict[str, Any]]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Score(d) = w_dense / (k_rrf + rank_dense(d)) + w_sparse / (k_rrf + rank_sparse(d))
    """
    rrf_scores: dict[str, float] = {}
    item_payloads: dict[str, dict[str, Any]] = {}

    # Process dense rankings (1-based rank)
    for rank, item in enumerate(dense_rankings, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_dense / (k_rrf + rank))
        item_payloads[cid] = item

    # Process sparse rankings (1-based rank)
    for rank, item in enumerate(sparse_rankings, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_sparse / (k_rrf + rank))
        if cid not in item_payloads:
            item_payloads[cid] = item

    # Sort items by descending RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

    results = []
    for cid, score in sorted_items:
        entry = dict(item_payloads[cid])
        entry["rrf_score"] = score
        results.append(entry)

    return results


class UnifiedElasticsearchStore:
    """
    Unified Elasticsearch 8 store managing parent documents and child chunks
    with dense vector (HNSW cosine similarity) and BM25 text fields.
    """

    def __init__(
        self, collection_name: str = "default", es_config: dict[str, Any] | None = None
    ):
        cfg = es_config or ELASTICSEARCH_CONFIG
        if cfg.get("hosts"):
            hosts = cfg["hosts"]
        elif "host" in cfg:
            scheme = cfg.get("scheme", "http")
            port = cfg.get("port", 9200)
            hosts = [f"{scheme}://{cfg['host']}:{port}"]
        else:
            hosts = ["http://localhost:9200"]

        api_key = cfg.get("api_key")
        auth = (
            (cfg["username"], cfg["password"])
            if (cfg.get("username") and cfg.get("password"))
            else None
        )

        self.client = Elasticsearch(
            hosts=hosts,
            api_key=api_key if api_key else None,
            basic_auth=auth if (auth and not api_key) else None,
            verify_certs=cfg.get("verify_certs", False),
            ssl_show_warn=False,
        )
        self.parents_index = f"rag_parents_{collection_name}".lower()
        self.children_index = f"rag_children_{collection_name}".lower()
        self.embedding_dim = GEMINI_CONFIG.get("embedding_dimension", 768)

    def create_indices(self, recreate: bool = False):
        if recreate:
            if self.client.indices.exists(index=self.parents_index):
                self.client.indices.delete(index=self.parents_index)
            if self.client.indices.exists(index=self.children_index):
                self.client.indices.delete(index=self.children_index)

        # Parent chunks index mapping
        if not self.client.indices.exists(index=self.parents_index):
            parent_mapping = {
                "mappings": {
                    "properties": {
                        "parent_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "content": {"type": "text"},
                        "page_start": {"type": "integer"},
                        "page_end": {"type": "integer"},
                        "metadata": {"type": "object", "enabled": True},
                    }
                }
            }
            self.client.indices.create(index=self.parents_index, body=parent_mapping)

        # Child chunks index mapping (Dense vector + BM25)
        if not self.client.indices.exists(index=self.children_index):
            child_mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "filename": {"type": "keyword"},
                        "content": {"type": "text"},
                        "contextualized_content": {
                            "type": "text",
                            "analyzer": "standard",
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": self.embedding_dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "page_number": {"type": "integer"},
                    }
                }
            }
            self.client.indices.create(index=self.children_index, body=child_mapping)

    def index_parent_chunks(self, parents: list[ParentChunk]):
        if not parents:
            return
        actions = []
        for p in parents:
            actions.append(
                {
                    "_index": self.parents_index,
                    "_id": p.parent_id,
                    "_source": {
                        "parent_id": p.parent_id,
                        "doc_id": p.doc_id,
                        "filename": p.filename,
                        "content": p.content,
                        "page_start": p.page_start,
                        "page_end": p.page_end,
                        "metadata": p.metadata,
                    },
                }
            )
        bulk(self.client, actions)
        self.client.indices.refresh(index=self.parents_index)

    def index_child_chunks(self, children: list[ChildChunk]):
        if not children:
            return
        actions = []
        for c in children:
            actions.append(
                {
                    "_index": self.children_index,
                    "_id": c.chunk_id,
                    "_source": {
                        "chunk_id": c.chunk_id,
                        "parent_id": c.parent_id,
                        "doc_id": c.doc_id,
                        "filename": c.filename,
                        "content": c.content,
                        "contextualized_content": c.contextualized_content or c.content,
                        "embedding": c.embedding,
                        "page_number": c.page_number,
                    },
                }
            )
        bulk(self.client, actions)
        self.client.indices.refresh(index=self.children_index)

    def search_sparse_bm25(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["contextualized_content^2", "content"],
                }
            },
        }
        res = self.client.search(index=self.children_index, body=body)
        results = []
        for hit in res["hits"]["hits"]:
            src = dict(hit["_source"])
            src["score"] = hit["_score"]
            results.append(src)
        return results

    def search_dense_knn(
        self, query_vector: list[float], top_k: int = 30
    ) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(100, top_k * 2),
            },
        }
        res = self.client.search(index=self.children_index, body=body)
        results = []
        for hit in res["hits"]["hits"]:
            src = dict(hit["_source"])
            src["score"] = hit["_score"]
            results.append(src)
        return results

    def search_hybrid_rrf(
        self, query: str, query_vector: list[float], top_k: int = 20
    ) -> list[dict[str, Any]]:
        sparse_hits = self.search_sparse_bm25(query, top_k=top_k * 2)
        dense_hits = self.search_dense_knn(query_vector, top_k=top_k * 2)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k_rrf=60)
        return fused[:top_k]

    def get_parents_by_ids(self, parent_ids: list[str]) -> dict[str, ParentChunk]:
        if not parent_ids:
            return {}
        unique_ids = list(set(parent_ids))
        res = self.client.mget(index=self.parents_index, body={"ids": unique_ids})
        parents_dict = {}
        for doc in res["docs"]:
            if doc.get("found"):
                src = doc["_source"]
                p = ParentChunk(
                    parent_id=src["parent_id"],
                    doc_id=src["doc_id"],
                    filename=src["filename"],
                    content=src["content"],
                    page_start=src.get("page_start", 1),
                    page_end=src.get("page_end", 1),
                    metadata=src.get("metadata", {}),
                )
                parents_dict[p.parent_id] = p
        return parents_dict
