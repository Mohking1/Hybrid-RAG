from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from adaptive_router import AdaptiveQueryRouter, QueryRouteDecision
from config import ELASTICSEARCH_CONFIG, GEMINI_CONFIG, RAG_CONFIG
from es_unified_store import UnifiedElasticsearchStore, reciprocal_rank_fusion
from parent_child_chunker import ParentChildChunker, ParentChunk


class RAGEngine:
    """
    RAG Engine with hierarchical parent-child chunking, unified Elasticsearch
    hybrid retrieval (dense vector + BM25), adaptive query routing, and cross-encoder reranking.
    """

    def __init__(
        self,
        collection_name: str = "default",
        es_config: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        reranker_model: str | None = None,
    ):
        self.collection_name = collection_name
        self.es_config = es_config or ELASTICSEARCH_CONFIG
        self.embedding_model = embedding_model or GEMINI_CONFIG.get(
            "embedding_model", "text-embedding-004"
        )
        self.reranker_model_name = reranker_model or RAG_CONFIG.get(
            "reranker_model", "BAAI/bge-reranker-large"
        )
        self.llm_model = GEMINI_CONFIG.get("generation_model", "gemini-2.5-flash")

        self.chunker = ParentChildChunker(
            parent_chunk_size=RAG_CONFIG.get("parent_chunk_size", 1000),
            parent_chunk_overlap=RAG_CONFIG.get("parent_chunk_overlap", 150),
            child_chunk_size=RAG_CONFIG.get("child_chunk_size", 200),
            child_chunk_overlap=RAG_CONFIG.get("child_chunk_overlap", 30),
        )
        self.es_store = UnifiedElasticsearchStore(
            collection_name=self.collection_name, es_config=self.es_config
        )
        self._gemini_client = None
        self._router = None
        self._reranker_model = None
        self._reranker_tokenizer = None

    @property
    def client(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client()
        return self._gemini_client

    @property
    def router(self) -> AdaptiveQueryRouter:
        if self._router is None:
            self._router = AdaptiveQueryRouter(gemini_client=self.client)
        return self._router

    def initialize_storage(self, recreate: bool = False):
        self.es_store.create_indices(recreate=recreate)

    def embed_text(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=[text],
            config=types.EmbedContentConfig(
                output_dimensionality=GEMINI_CONFIG.get("embedding_dimension", 768),
                task_type="RETRIEVAL_QUERY",
            ),
        )
        values = np.array(result.embeddings[0].values)
        norm = np.linalg.norm(values)
        if norm > 0:
            values = values / norm
        return values.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    output_dimensionality=GEMINI_CONFIG.get("embedding_dimension", 768),
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            for emb_obj in result.embeddings:
                values = np.array(emb_obj.values)
                norm = np.linalg.norm(values)
                if norm > 0:
                    values = values / norm
                all_embeddings.append(values.tolist())
        return all_embeddings

    def ingest_pdf(self, pdf_path: str, doc_id: str | None = None) -> dict[str, Any]:
        self.initialize_storage(recreate=False)
        parents, children = self.chunker.chunk_pdf(pdf_path, doc_id=doc_id)

        if not parents:
            return {"status": "empty", "parents": 0, "children": 0}

        parent_map = {p.parent_id: p for p in parents}

        child_texts_to_embed = []
        for child in children:
            parent = parent_map.get(child.parent_id)
            if parent:
                child.contextualized_content = f"Document: {child.filename} (Page {child.page_number})\nSection: {parent.content[:200]}...\n\n{child.content}"
            else:
                child.contextualized_content = child.content
            child_texts_to_embed.append(child.contextualized_content)

        embeddings = self.embed_batch(child_texts_to_embed)
        for child, emb in zip(children, embeddings):
            child.embedding = emb

        self.es_store.index_parent_chunks(parents)
        self.es_store.index_child_chunks(children)

        return {
            "status": "success",
            "doc_id": parents[0].doc_id,
            "filename": parents[0].filename,
            "parent_chunks_indexed": len(parents),
            "child_chunks_indexed": len(children),
        }

    def _get_reranker(self):
        if self._reranker_model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name
            )
            self._reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name, device_map=device
            )
            self._reranker_model.eval()
        return self._reranker_model, self._reranker_tokenizer

    def rerank_parents(
        self, query: str, parent_chunks: list[ParentChunk], top_k: int = 5
    ) -> list[tuple[ParentChunk, float]]:
        if not parent_chunks:
            return []
        try:
            import torch

            model, tokenizer = self._get_reranker()
            pairs = [(query, p.content) for p in parent_chunks]
            with torch.no_grad():
                inputs = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                scores = model(**inputs, return_dict=True).logits.view(-1).float()

            scored = list(zip(parent_chunks, scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        except Exception:  # noqa: BLE001
            return [
                (p, 1.0 - (idx * 0.05)) for idx, p in enumerate(parent_chunks[:top_k])
            ]

    def search(
        self, query: str, top_k: int = 5, use_reranking: bool = True
    ) -> dict[str, Any]:
        decision = self.router.route(
            query=query, es_store=self.es_store, embed_fn=self.embed_text
        )

        child_candidates: list[dict[str, Any]] = []

        if decision.tier == "direct":
            query_vector = self.embed_text(query)
            child_candidates = self.es_store.search_hybrid_rrf(
                query, query_vector, top_k=top_k * 4
            )

        elif decision.tier == "hyde" and decision.hypothetical_doc:
            hyde_vector = self.embed_text(decision.hypothetical_doc)
            child_candidates = self.es_store.search_hybrid_rrf(
                query, hyde_vector, top_k=top_k * 4
            )

        elif decision.tier == "multi_query" and decision.expanded_queries:
            sub_results_list = []
            all_queries = [query] + decision.expanded_queries
            for q in all_queries:
                q_vec = self.embed_text(q)
                hits = self.es_store.search_hybrid_rrf(q, q_vec, top_k=top_k * 2)
                sub_results_list.append(hits)

            if len(sub_results_list) >= 2:
                merged = reciprocal_rank_fusion(
                    sub_results_list[0], sub_results_list[1], k_rrf=60
                )
                for next_sub in sub_results_list[2:]:
                    merged = reciprocal_rank_fusion(merged, next_sub, k_rrf=60)
                child_candidates = merged[: top_k * 4]
            else:
                child_candidates = sub_results_list[0] if sub_results_list else []

        parent_ids = [c["parent_id"] for c in child_candidates if "parent_id" in c]
        parent_dict = self.es_store.get_parents_by_ids(parent_ids)

        seen_parents = set()
        resolved_parents: list[ParentChunk] = []
        for pid in parent_ids:
            if pid in parent_dict and pid not in seen_parents:
                resolved_parents.append(parent_dict[pid])
                seen_parents.add(pid)

        if use_reranking and resolved_parents:
            reranked_parents = self.rerank_parents(query, resolved_parents, top_k=top_k)
        else:
            reranked_parents = [(p, 1.0) for p in resolved_parents[:top_k]]

        return {
            "query": query,
            "route_decision": decision,
            "top_parents": reranked_parents,
            "child_candidates": child_candidates[: top_k * 2],
        }

    def answer_question(
        self, query: str, top_k: int = 5, use_reranking: bool = True
    ) -> dict[str, Any]:
        search_result = self.search(
            query=query, top_k=top_k, use_reranking=use_reranking
        )
        top_parents: list[tuple[ParentChunk, float]] = search_result["top_parents"]
        decision: QueryRouteDecision = search_result["route_decision"]

        if not top_parents:
            return {
                "answer": "No relevant documents found in the database to answer this question.",
                "route_decision": decision,
                "sources": [],
            }

        context_blocks = []
        sources = []
        for idx, (p, score) in enumerate(top_parents, start=1):
            source_info = {
                "index": idx,
                "filename": p.filename,
                "page_start": p.page_start,
                "page_end": p.page_end,
                "score": score,
                "content_preview": p.content[:200],
            }
            sources.append(source_info)
            context_blocks.append(
                f"[Source {idx}] (Document: {p.filename}, Page {p.page_start})\n{p.content}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        prompt = (
            f"You are a helpful research assistant. Answer the user's question accurately based ONLY on the provided context.\n"
            f"For every key factual claim, cite the relevant source using [Source X] notation.\n"
            f"If the context does not contain enough information to answer, state clearly what is missing.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        response = self.client.models.generate_content(
            model=self.llm_model, contents=prompt
        )

        return {
            "answer": response.text.strip(),
            "route_decision": decision,
            "sources": sources,
            "context_used": context_str,
        }
