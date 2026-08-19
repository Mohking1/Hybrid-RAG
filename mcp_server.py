import json
import os

from mcp.server.mcpserver import MCPServer

from rag_engine import RAGEngine

# Initialize MCP Server
app = MCPServer(
    name="rag-server",
    version="1.0.0",
    description="Model Context Protocol (MCP) server for Hybrid RAG with Adaptive Routing and RRF.",
)

_engines: dict[str, RAGEngine] = {}


def get_engine(collection_name: str = "default") -> RAGEngine:
    if collection_name not in _engines:
        _engines[collection_name] = RAGEngine(collection_name=collection_name)
    return _engines[collection_name]


@app.tool(
    name="rag_ingest_pdf",
    description="Ingest a PDF file into the hierarchical parent-child Elasticsearch knowledge base.",
)
def rag_ingest_pdf(
    pdf_path: str, collection_name: str = "default", doc_id: str | None = None
) -> str:
    """
    Ingests a PDF document.

    Args:
        pdf_path: Absolute or relative path to the PDF document.
        collection_name: Target document collection name (default: "default").
        doc_id: Optional custom document identifier.
    """
    if not os.path.exists(pdf_path):
        return json.dumps({"status": "error", "message": f"File not found: {pdf_path}"})

    try:
        engine = get_engine(collection_name)
        result = engine.ingest_pdf(pdf_path=pdf_path, doc_id=doc_id)
        return json.dumps(result, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "message": str(e)})


@app.tool(
    name="rag_ask",
    description="Ask a question and receive a citation-grounded answer using Adaptive Router, RRF Hybrid Search, Parent Resolution, and Cross-Encoder Reranking.",
)
def rag_ask(
    question: str,
    collection_name: str = "default",
    top_k: int = 5,
    use_reranking: bool = True,
) -> str:
    """
    Answers a question based on indexed knowledge.

    Args:
        question: The user query or question to answer.
        collection_name: Target document collection name (default: "default").
        top_k: Number of parent context chunks to retrieve (default: 5).
        use_reranking: Whether to apply cross-encoder neural reranking (default: True).
    """
    try:
        engine = get_engine(collection_name)
        result = engine.answer_question(
            query=question, top_k=top_k, use_reranking=use_reranking
        )
        output = {
            "answer": result["answer"],
            "router_tier": result["route_decision"].tier,
            "router_reason": result["route_decision"].reason,
            "sources": result["sources"],
        }
        return json.dumps(output, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "message": str(e)})


@app.tool(
    name="rag_search",
    description="Retrieve relevant parent context chunks and ranking diagnostics without generating an answer.",
)
def rag_search(
    query: str,
    collection_name: str = "default",
    top_k: int = 5,
    use_reranking: bool = True,
) -> str:
    """
    Searches indexed documents and returns relevant parent passages with scores.

    Args:
        query: Search query.
        collection_name: Document collection name (default: "default").
        top_k: Number of results to return (default: 5).
        use_reranking: Whether to apply cross-encoder reranking (default: True).
    """
    try:
        engine = get_engine(collection_name)
        result = engine.search(query=query, top_k=top_k, use_reranking=use_reranking)

        parents_data = []
        for p, score in result["top_parents"]:
            parents_data.append(
                {
                    "parent_id": p.parent_id,
                    "doc_id": p.doc_id,
                    "filename": p.filename,
                    "score": score,
                    "page_start": p.page_start,
                    "page_end": p.page_end,
                    "content": p.content,
                }
            )

        output = {
            "query": query,
            "router_tier": result["route_decision"].tier,
            "router_reason": result["route_decision"].reason,
            "results_count": len(parents_data),
            "results": parents_data,
        }
        return json.dumps(output, indent=2)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "message": str(e)})


@app.tool(
    name="rag_route_query",
    description="Analyze query complexity and return mathematical routing metrics (Mean IDF Specificity, Fast-Probe Score Entropy & Margin).",
)
def rag_route_query(query: str, collection_name: str = "default") -> str:
    """
    Calculates mathematical complexity metrics and routing tier for a query.

    Args:
        query: User query to inspect.
        collection_name: Document collection name (default: "default").
    """
    try:
        engine = get_engine(collection_name)
        decision = engine.router.route(
            query=query, es_store=engine.es_store, embed_fn=engine.embed_text
        )
        return json.dumps(
            {
                "query": query,
                "tier": decision.tier,
                "specificity_score": decision.specificity_score,
                "score_margin": decision.score_margin,
                "entropy": decision.entropy,
                "reason": decision.reason,
                "expanded_queries": decision.expanded_queries,
                "hypothetical_doc": decision.hypothetical_doc,
            },
            indent=2,
        )
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run()
