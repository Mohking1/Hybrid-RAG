# Hybrid RAG & MCP Server

A modular RAG engine built to solve context dilution and score calibration issues in standard vector search.

## Why this exists

1. **Context vs Precision tradeoff**: Small chunks embed well for search but lack context for generation; large chunks dilute vector embeddings. This repo uses **Parent-Child Chunking** (matches on 200-token leaf chunks, resolves 1000-token parent sections for LLM context).
2. **Score Calibration**: Combining BM25 and dense cosine scores with linear weights ($\alpha \cdot \text{dense} + \beta \cdot \text{sparse}$) is fragile across different document distributions. This repo uses **Reciprocal Rank Fusion (RRF)**.
3. **Query Expansion Overhead**: Avoids wasting LLM calls on simple queries by using a **Mathematical Adaptive Router** (Mean IDF + Fast-Probe Score Entropy) to decide between direct search, multi-query expansion, and HyDE.

---

## Core Components

* `parent_child_chunker.py` — Splits text into linked 1000-token parent blocks and 200-token child chunks.
* `es_unified_store.py` — Elasticsearch 8 store handling HNSW dense vectors, BM25 text, and RRF rank merging ($k=60$).
* `adaptive_router.py` — Evaluates query term IDF and probe distribution entropy to pick the retrieval tier.
* `rag_engine.py` — Orchestrates ingestion, retrieval, parent context resolution, `bge-reranker-large` reranking, and generation.
* `mcp_server.py` — Standard MCP stdio server for LLMs and agentic IDEs.

---

## Quickstart

```bash
# 1. Clone & install
git clone <repo-url>
cd RAG
pip install -r requirements.txt

# 2. Start Elasticsearch & Streamlit UI
docker compose up -d

# 3. Or run standalone Python interface
python3 -c "
from rag_engine import RAGEngine
engine = RAGEngine()
engine.ingest_pdf('sample.pdf')
print(engine.answer_question('What are the findings?')['answer'])
"
```

---

## MCP Server Configuration

Add to your MCP settings (`claude_desktop_config.json` or `mcp_config.json`):

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python3",
      "args": ["/path/to/RAG/mcp_server.py"],
      "env": {
        "GEMINI_API_KEY": "your_key",
        "ELASTICSEARCH_HOST": "http://localhost:9200"
      }
    }
  }
}
```