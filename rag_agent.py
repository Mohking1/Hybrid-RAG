"""
RAG Agent Interface - Function-callable RAG system for agentic AI

This module provides a clean, function-callable interface to the RAG system,
allowing agentic AI systems to easily insert PDFs and search for answers.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
from datetime import datetime

# Import our RAG components
from pdf_chunker import PDFClusterSemanticChunker
from rag_pipeline import (
    ContextualVectorDB,
    ElasticsearchBM25,
    retrieve_advanced,
    rerank_with_m3,
    answer_question
)
from config import ELASTICSEARCH_CONFIG, GEMINI_CONFIG, RAG_CONFIG


class RAGAgent:
    """
    A function-callable RAG system for agentic AI.

    This class provides methods to:
    - Insert PDFs into the knowledge base
    - Search for answers from the processed documents
    - Manage multiple document collections
    """

    def __init__(self, collection_name: str = "default", config: Dict[str, Any] = None):
        """
        Initialize the RAG Agent.

        Args:
            collection_name: Name of the document collection
            config: Optional configuration override
        """
        self.collection_name = collection_name
        self.config = config or {}

        # Initialize components
        self.chunker = PDFClusterSemanticChunker(
            max_chunk_size=self.config.get('max_chunk_size', 400),
            min_chunk_size=self.config.get('min_chunk_size', 50)
        )

        self.vector_db = None
        self.es_bm25 = None
        self.processed_documents = []
        self.document_metadata = {}

        # Data paths
        self.data_dir = f"./data/{collection_name}"
        self.metadata_file = f"{self.data_dir}/document_metadata.json"

        # Load existing data if available
        self._load_existing_data()

    def _load_existing_data(self):
        """Load existing vector database and metadata."""
        try:
            # Load vector database
            vector_db_path = f"{self.data_dir}/contextual_vector_db.pkl"
            if os.path.exists(vector_db_path):
                self.vector_db = ContextualVectorDB(self.collection_name)
                self.vector_db.load_db()
                print(f"✅ Loaded existing vector database for collection '{self.collection_name}'")

            # Load Elasticsearch index
            self.es_bm25 = ElasticsearchBM25(f"{self.collection_name}_bm25")
            if self.es_bm25.available:
                print(f"✅ Loaded existing Elasticsearch index for collection '{self.collection_name}'")

            # Load document metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.document_metadata = json.load(f)
                print(f"✅ Loaded metadata for {len(self.document_metadata)} documents")

        except Exception as e:
            print(f"⚠️  Error loading existing data: {e}")

    def _save_metadata(self):
        """Save document metadata to disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(self.document_metadata, f, indent=2, default=str)

    def insert_pdf(self, pdf_path: Union[str, Path], doc_id: Optional[str] = None,
                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Insert a PDF document into the RAG system.

        Args:
            pdf_path: Path to the PDF file
            doc_id: Optional custom document ID (defaults to filename)
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if doc_id is None:
            doc_id = pdf_path.stem

        print(f"📄 Processing PDF: {pdf_path.name}")

        try:
            # Process the PDF
            dataset_entry = self.chunker.create_dataset_from_pdf(str(pdf_path), doc_id=doc_id)

            # Add metadata
            dataset_entry['filename'] = pdf_path.name
            dataset_entry['pdf_path'] = str(pdf_path)
            dataset_entry['inserted_at'] = datetime.now().isoformat()
            dataset_entry['custom_metadata'] = metadata or {}

            # Check if document already exists
            if doc_id in self.document_metadata:
                print(f"⚠️  Document '{doc_id}' already exists. Updating...")
                # Remove old document from processed_documents
                self.processed_documents = [
                    doc for doc in self.processed_documents
                    if doc['doc_id'] != doc_id
                ]

            # Add to processed documents
            self.processed_documents.append(dataset_entry)

            # Update metadata
            self.document_metadata[doc_id] = {
                'filename': pdf_path.name,
                'pdf_path': str(pdf_path),
                'chunks': len(dataset_entry['chunks']),
                'inserted_at': dataset_entry['inserted_at'],
                'custom_metadata': metadata or {}
            }

            # Rebuild vector database and search index
            self._rebuild_indexes()

            # Save metadata
            self._save_metadata()

            result = {
                'status': 'success',
                'doc_id': doc_id,
                'chunks_processed': len(dataset_entry['chunks']),
                'total_documents': len(self.processed_documents)
            }

            print(f"✅ Successfully inserted '{doc_id}' with {len(dataset_entry['chunks'])} chunks")
            return result

        except Exception as e:
            error_result = {
                'status': 'error',
                'doc_id': doc_id,
                'error': str(e)
            }
            print(f"❌ Error processing '{doc_id}': {e}")
            return error_result

    def insert_pdfs(self, pdf_paths: List[Union[str, Path]],
                    metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Insert multiple PDF documents into the RAG system.

        Args:
            pdf_paths: List of paths to PDF files
            metadata_list: Optional list of metadata dictionaries (same length as pdf_paths)

        Returns:
            List of processing results
        """
        if metadata_list and len(metadata_list) != len(pdf_paths):
            raise ValueError("metadata_list must have the same length as pdf_paths")

        results = []
        for i, pdf_path in enumerate(pdf_paths):
            metadata = metadata_list[i] if metadata_list else None
            result = self.insert_pdf(pdf_path, metadata=metadata)
            results.append(result)

        return results

    def _rebuild_indexes(self):
        """Rebuild vector database and search indexes with current documents."""
        if not self.processed_documents:
            return

        print("🔄 Rebuilding indexes...")

        # Create/update vector database
        self.vector_db = ContextualVectorDB(self.collection_name)
        self.vector_db.load_data(self.processed_documents, parallel_threads=8)

        # Create/update BM25 index
        self.es_bm25 = ElasticsearchBM25(f"{self.collection_name}_bm25")
        self.es_bm25.index_documents(self.vector_db.metadata)

        print("✅ Indexes rebuilt successfully")

    def search(self, query: str, k: int = 5, use_reranking: bool = True,
               include_context: bool = True) -> Dict[str, Any]:
        """
        Search for answers in the processed documents.

        Args:
            query: The search query
            k: Number of results to return
            use_reranking: Whether to use neural reranking
            include_context: Whether to include full context in results

        Returns:
            Dictionary with search results and answer
        """
        if not self.vector_db or not self.es_bm25:
            raise ValueError("No documents have been inserted yet. Use insert_pdf() first.")

        try:
            # Get search results
            if use_reranking:
                # Get more results for reranking
                candidate_results = self.vector_db.search(query, k=k*3)
                search_results = rerank_with_m3(query, candidate_results, k)
            else:
                # Use hybrid search
                results, _, _ = retrieve_advanced(query, self.vector_db, self.es_bm25, k)
                search_results = results

            # Generate answer
            answer, _ = answer_question(
                query=query,
                contextual_db=self.vector_db,
                es_bm25=self.es_bm25,
                k=k,
                use_reranking=use_reranking
            )

            # Format results
            formatted_results = []
            for i, result in enumerate(search_results):
                chunk_data = result.get('chunk', result.get('metadata', {}))
                formatted_result = {
                    'rank': i + 1,
                    'doc_id': chunk_data.get('doc_id', 'Unknown'),
                    'score': result.get('score', 0),
                    'page_start': chunk_data.get('page_start'),
                    'page_end': chunk_data.get('page_end'),
                    'content': chunk_data.get('original_content', '') if include_context else None
                }
                formatted_results.append(formatted_result)

            return {
                'status': 'success',
                'query': query,
                'answer': answer,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }

        except Exception as e:
            return {
                'status': 'error',
                'query': query,
                'error': str(e)
            }

    def get_document_info(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about processed documents.

        Args:
            doc_id: Specific document ID to get info for, or None for all documents

        Returns:
            Dictionary with document information
        """
        if doc_id:
            if doc_id in self.document_metadata:
                return {
                    'status': 'success',
                    'document': self.document_metadata[doc_id]
                }
            else:
                return {
                    'status': 'error',
                    'error': f"Document '{doc_id}' not found"
                }
        else:
            return {
                'status': 'success',
                'total_documents': len(self.document_metadata),
                'documents': self.document_metadata
            }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document from the RAG system.

        Args:
            doc_id: Document ID to delete

        Returns:
            Dictionary with deletion result
        """
        if doc_id not in self.document_metadata:
            return {
                'status': 'error',
                'error': f"Document '{doc_id}' not found"
            }

        try:
            # Remove from processed documents
            self.processed_documents = [
                doc for doc in self.processed_documents
                if doc['doc_id'] != doc_id
            ]

            # Remove from metadata
            del self.document_metadata[doc_id]

            # Rebuild indexes
            if self.processed_documents:
                self._rebuild_indexes()
            else:
                # No documents left, clean up
                self.vector_db = None
                self.es_bm25 = None

            # Save metadata
            self._save_metadata()

            return {
                'status': 'success',
                'doc_id': doc_id,
                'message': f"Document '{doc_id}' deleted successfully"
            }

        except Exception as e:
            return {
                'status': 'error',
                'doc_id': doc_id,
                'error': str(e)
            }

    def clear_collection(self) -> Dict[str, Any]:
        """
        Clear all documents from the collection.

        Returns:
            Dictionary with clear result
        """
        try:
            # Delete data directory
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)

            # Reset instance variables
            self.vector_db = None
            self.es_bm25 = None
            self.processed_documents = []
            self.document_metadata = {}

            return {
                'status': 'success',
                'message': f"Collection '{self.collection_name}' cleared successfully"
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


# Convenience functions for easy agentic AI usage

def create_rag_agent(collection_name: str = "default") -> RAGAgent:
    """
    Create a new RAG agent instance.

    Args:
        collection_name: Name of the document collection

    Returns:
        RAGAgent instance
    """
    return RAGAgent(collection_name)


def insert_document(agent: RAGAgent, pdf_path: str, doc_id: Optional[str] = None,
                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Insert a PDF document into the RAG system.

    Args:
        agent: RAGAgent instance
        pdf_path: Path to the PDF file
        doc_id: Optional custom document ID
        metadata: Optional metadata dictionary

    Returns:
        Dictionary with processing results
    """
    return agent.insert_pdf(pdf_path, doc_id, metadata)


def search_documents(agent: RAGAgent, query: str, k: int = 5,
                    use_reranking: bool = True) -> Dict[str, Any]:
    """
    Search for answers in the processed documents.

    Args:
        agent: RAGAgent instance
        query: The search query
        k: Number of results to return
        use_reranking: Whether to use neural reranking

    Returns:
        Dictionary with search results and answer
    """
    return agent.search(query, k, use_reranking)


def get_document_list(agent: RAGAgent) -> Dict[str, Any]:
    """
    Get list of all documents in the collection.

    Args:
        agent: RAGAgent instance

    Returns:
        Dictionary with document information
    """
    return agent.get_document_info()


# Example usage for agentic AI
if __name__ == "__main__":
    # Create a RAG agent
    agent = create_rag_agent("my_collection")

    # Insert some PDFs
    result1 = insert_document(agent, "path/to/document1.pdf")
    result2 = insert_document(agent, "path/to/document2.pdf", doc_id="custom_id")

    # Search for answers
    search_result = search_documents(agent, "What are the main topics discussed?")
    print("Answer:", search_result['answer'])

    # Get document list
    docs = get_document_list(agent)
    print(f"Total documents: {docs['total_documents']}")