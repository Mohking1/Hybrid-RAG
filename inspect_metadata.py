#!/usr/bin/env python3
"""
Debug script to inspect chunk metadata from the vector database.
"""

from rag_pipeline import ContextualVectorDB
import json
import os

def inspect_metadata():
    """Load and display metadata from the vector database."""

    # Initialize the database
    db = ContextualVectorDB("multi_doc_db")
    
    # Try to load existing data from disk
    try:
        if os.path.exists(db.db_path):
            print("Loading vector database from disk...")
            db.load_db()
        else:
            print("No vector database file found. Make sure you've processed documents first.")
            return
    except Exception as e:
        print(f"Error loading database: {e}")
        return

    if not db.metadata:
        print("No metadata found in database. Make sure you've processed documents first.")
        return

    print(f"Total chunks: {len(db.metadata)}")
    print("\n" + "="*80)

    # Show metadata for first few chunks
    for i, chunk in enumerate(db.metadata[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print(f"  Document ID: {chunk.get('doc_id', 'N/A')}")
        print(f"  Chunk ID: {chunk.get('chunk_id', 'N/A')}")
        print(f"  Original Index: {chunk.get('original_index', 'N/A')}")
        print(f"  Page Start: {chunk.get('page_start', 'N/A')}")
        print(f"  Page End: {chunk.get('page_end', 'N/A')}")
        print(f"  Page Info: {chunk.get('page_info', 'N/A')}")

        # Show first 200 characters of content
        content = chunk.get('original_content', '')
        print(f"  Content Preview: {content[:200]}{'...' if len(content) > 200 else ''}")

        # Show contextualized content if available
        contextual = chunk.get('contextualized_content', '')
        if contextual:
            print(f"  Contextual Info: {contextual[:200]}{'...' if len(contextual) > 200 else ''}")

        print("-" * 40)

    # Option to save full metadata to JSON
    save_full = input("\nSave full metadata to JSON file? (y/n): ").lower().strip()
    if save_full == 'y':
        filename = "chunk_metadata.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(db.metadata, f, indent=2, ensure_ascii=False)
        print(f"Full metadata saved to {filename}")

if __name__ == "__main__":
    inspect_metadata()