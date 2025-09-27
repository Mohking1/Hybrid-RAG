from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import json
from typing import List, Dict, Any
import time

# Import our custom modules
from pdf_chunker import PDFClusterSemanticChunker
from rag_pipeline import ContextualVectorDB, ElasticsearchBM25, retrieve_advanced, rerank_with_m3
from google import genai
from google.genai import types

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Document RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'es_bm25' not in st.session_state:
    st.session_state.es_bm25 = None
if 'chunker' not in st.session_state:
    st.session_state.chunker = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None

class MultiDocumentRAGSystem:
    def __init__(self):
        self.gemini_client = genai.Client()
        self.chunker = PDFClusterSemanticChunker(
            max_chunk_size=400,
            min_chunk_size=50
        )
        self.vector_db = None
        self.es_bm25 = None
        self.processed_documents = []
        self.metadata_file = "./data/processed_files_metadata.json"
        
    def save_processed_metadata(self, processed_files: Dict[str, Any]):
        """Save processed files metadata to disk."""
        os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
        with open(self.metadata_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
    
    def load_processed_metadata(self) -> Dict[str, Any]:
        """Load processed files metadata from disk."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def load_existing_data(self):
        """Load existing vector DB and ES index if they exist."""
        try:
            # Load vector database
            vector_db_path = "./data/multi_doc_db/contextual_vector_db.pkl"
            if os.path.exists(vector_db_path):
                self.vector_db = ContextualVectorDB("multi_doc_db")
                self.vector_db.load_db()
                print("Loaded existing vector database")
            
            # Load Elasticsearch index
            self.es_bm25 = ElasticsearchBM25("multi_doc_bm25")
            if self.es_bm25.available:
                print("Loaded existing Elasticsearch index")
            
            # Load processed files metadata
            processed_files = self.load_processed_metadata()
            return processed_files
            
        except Exception as e:
            print(f"Error loading existing data: {e}")
            return {}
    
    def delete_all_data(self):
        """Delete all processed data and indices."""
        try:
            # Delete vector database files
            import shutil
            db_dir = "./data/multi_doc_db"
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)
                print("Deleted vector database")
            
            # Delete Elasticsearch index
            if self.es_bm25 and self.es_bm25.available:
                try:
                    self.es_bm25.es_client.indices.delete(index=self.es_bm25.index_name)
                    print("Deleted Elasticsearch index")
                except Exception as e:
                    print(f"Error deleting ES index: {e}")
            
            # Delete metadata file
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
                print("Deleted metadata file")
            
            # Reset instance variables
            self.vector_db = None
            self.es_bm25 = None
            self.processed_documents = []
            
        except Exception as e:
            print(f"Error deleting data: {e}")
            raise
        
    def process_uploaded_files(self, uploaded_files, progress_callback=None):
        """Process multiple uploaded PDF files."""
        temp_dir = tempfile.mkdtemp()
        processed_files = {}
        all_documents = []
        
        try:
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                if progress_callback:
                    progress_callback(f"Processing {uploaded_file.name}...", (i + 1) / total_files)
                
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the PDF
                try:
                    dataset_entry = self.chunker.create_dataset_from_pdf(
                        temp_path, 
                        doc_id=Path(uploaded_file.name).stem
                    )
                    
                    # Add file metadata
                    dataset_entry['filename'] = uploaded_file.name
                    dataset_entry['pdf_path'] = temp_path  # Keep temp path for now
                    
                    all_documents.append(dataset_entry)
                    processed_files[uploaded_file.name] = {
                        'doc_id': dataset_entry['doc_id'],
                        'chunks': len(dataset_entry['chunks']),
                        'status': 'success'
                    }
                    
                except Exception as e:
                    processed_files[uploaded_file.name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            if progress_callback:
                progress_callback("Creating vector database...", 0.8)
            
            # Create vector database
            self.vector_db = ContextualVectorDB("multi_doc_db")
            self.vector_db.load_data(all_documents, parallel_threads=8)
            
            if progress_callback:
                progress_callback("Setting up BM25 search...", 0.9)
            
            # Create BM25 index
            self.es_bm25 = ElasticsearchBM25("multi_doc_bm25")
            self.es_bm25.index_documents(self.vector_db.metadata)
            
            self.processed_documents = all_documents
            
            if progress_callback:
                progress_callback("Complete!", 1.0)
            
            # Save processed files metadata
            self.save_processed_metadata(processed_files)
            
            return processed_files, len(all_documents)
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def search_documents(self, query: str, k: int = 10, use_reranking: bool = True):
        """Search through processed documents."""
        if not self.vector_db or not self.es_bm25:
            raise ValueError("No documents processed yet!")
        
        # Get hybrid search results
        results, semantic_count, bm25_count = retrieve_advanced(
            query, self.vector_db, self.es_bm25, k=k*2  # Get more for reranking
        )
        
        # Apply reranking if requested
        if use_reranking and len(results) > 0:
            # Convert results to the format expected by reranker
            candidate_chunks = [{'metadata': result['chunk']} for result in results]
            reranked_results = rerank_with_m3(query, candidate_chunks, k)
            return reranked_results
        
        # Otherwise return top results
        return results[:k]
    
    def generate_response(self, query: str, search_results: List[Dict], conversation_history: List[Dict] = None):
        """Generate response using Gemini with retrieved context."""
        # Prepare context from search results
        context_parts = []
        citations = []

        for i, result in enumerate(search_results[:15]):  # Use top 15 results
            chunk_data = result.get('chunk', result.get('metadata', {}))
            content = chunk_data.get('original_content', '')
            doc_id = chunk_data.get('doc_id', 'Unknown')
            page_start = chunk_data.get('page_start')
            page_end = chunk_data.get('page_end')
            
            # Extract structured metadata from contextualized_content
            contextualized_content = chunk_data.get('contextualized_content', '')
            
            # Parse structured metadata
            section = None
            chapter = None  
            heading = None
            sub_heading = None
            
            if contextualized_content:
                lines = contextualized_content.split('\n')
                for line in lines:
                    if line.startswith('Section: '):
                        section = line.replace('Section: ', '').strip()
                    elif line.startswith('Chapter: '):
                        chapter = line.replace('Chapter: ', '').strip()
                    elif line.startswith('Heading: '):
                        heading = line.replace('Heading: ', '').strip()
                    elif line.startswith('Sub-Heading: '):
                        sub_heading = line.replace('Sub-Heading: ', '').strip()
            
            # Format page info
            if page_start and page_end:
                if page_start == page_end:
                    page_info = f"p. {page_start}"
                else:
                    page_info = f"pp. {page_start}-{page_end}"
            else:
                page_info = "page unknown"
            
            # Build citation with structured info
            citation_parts = [f"[{i+1}]"]
            if chapter:
                citation_parts.append(chapter)
            elif section:
                citation_parts.append(section)
            if heading and heading != chapter:
                citation_parts.append(f"→ {heading}")
            if sub_heading:
                citation_parts.append(f"→ {sub_heading}")
            citation_parts.append(page_info)
            
            citation = " ".join(citation_parts)
            citations.append(citation)
            
            # Build context with structured metadata
            context_part = f"[Source {i+1}]"
            if chapter:
                context_part += f" ({chapter}"
                if heading and heading != chapter:
                    context_part += f" → {heading}"
                if sub_heading:
                    context_part += f" → {sub_heading}"
                context_part += f", {page_info})"
            elif section:
                context_part += f" ({section}, {page_info})"
            else:
                context_part += f" ({page_info})"
            
            context_part += f": {content}"
            context_parts.append(context_part)
        
        # Prepare conversation context
        conversation_context = ""
        if conversation_history:
            recent_messages = conversation_history[-6:]  # Last 3 exchanges
            for msg in recent_messages:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Create prompt
        context_text = "\n\n".join(context_parts)
        prompt = f"""You are an AI assistant helping users understand documents. Based on the provided context and conversation history, answer the user's question accurately and helpfully.

Context from documents:
{context_text}

Previous conversation:
{conversation_context}

Current question: {query}

Instructions:
1. Answer based solely on the provided context
2. If the context doesn't contain enough information, say so clearly.
3. Include relevant citations from the document like section, header, page number etc, when referencing specific sources.
4. Be conversational and helpful while staying accurate
5. If referring to previous parts of the conversation, make that clear

Answer:"""

        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt]
            )
            
            answer = response.text
            
            # Add citations at the end
            if citations:
                answer += "\n\n**Sources:**\n" + "\n".join(citations)
            
            return answer
            
        except Exception as e:
            return f"Sorry, I encountered an error generating the response: {str(e)}"

def main():
    st.title("📚 Multi-Document RAG Chat")
    st.markdown("Upload multiple PDF documents and chat with them using AI!")
    
    # Check for Gemini API key
    if not os.getenv('GEMINI_API_KEY'):
        st.error("Please set your GEMINI_API_KEY environment variable!")
        st.stop()
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultiDocumentRAGSystem()
    
    # Load existing data on startup
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading existing documents..."):
            existing_processed_files = st.session_state.rag_system.load_existing_data()
            if existing_processed_files:
                st.session_state.processed_files = existing_processed_files
                st.success("✅ Loaded existing processed documents!")
            st.session_state.data_loaded = True
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with"
        )
        
        if uploaded_files:
            # Check if there are existing processed files
            has_existing_data = bool(st.session_state.processed_files)
            
            if has_existing_data:
                st.warning("⚠️ You have existing processed documents. Processing new files will replace them.")
                replace_confirm = st.checkbox("I understand - replace existing documents", value=False)
                process_button_disabled = not replace_confirm
            else:
                process_button_disabled = False
            
            if st.button("🔄 Process Documents", type="primary", disabled=process_button_disabled):
                if len(uploaded_files) > 0:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(message, progress):
                        status_text.text(message)
                        progress_bar.progress(progress)
                    
                    with st.spinner("Processing documents..."):
                        try:
                            processed_files, total_docs = st.session_state.rag_system.process_uploaded_files(
                                uploaded_files, 
                                progress_callback=update_progress
                            )
                            
                            st.session_state.processed_files = processed_files
                            st.success(f"✅ Successfully processed {total_docs} documents!")
                            
                            # Show processing results
                            for filename, info in processed_files.items():
                                if info['status'] == 'success':
                                    st.info(f"📄 {filename}: {info['chunks']} chunks")
                                else:
                                    st.error(f"❌ {filename}: {info['error']}")
                                    
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
                        finally:
                            progress_bar.empty()
                            status_text.empty()
        
        # Show current processed files
        if st.session_state.processed_files:
            st.header("📋 Processed Documents")
            for filename, info in st.session_state.processed_files.items():
                if info['status'] == 'success':
                    st.success(f"✅ {filename} ({info['chunks']} chunks)")
                else:
                    st.error(f"❌ {filename}")
            
            # Delete all data button
            st.header("🗑️ Data Management")
            if st.button("🗑️ Delete All Processed Data", type="secondary", 
                        help="This will permanently delete all processed documents, vector database, and search indices"):
                with st.spinner("Deleting all data..."):
                    try:
                        st.session_state.rag_system.delete_all_data()
                        st.session_state.processed_files = {}
                        st.session_state.messages = []  # Also clear chat history
                        st.success("✅ All data deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting data: {str(e)}")
        
        # Search settings
        st.header("⚙️ Search Settings")
        use_reranking = st.checkbox("Use Reranking", value=True, help="Apply neural reranking for better results")
        num_results = st.slider("Results to retrieve", min_value=3, max_value=15, value=8)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are processed
        if not st.session_state.processed_files or not st.session_state.rag_system.vector_db:
            st.warning("⚠️ Please upload and process documents first!")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Search documents
                    search_results = st.session_state.rag_system.search_documents(
                        prompt, 
                        k=num_results,
                        use_reranking=use_reranking
                    )
                    
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        prompt, 
                        search_results,
                        conversation_history=st.session_state.messages[:-1]  # Exclude current message
                    )
                    
                    st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show search results in expander
                    with st.expander("🔍 Retrieved Context", expanded=False):
                        for i, result in enumerate(search_results[:3]):
                            chunk_data = result.get('chunk', result.get('metadata', {}))
                            content = chunk_data.get('original_content', '')[:300] + "..."
                            doc_id = chunk_data.get('doc_id', 'Unknown')
                            page_info = chunk_data.get('page_start', 'Unknown')
                            score = result.get('score', 0)
                            
                            st.markdown(f"**Result {i+1}** (Score: {score:.3f})")
                            st.markdown(f"*Document: {doc_id}, Page: {page_info}*")
                            st.text(content)
                            st.divider()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()